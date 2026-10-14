"""Generate docs/validation.md from the bundled published tests.

    python scripts/validation_report.py          # print the page
    python scripts/validation_report.py --write  # rewrite docs/validation.md

``tests/test_docs.py`` fails when the committed page and a fresh run
disagree, so the documentation cannot drift from the code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pumptest import (  # noqa: E402
    MODELS,
    Model,
    Neuman,
    Observation,
    PapadopulosCooper,
    PumpingTest,
    Theis,
    Well,
    cooper_jacob,
    datasets,
    fit,
    step_drawdown,
)
from pumptest.models import HantushJacob  # noqa: E402

PAGE = REPO / "docs" / "validation.md"
SECOND = 1.0 / 86400.0
FITTED = ("oude_korendijk", "gridley", "sioux_flats", "texas_hill", "dalem")
"""The tests whose published parameters come from a least-squares program
or a full type-curve analysis of every well, so a parameter comparison is fair."""


def model_for(name: str) -> Model:
    key = datasets.published(name).model.lower()
    return MODELS[key]()


def _rmse(test: PumpingTest, model: Model, values: dict[str, float]) -> float:
    res = np.concatenate([o.drawdown - model.drawdown(values, test, o) for o in test.observations])
    return float(np.sqrt(np.mean(res**2)))


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _c(values: dict[str, float]) -> str:
    return _fmt(values["B"] ** 2 / values["T"]) if "B" in values else ""


def published_rows() -> list[str]:
    rows = []
    for name in FITTED:
        test, pub = datasets.load(name), datasets.published(name)
        model = model_for(name)
        ours = fit(test, model).values
        p = pub.values
        rows.append(
            f"| {test.name} | {model.name} | {pub.source.split(' (')[0]} "
            f"| {_fmt(p['T'])} | {_fmt(ours['T'])} "
            f"| {_fmt(p['S'])} | {_fmt(ours['S'])} "
            f"| {_c(p)} | {_c(ours)} "
            f"| {_fmt(_rmse(test, model, p))} | {_fmt(_rmse(test, model, ours))} |"
        )
    return rows


def _from(test: PumpingTest, seconds: float, well: Well | None = None) -> PumpingTest:
    obs = test.observations[0]
    # Times are whole seconds stored in days; cutting half a second early keeps
    # the point at the cut whichever way its conversion rounded.
    keep = obs.time >= (seconds - 0.5) * SECOND
    return PumpingTest(
        well or test.well,
        test.schedule,
        [Observation(obs.name, 0.0, obs.time[keep], obs.drawdown[keep])],
    )


def textbook_rows() -> list[str]:
    """The Kruseman and de Ridder worked examples, one row per published quantity."""

    rows: list[str] = []
    q, s = datasets.step_test()
    st = step_drawdown(q, s)

    def miss(curve: np.ndarray) -> float:
        return float(np.sqrt(np.mean((s - curve) ** 2)))

    book = miss(3.26e-3 * q + 1.45e-7 * q * q)
    rows.append(
        f"| Well 1 step test | B (d/m2) | {_fmt(3.26e-3)} | {_fmt(st.aquifer_loss_coefficient)} "
        f"| drawdown RMSE {_fmt(book)} m published, {_fmt(miss(st.drawdown_curve(q)))} m here |"
    )
    rows.append(
        f"| Well 1 step test | C (d2/m5) | {_fmt(1.45e-7)} "
        f"| {_fmt(st.well_loss_coefficient)} | same fit |"
    )

    hz = datasets.load("hoogezand")
    obs, rate = hz.observations[0], hz.schedule.first_rate
    jacob = cooper_jacob(obs.time, obs.drawdown, rate, 0.185, start=41.0 * SECOND)
    rows.append(
        f"| Hoogezand single well | T, Jacob (m2/d) | 1800 | {_fmt(jacob.transmissivity)} "
        f"| slope 0.07 m/cycle read by eye, {_fmt(jacob.line.slope)} by regression |"
    )
    skin = fit(
        _from(hz, 45.0, Well(0.185, casing_radius=1e-3)),
        PapadopulosCooper(skin=True),
        fixed={"S": 4e-4},
    )
    rows.append(
        f"| Hoogezand single well | T with skin, S = 4e-4 (m2/d) | not published "
        f"| {_fmt(skin.values['T'])} | skin {_fmt(skin.values['Sk'])}, "
        f"RMSE {_fmt(skin.rmse)} m |"
    )
    rcfit = fit(_from(hz, 5.0), PapadopulosCooper(skin=True, casing_radius=True), fixed={"S": 4e-4})
    rows.append(
        f"| Hoogezand single well | T, full record, fitted rc (m2/d) | 1800 "
        f"| {_fmt(rcfit.values['T'])} | rc {_fmt(rcfit.values['rc'])} m of 0.185, skin "
        f"{_fmt(rcfit.values['Sk'])}, RMSE {_fmt(rcfit.rmse)} m |"
    )

    venne = datasets.load("vennebulten")
    vpub = datasets.published("vennebulten").values
    nm = fit(venne, Neuman())
    th, hj = fit(venne, Theis()), fit(venne, HantushJacob())
    rows.append(
        f"| Vennebulten unconfined | T, Neuman (m2/d) | 1447 to 1616 | {_fmt(nm.values['T'])} "
        f"| RMSE {_fmt(nm.rmse)} m; Theis {_fmt(th.rmse)}, Hantush {_fmt(hj.rmse)} |"
    )
    rows.append(f"| Vennebulten unconfined | S | {_fmt(vpub['S'])} | {_fmt(nm.values['S'])} | |")
    rows.append(
        f"| Vennebulten unconfined | Sy | {_fmt(vpub['Sy'])} (the book: unreasonably low) "
        f"| {_fmt(nm.values['Sy'])} | |"
    )
    rows.append(
        f"| Vennebulten unconfined | Kd = Kz/Kr | {_fmt(vpub['Kd'])} | {_fmt(nm.values['Kd'])} | |"
    )
    return rows


def page() -> str:
    lines = [
        "# Validation",
        "",
        "Generated by `python scripts/validation_report.py --write`; a test fails",
        "when this page and a fresh run disagree.",
        "",
        "![Measured drawdown and fitted solutions](assets/fits.png)",
        "",
        "## Against AQTESOLV and published analyses",
        "",
        "Each published test is fitted here and compared with the parameters",
        "AQTESOLV or Kruseman and de Ridder published for it. T is in square metres",
        "per day, c (aquitard resistance) in days. Both RMSE columns are computed",
        "here, on the same data, in metres: the published parameters' fit and ours.",
        "",
        "| Test | Model | Published by | T published | T here | S published | S here "
        "| c published | c here | RMSE published | RMSE here |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *published_rows(),
        "",
        "![Published against pumptest parameters](assets/parity.png)",
        "",
        "Data: the TTim documentation (MIT); published values: Yang (2020), as",
        "tabulated in the TTim notebooks. Provenance is in",
        "`src/pumptest/datasets/data/README.md`.",
        "",
        "## Against the published type-curve tables",
        "",
        "Theis is checked against the exponential integral summed from its power",
        "series in 60-digit decimal arithmetic, and Hantush, Neuman and",
        "Papadopulos-Cooper against Kruseman and de Ridder (1994), Annexes 4.2, 5.1",
        "and 15.1, within the tables' three-figure rounding.",
        "See `tests/test_type_curves.py`.",
        "",
        "## Worked examples from Kruseman and de Ridder (1994)",
        "",
        "Their answers come from lines and type curves fitted by eye, so they are",
        "compared within what a graph resolves.",
        "",
        "| Example | Quantity | Published | Here | Notes |",
        "|---|---|---|---|---|",
        *textbook_rows(),
        "",
        "![Single-well and step-drawdown tests](assets/single_well.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite docs/validation.md")
    args = parser.parse_args()
    text = page()
    if args.write:
        PAGE.write_text(text, encoding="utf-8")
        print(f"wrote {PAGE.relative_to(REPO)}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
