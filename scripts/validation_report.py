"""Generate docs/validation.md from the published tests, so its numbers are measured.

    python scripts/validation_report.py          # print the page
    python scripts/validation_report.py --write  # rewrite docs/validation.md

``tests/test_validation_report.py`` fails when the committed page and a fresh
run disagree, so the documentation cannot drift from the code.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pumptest import (  # noqa: E402
    HantushJacob,
    Model,
    Neuman,
    Observation,
    PapadopulosCooper,
    PumpingTest,
    Schedule,
    Theis,
    Well,
    cooper_jacob,
    fit,
    step_drawdown,
)

DATA = REPO / "tests" / "data" / "ttim"
KDR = REPO / "tests" / "data" / "kdr"
DAY_S = 86400.0
PAGE = REPO / "docs" / "validation.md"
FT = 0.3048


def _series(name: str, *, minutes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(DATA / name, comments="#")
    t = raw[:, 0] / 1440.0 if minutes else raw[:, 0]
    return t, np.abs(raw[:, 1])


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    test: PumpingTest
    model: Model
    published: dict[str, float]


def cases() -> list[Case]:
    """The five published tests, in metres and days."""

    ok = PumpingTest(
        Well(0.2),
        Schedule.constant(788.0),
        [
            Observation("H30", 30.0, *_series("oudekorendijk_h30.dat", minutes=True)),
            Observation("H90", 90.0, *_series("oudekorendijk_h90.dat", minutes=True)),
        ],
        thickness=7.0,
    )
    gridley = PumpingTest(
        Well(0.1524),
        Schedule.constant(1199.218),
        [
            Observation("Well 1", 251.1552, *_series("gridley_well_1.txt")),
            Observation("Well 3 (pumped)", 0.0, *_series("gridley_well_3.txt")),
        ],
        thickness=5.4846,
    )
    sioux = PumpingTest(
        Well(0.1524),
        Schedule.constant(6605.754),
        [Observation(f"OW{d}", d * FT, *_series(f"sioux{d}.txt")) for d in (100, 200, 400)],
        thickness=15.24,
    )
    texas = PumpingTest(
        Well(0.1524),
        Schedule.constant(24464.06),
        [
            Observation("OW1", 12.191, *_series("texas40.txt")),
            Observation("OW2", 24.383, *_series("texas80.txt")),
            Observation("OW3", 48.766, *_series("texas160.txt")),
        ],
        thickness=15.24,
    )
    dalem = PumpingTest(
        Well(0.1),
        Schedule.constant(761.0, stop=0.34),
        [Observation(f"P{d}", float(d), *_series(f"dalem_p{d}.txt")) for d in (30, 60, 90, 120)],
        thickness=37.0,
    )
    t_texas, t_dalem = 224.726 * 15.24, 45.332 * 37.0
    return [
        Case(
            "Oude Korendijk",
            "AQTESOLV",
            ok,
            Theis(),
            {"T": 66.086 * 7.0, "S": 2.541e-5 * 7.0},
        ),
        Case(
            "Gridley",
            "AQTESOLV",
            gridley,
            Theis(),
            {"T": 37.803 * 5.4846, "S": 1.356e-6 * 5.4846},
        ),
        Case(
            "Sioux Flats",
            "AQTESOLV",
            sioux,
            Theis(),
            {"T": 282.659 * 15.24, "S": 4.211e-3 * 15.24},
        ),
        Case(
            "Texas Hill",
            "AQTESOLV",
            texas,
            HantushJacob(),
            {"T": t_texas, "S": 2.125e-4 * 15.24, "B": math.sqrt(t_texas * 43.964)},
        ),
        Case(
            "Dalem",
            "Kruseman and de Ridder",
            dalem,
            HantushJacob(),
            {"T": t_dalem, "S": 4.762e-5 * 37.0, "B": math.sqrt(t_dalem * 331.141)},
        ),
    ]


def _rmse(case: Case, values: dict[str, float]) -> float:
    res = np.concatenate(
        [o.drawdown - case.model.drawdown(values, case.test, o) for o in case.test.observations]
    )
    return float(np.sqrt(np.mean(res**2)))


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _c(values: dict[str, float]) -> str:
    return _fmt(values["B"] ** 2 / values["T"]) if "B" in values else ""


def textbook_rows() -> list[str]:
    """The Kruseman and de Ridder worked examples, one row per published quantity."""

    rows: list[str] = []
    steps = np.loadtxt(KDR / "well1_steps.txt", comments="#")
    q, s = steps[:, 0], steps[:, 1]
    st = step_drawdown(q, s)
    b, c = st.aquifer_loss_coefficient, st.well_loss_coefficient

    def miss(bb: float, cc: float) -> float:
        return float(np.sqrt(np.mean((s - (bb * q + cc * q * q)) ** 2)))

    rows.append(
        f"| Well 1 step test | B (d/m2) | {_fmt(3.26e-3)} | {_fmt(b)} "
        f"| drawdown RMSE {_fmt(miss(3.26e-3, 1.45e-7))} m published, {_fmt(miss(b, c))} m here |"
    )
    rows.append(f"| Well 1 step test | C (d2/m5) | {_fmt(1.45e-7)} | {_fmt(c)} | same fit |")

    raw = np.loadtxt(KDR / "hoogezand.txt", comments="#")
    t, sw, rate = raw[:, 0] / DAY_S, raw[:, 1], 28.7 * 24.0
    jacob = cooper_jacob(t, sw, rate, 0.185, start=41.0 / DAY_S)
    rows.append(
        f"| Hoogezand single well | T, Jacob (m2/d) | 1800 | {_fmt(jacob.transmissivity)} "
        f"| slope {_fmt(0.07)} m/cycle read by eye, {_fmt(jacob.line.slope)} by regression |"
    )
    late = t >= 45.0 / DAY_S
    hz = PumpingTest(
        Well(0.185, casing_radius=1e-3),
        Schedule.constant(rate),
        [Observation("pumped well", 0.0, t[late], sw[late])],
    )
    skin = fit(hz, PapadopulosCooper(skin=True), fixed={"S": 4e-4})
    rows.append(
        f"| Hoogezand single well | T with skin, S = 4e-4 (m2/d) | not published "
        f"| {_fmt(skin.values['T'])} | skin {_fmt(skin.values['Sk'])}, "
        f"RMSE {_fmt(skin.rmse)} m |"
    )

    full = PumpingTest(
        Well(0.185),
        Schedule.constant(rate),
        [Observation("pumped well", 0.0, t[t >= 5.0 / DAY_S], sw[t >= 5.0 / DAY_S])],
    )
    rcfit = fit(full, PapadopulosCooper(skin=True, casing_radius=True), fixed={"S": 4e-4})
    rows.append(
        f"| Hoogezand single well | T, full record, fitted rc (m2/d) | 1800 "
        f"| {_fmt(rcfit.values['T'])} | rc {_fmt(rcfit.values['rc'])} m of 0.185, skin "
        f"{_fmt(rcfit.values['Sk'])}, RMSE {_fmt(rcfit.rmse)} m |"
    )

    vb = np.loadtxt(DATA / "venne_deep.txt", comments="#")
    vb = vb[vb[:, 0] > 0]
    venne = PumpingTest(
        Well(0.2),
        Schedule.constant(873.0),
        [Observation("deep", 90.0, vb[:, 0] / 1440.0, np.abs(vb[:, 1]))],
        thickness=21.0,
    )
    nm = fit(venne, Neuman())
    th, hj = fit(venne, Theis()), fit(venne, HantushJacob())
    rows.append(
        f"| Vennebulten unconfined | T, Neuman (m2/d) | 1447 to 1616 | {_fmt(nm.values['T'])} "
        f"| RMSE {_fmt(nm.rmse)} m; Theis {_fmt(th.rmse)}, Hantush {_fmt(hj.rmse)} |"
    )
    rows.append(f"| Vennebulten unconfined | S | 5.2e-4 | {_fmt(nm.values['S'])} | |")
    rows.append(
        f"| Vennebulten unconfined | Sy | 4.9e-3 (the book: unreasonably low) "
        f"| {_fmt(nm.values['Sy'])} | |"
    )
    kd_pub = 0.01 * 21.0**2 / 90.0**2
    rows.append(
        f"| Vennebulten unconfined | Kd = Kz/Kr | {_fmt(kd_pub)} | {_fmt(nm.values['Kd'])} | |"
    )
    return rows


def page() -> str:
    rows = []
    for case in cases():
        ours = fit(case.test, case.model).values
        pub = case.published
        rows.append(
            f"| {case.name} | {case.model.name} | {case.source} "
            f"| {_fmt(pub['T'])} | {_fmt(ours['T'])} "
            f"| {_fmt(pub['S'])} | {_fmt(ours['S'])} "
            f"| {_c(pub)} | {_c(ours)} "
            f"| {_fmt(_rmse(case, pub))} | {_fmt(_rmse(case, ours))} |"
        )
    header = [
        "# Validation",
        "",
        "Generated by `python scripts/validation_report.py --write`; a test fails",
        "when this page and a fresh run disagree.",
        "",
        "Each published test is fitted here and compared with the parameters",
        "AQTESOLV or Kruseman and de Ridder published for it. T is in square metres",
        "per day, c (aquitard resistance) in days. Both RMSE columns are computed",
        "here, on the same data, in metres: the published parameters' fit and ours.",
        "",
        "| Test | Model | Published by | T published | T here | S published | S here "
        "| c published | c here | RMSE published | RMSE here |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    footer = [
        "",
        "Data: the TTim documentation (MIT), see `tests/data/ttim/README.md`. Published",
        "values: Yang (2020), as tabulated in the TTim notebooks.",
        "",
        "The well functions are also checked against published tables: Theis",
        "against Abramowitz and Stegun (Table 5.1), and Hantush, Neuman and",
        "Papadopulos-Cooper against Kruseman and de Ridder (1994), Annexes 4.2, 5.1",
        "and 15.1. See `tests/test_type_curves.py`.",
        "",
        "## Worked examples from Kruseman and de Ridder (1994)",
        "",
        "Their answers come from lines and type curves fitted by eye, so they are",
        "compared within what a graph resolves. Data transcribed in `tests/data/kdr`.",
        "",
        "| Example | Quantity | Published | Here | Notes |",
        "|---|---|---|---|---|",
        *textbook_rows(),
        "",
    ]
    return "\n".join([*header, *rows, *footer])


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
