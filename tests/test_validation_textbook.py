"""Worked examples from Kruseman and de Ridder (1994), against their published answers.

The book's answers come from hand-drawn lines and type-curve overlays, so the
parameters are compared within the accuracy of reading a graph, and wherever
both can be measured the stronger check is applied: our estimate must fit
the data at least as well as the published one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pumptest import (
    HantushJacob,
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

DATA = Path(__file__).parent / "data"
SECONDS_PER_DAY = 86400.0


def test_well1_step_drawdown_fits_better_than_the_published_line() -> None:
    raw = np.loadtxt(DATA / "kdr" / "well1_steps.txt", comments="#")
    q, s = raw[:, 0], raw[:, 1]
    result = step_drawdown(q, s)
    ours = np.sqrt(
        np.mean(
            (s - (result.aquifer_loss_coefficient * q + result.well_loss_coefficient * q**2)) ** 2
        )
    )
    published = np.sqrt(np.mean((s - (3.26e-3 * q + 1.45e-7 * q**2)) ** 2))
    assert ours < published
    assert result.aquifer_loss_coefficient == pytest.approx(3.26e-3, rel=0.03)
    assert result.well_loss_coefficient == pytest.approx(1.45e-7, rel=0.1)


def _hoogezand() -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.loadtxt(DATA / "kdr" / "hoogezand.txt", comments="#")
    return raw[:, 0] / SECONDS_PER_DAY, raw[:, 1], 28.7 * 24.0


def test_hoogezand_jacob_transmissivity() -> None:
    # The book reads a slope of 0.07 m per log cycle by eye and gets 1800 m2/d;
    # the regression slope over the same points (t > 41 s) is 0.0655 m.
    t, s, rate = _hoogezand()
    result = cooper_jacob(t, s, rate, 0.185, start=41.0 / SECONDS_PER_DAY)
    assert result.transmissivity == pytest.approx(1800.0, rel=0.08)
    assert result.line.slope == pytest.approx(0.07, rel=0.08)


def test_hoogezand_skin_with_storativity_held_at_mulders_value() -> None:
    # A single pumped well cannot separate S from skin, so S is fixed at
    # Mulder's 4e-4. The record shows no casing storage (1.06 m of drawdown
    # after 5 s, more than the 0.185 m casing could supply), so storage is
    # switched off with a negligible casing radius and the fit is limited to
    # the semi-log period the book uses.
    t, s, rate = _hoogezand()
    late = t >= 45.0 / SECONDS_PER_DAY
    test = PumpingTest(
        Well(0.185, casing_radius=1e-3),
        Schedule.constant(rate),
        [Observation("pumped well", 0.0, t[late], s[late])],
    )
    result = fit(test, PapadopulosCooper(skin=True), fixed={"S": 4e-4})
    jacob = cooper_jacob(t, s, rate, 0.185, start=41.0 / SECONDS_PER_DAY)
    assert result.values["T"] == pytest.approx(jacob.transmissivity, rel=0.05)
    assert 20.0 < result.values["Sk"] < 35.0
    assert result.rmse < 0.01


@pytest.fixture
def vennebulten() -> PumpingTest:
    raw = np.loadtxt(DATA / "ttim" / "venne_deep.txt", comments="#")
    raw = raw[raw[:, 0] > 0]
    obs = Observation("WI1/90 deep", 90.0, raw[:, 0] / 1440.0, np.abs(raw[:, 1]))
    return PumpingTest(Well(0.2), Schedule.constant(873.0), [obs], thickness=21.0)


def test_vennebulten_neuman(vennebulten: PumpingTest) -> None:
    result = fit(vennebulten, Neuman())
    # The book's type-A and type-B matches bracket T at 1447 and 1616 m2/d
    # and give SA = 5.2e-4; our T sits 8 percent under the type-A value,
    # within what an overlay by eye resolves.
    assert 1250.0 < result.values["T"] < 1616.0
    assert result.values["S"] == pytest.approx(5.2e-4, rel=0.15)
    # Kd from the book's beta = 0.01: beta * b**2 / r**2.
    assert result.values["Kd"] == pytest.approx(0.01 * 21.0**2 / 90.0**2, rel=0.5)
    assert result.values["Sy"] > 4.9e-3


def test_vennebulten_picks_the_unconfined_model(vennebulten: PumpingTest) -> None:
    rmse = {m.name: fit(vennebulten, m).rmse for m in (Theis(), HantushJacob(), Neuman())}
    assert rmse["Neuman"] < rmse["Hantush-Jacob"] < rmse["Theis"]


def test_hoogezand_full_record_with_fitted_casing_radius() -> None:
    # Fitting the effective casing radius explains the early record too, and
    # lands on the book's transmissivity with rc about half the casing: the
    # pump column takes up the rest.
    t, s, rate = _hoogezand()
    keep = t >= 5.0 / SECONDS_PER_DAY
    test = PumpingTest(
        Well(0.185), Schedule.constant(rate), [Observation("pumped well", 0.0, t[keep], s[keep])]
    )
    model = PapadopulosCooper(skin=True, casing_radius=True)
    result = fit(test, model, fixed={"S": 4e-4})
    assert result.values["T"] == pytest.approx(1800.0, rel=0.05)
    assert 0.05 < result.values["rc"] < 0.185
    assert result.rmse < 0.015
