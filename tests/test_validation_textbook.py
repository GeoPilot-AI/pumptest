"""Worked examples from Kruseman and de Ridder (1994), against their published answers.

The book's answers come from hand-drawn lines and type-curve overlays, so the
parameters are compared within the accuracy of reading a graph, and wherever
both can be measured the stronger check is applied: our estimate must fit
the data at least as well as the published one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumptest import (
    HantushJacob,
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

SECOND = 1.0 / 86400.0


def test_well1_step_drawdown_fits_better_than_the_published_line() -> None:
    q, s = datasets.step_test()
    result = step_drawdown(q, s)
    ours = np.sqrt(np.mean((s - result.drawdown_curve(q)) ** 2))
    published = np.sqrt(np.mean((s - (3.26e-3 * q + 1.45e-7 * q**2)) ** 2))
    assert ours < published
    assert result.aquifer_loss_coefficient == pytest.approx(3.26e-3, rel=0.03)
    assert result.well_loss_coefficient == pytest.approx(1.45e-7, rel=0.1)


def _from(test: PumpingTest, seconds: float, well: Well | None = None) -> PumpingTest:
    obs = test.observations[0]
    keep = obs.time >= seconds * SECOND
    return PumpingTest(
        well or test.well,
        test.schedule,
        [Observation(obs.name, 0.0, obs.time[keep], obs.drawdown[keep])],
    )


def test_hoogezand_jacob_transmissivity(hoogezand: PumpingTest) -> None:
    # The book reads a slope of 0.07 m per log cycle by eye and gets 1800 m2/d;
    # the regression slope over the same points (t > 41 s) is 0.0655 m.
    obs = hoogezand.observations[0]
    rate = hoogezand.schedule.first_rate
    result = cooper_jacob(obs.time, obs.drawdown, rate, 0.185, start=41.0 * SECOND)
    assert result.transmissivity == pytest.approx(1800.0, rel=0.08)
    assert result.line.slope == pytest.approx(0.07, rel=0.08)


def test_hoogezand_skin_with_storativity_held_at_mulders_value(hoogezand: PumpingTest) -> None:
    # A single pumped well cannot separate S from skin, so S is fixed at
    # Mulder's 4e-4. Casing storage is switched off with a negligible casing
    # radius and the fit is limited to the semi-log period the book uses.
    test = _from(hoogezand, 45.0, Well(0.185, casing_radius=1e-3))
    result = fit(test, PapadopulosCooper(skin=True), fixed={"S": 4e-4})
    obs = hoogezand.observations[0]
    jacob = cooper_jacob(
        obs.time, obs.drawdown, hoogezand.schedule.first_rate, 0.185, start=41.0 * SECOND
    )
    assert result.values["T"] == pytest.approx(jacob.transmissivity, rel=0.05)
    assert 20.0 < result.values["Sk"] < 35.0
    assert result.rmse < 0.01


def test_hoogezand_full_record_with_fitted_casing_radius(hoogezand: PumpingTest) -> None:
    # The record shows less casing storage than a 0.185 m casing implies (1.06 m
    # of drawdown after 5 s). Fitting the effective casing radius explains the
    # early record too and lands on the book's transmissivity, with rc about
    # half the casing: the pump column takes up the rest.
    test = _from(hoogezand, 5.0)
    model = PapadopulosCooper(skin=True, casing_radius=True)
    result = fit(test, model, fixed={"S": 4e-4})
    assert result.values["T"] == pytest.approx(1800.0, rel=0.05)
    assert 0.05 < result.values["rc"] < 0.185
    assert result.rmse < 0.015


def test_vennebulten_neuman(vennebulten: PumpingTest) -> None:
    result = fit(vennebulten, Neuman())
    # The book's type-A and type-B matches bracket T at 1447 and 1616 m2/d
    # and give SA = 5.2e-4; our T sits 8 percent under the type-A value,
    # within what an overlay by eye resolves.
    published = datasets.published("vennebulten").values
    assert 1250.0 < result.values["T"] < 1616.0
    assert result.values["S"] == pytest.approx(published["S"], rel=0.15)
    assert result.values["Kd"] == pytest.approx(published["Kd"], rel=0.5)
    assert result.values["Sy"] > published["Sy"]


def test_vennebulten_picks_the_unconfined_model(vennebulten: PumpingTest) -> None:
    rmse = {m.name: fit(vennebulten, m).rmse for m in (Theis(), HantushJacob(), Neuman())}
    assert rmse["Neuman"] < rmse["Hantush-Jacob"] < rmse["Theis"]
