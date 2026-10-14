"""The package against AQTESOLV, MLU and the textbook, on published tests.

Two criteria per test. The parameters must land close to the published ones;
and our optimum must fit the data at least as well as the published parameters
do when evaluated on the same data here. The second is the stronger check:
the RMSE values printed beside AQTESOLV results were computed on that program's
own copy of the data (feet and minutes, rounded), so they are not comparable
digit for digit, but a fit that is worse than theirs on our data would be a
real defect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pumptest import HantushJacob, Model, PumpingTest, Theis, fit


def rmse_at(test: PumpingTest, model: Model, values: dict[str, float]) -> float:
    res = np.concatenate([o.drawdown - model.drawdown(values, test, o) for o in test.observations])
    return float(np.sqrt(np.mean(res**2)))


def test_oude_korendijk_matches_aqtesolv(oude_korendijk: PumpingTest) -> None:
    # AQTESOLV: K = 66.086 m/d, Ss = 2.541e-5 /m over b = 7 m, RMSE 0.05006 m.
    published = {"T": 66.086 * 7.0, "S": 2.541e-5 * 7.0}
    result = fit(oude_korendijk, Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.002)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.002)
    assert result.rmse == pytest.approx(0.05006, abs=5e-5)
    assert result.derived["K"] == pytest.approx(66.086, rel=0.002)


def test_sioux_matches_aqtesolv(sioux: PumpingTest) -> None:
    published = {"T": 282.659 * 15.24, "S": 4.211e-3 * 15.24}
    result = fit(sioux, Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.002)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.002)
    assert result.rmse <= rmse_at(sioux, Theis(), published) + 1e-9


def test_gridley_with_pumping_well_matches_aqtesolv(gridley: PumpingTest) -> None:
    # Fitted to the observation well and the pumped well together. S is poorly
    # constrained here (about 16 percent standard error), so it is checked
    # against the published value to within its own uncertainty.
    published = {"T": 37.803 * 5.4846, "S": 1.356e-6 * 5.4846}
    result = fit(gridley, Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.01)
    assert abs(result.values["S"] - published["S"]) < result.standard_errors["S"]
    assert result.rmse <= rmse_at(gridley, Theis(), published) + 1e-9


def test_texas_hill_leaky_matches_aqtesolv(texas_hill: PumpingTest) -> None:
    # AQTESOLV Hantush-Jacob: K = 224.726 m/d, Ss = 2.125e-4 /m, c = 43.964 d.
    t_pub = 224.726 * 15.24
    published = {"T": t_pub, "S": 2.125e-4 * 15.24, "B": math.sqrt(t_pub * 43.964)}
    result = fit(texas_hill, HantushJacob())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.002)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.01)
    assert result.derived["c"] == pytest.approx(43.964, rel=0.01)
    assert result.rmse <= rmse_at(texas_hill, HantushJacob(), published) + 1e-9


def test_dalem_leaky_reproduces_the_textbook(dalem: PumpingTest) -> None:
    # Kruseman and de Ridder's Hantush analysis: K = 45.332 m/d, Ss = 4.762e-5,
    # c = 331.141 d, which is also the lowest RMSE (0.005917 m) any program in
    # the comparison reports for this model.
    t_pub = 45.332 * 37.0
    published = {"T": t_pub, "S": 4.762e-5 * 37.0, "B": math.sqrt(t_pub * 331.141)}
    result = fit(dalem, HantushJacob())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.002)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.005)
    assert result.derived["c"] == pytest.approx(331.141, rel=0.005)
    assert result.rmse <= rmse_at(dalem, HantushJacob(), published) + 1e-9
    assert result.rmse == pytest.approx(0.005917, abs=2e-6)
