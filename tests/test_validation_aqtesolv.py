"""The package against AQTESOLV, on the tests published in its manual.

Two criteria per test. The parameters must land close to the published ones;
and our optimum must fit the data at least as well as the published parameters
do when evaluated on the same data here. The second is the stronger check:
the RMSE values printed beside AQTESOLV results were computed on that program's
own copy of the data, which we do not have, so they are not comparable digit
for digit, but a fit that is worse than theirs on our data would be a real
defect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pumptest import HantushJacob, Model, PumpingTest, Theis, fit


def rmse_at(test: PumpingTest, model: Model, values: dict[str, float]) -> float:
    res = np.concatenate([o.drawdown - model.drawdown(values, test, o) for o in test.observations])
    return float(np.sqrt(np.mean(res**2)))


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
    # AQTESOLV Hantush-Jacob (Yang 2020, Table 18): K = 224.723 m/d, Ss = 2.125e-4 /m,
    # c = 43.965 d, with the casing radius fixed at 0.5 ft.
    t_pub = 224.723 * 15.24
    published = {"T": t_pub, "S": 2.125e-4 * 15.24, "B": math.sqrt(t_pub * 43.965)}
    result = fit(texas_hill, HantushJacob())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.002)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.01)
    assert result.derived["c"] == pytest.approx(43.965, rel=0.01)
    assert result.rmse <= rmse_at(texas_hill, HantushJacob(), published) + 1e-9
