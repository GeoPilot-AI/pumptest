"""Do the reported standard errors mean what they say?

Synthetic tests with known parameters and noise, refitted many times. With
independent noise the 95 percent intervals (estimate +/- 1.96 standard
errors) must cover the truth about 95 percent of the time and the estimates
must be unbiased. With strongly correlated noise, as real drawdown records
have, the same intervals are too narrow: the last test pins that limitation
down so a change to the uncertainty method has to update it knowingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumptest import HantushJacob, Model, Observation, PumpingTest, Schedule, Theis, Well, fit

TIMES = np.logspace(-4, 0, 30)
DISTANCES = (20.0, 60.0)
NOISE = 0.005
CASES = [
    (Theis(), {"T": 300.0, "S": 2e-4}),
    (HantushJacob(), {"T": 300.0, "S": 2e-4, "B": 600.0}),
]


def _noisy(
    model: Model, truth: dict[str, float], rho: float, rng: np.random.Generator
) -> PumpingTest:
    well, schedule = Well(0.1), Schedule.constant(500.0)
    obs = []
    for r in DISTANCES:
        template = PumpingTest(well, schedule, [Observation("o", r, TIMES, np.ones_like(TIMES))])
        clean = model.drawdown(truth, template, template.observations[0])
        # AR(1) noise with stationary standard deviation NOISE.
        e = np.empty(TIMES.size)
        e[0] = rng.normal(0.0, NOISE)
        for i in range(1, TIMES.size):
            e[i] = rho * e[i - 1] + rng.normal(0.0, NOISE * np.sqrt(1.0 - rho**2))
        obs.append(Observation(f"r{r:g}", r, TIMES, clean + e))
    return PumpingTest(well, schedule, obs)


def _coverage(
    model: Model, truth: dict[str, float], rho: float, reps: int = 100
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    hits = dict.fromkeys(truth, 0)
    errors: dict[str, list[float]] = {k: [] for k in truth}
    for _ in range(reps):
        result = fit(_noisy(model, truth, rho, rng), model)
        for k, v in truth.items():
            hits[k] += abs(result.values[k] - v) <= 1.96 * result.standard_errors[k]
            errors[k].append(result.values[k] / v - 1.0)
    bias = {f"bias {k}": float(np.median(e)) for k, e in errors.items()}
    return {**{k: h / reps for k, h in hits.items()}, **bias}


@pytest.mark.parametrize(("model", "truth"), CASES, ids=lambda c: getattr(c, "name", ""))
def test_intervals_cover_the_truth_with_independent_noise(
    model: Model, truth: dict[str, float]
) -> None:
    result = _coverage(model, truth, rho=0.0)
    for k in truth:
        assert 0.88 <= result[k] <= 1.0, (k, result[k])
        assert abs(result[f"bias {k}"]) < 0.01, (k, result[f"bias {k}"])


def test_intervals_are_too_narrow_with_correlated_noise() -> None:
    result = _coverage(Theis(), {"T": 300.0, "S": 2e-4}, rho=0.9)
    assert result["T"] < 0.7
    assert result["S"] < 0.7
