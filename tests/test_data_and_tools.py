"""The test description, Laplace inversion, units and the derivative."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pumptest import (
    MODELS,
    Observation,
    PumpingTest,
    Schedule,
    Theis,
    Well,
    log_derivative,
    theis_w,
)
from pumptest.laplace import invert, stehfest_weights
from pumptest.units import convert, drawdown_from_levels


def test_schedule_steps_and_rates() -> None:
    steps = Schedule(((0.0, 100.0), (1.0, 200.0), (2.0, 0.0)))
    assert steps.first_rate == 100.0
    assert [steps.rate_at(t) for t in (-1.0, 0.5, 1.5, 3.0)] == [0.0, 100.0, 200.0, 0.0]
    assert Schedule.constant(5.0, stop=2.0).steps == ((0.0, 5.0), (2.0, 0.0))
    with pytest.raises(ValueError, match="at least one"):
        Schedule(())
    with pytest.raises(ValueError, match="increasing"):
        Schedule(((0.0, 1.0), (0.0, 2.0)))


def test_superposition_of_a_step_test() -> None:
    # Two equal steps of 100 equal one Theis response plus a delayed copy.
    schedule = Schedule(((0.0, 100.0), (1.0, 200.0), (1.5, 200.0)))
    t = np.array([0.5, 2.0])

    def unit(times: np.ndarray) -> np.ndarray:
        return theis_w(1e-3 / times)

    expected = 100.0 * unit(t) + 100.0 * np.where(t > 1.0, unit(np.maximum(t - 1.0, 1e-9)), 0.0)
    assert np.allclose(schedule.superpose(unit, t), expected)


def test_observations_are_validated_and_sorted() -> None:
    obs = Observation("a", 10.0, np.array([2.0, 1.0]), np.array([0.2, 0.1]))
    assert list(obs.time) == [1.0, 2.0]
    assert list(obs.drawdown) == [0.1, 0.2]
    assert not obs.at_pumping_well
    with pytest.raises(ValueError, match="same length"):
        Observation("b", 1.0, np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="after the start"):
        Observation("c", 1.0, np.array([0.0, 1.0]), np.ones(2))
    with pytest.raises(ValueError, match="negative"):
        Observation("d", -1.0, np.ones(1), np.ones(1))


def test_pumping_test_rules() -> None:
    obs = Observation("PW", 0.0, np.ones(1), np.ones(1))
    test = PumpingTest(Well(0.2, 0.3), Schedule.constant(1.0), [obs], notes={"site": "x"})
    assert test.radius_of(obs) == 0.2
    assert test.well.storage_radius == 0.3
    assert Well(0.2).storage_radius == 0.2
    assert test.notes == {"site": "x"}
    with pytest.raises(ValueError, match="at least one"):
        PumpingTest(Well(0.2), Schedule.constant(1.0), [])
    with pytest.raises(ValueError, match="unique"):
        PumpingTest(Well(0.2), Schedule.constant(1.0), [obs, obs])


def test_model_registry_and_derived_without_thickness() -> None:
    assert set(MODELS) == {"theis", "hantush-jacob", "papadopulos-cooper", "neuman"}
    assert all(callable(factory) for factory in MODELS.values())
    obs = Observation("o", 5.0, np.ones(1), np.ones(1))
    test = PumpingTest(Well(0.2), Schedule.constant(1.0), [obs])
    assert Theis().derived({"T": 1.0, "S": 1e-4}, test) == {}
    assert Theis().parameter_names == ("T", "S")


def test_stehfest_inverts_known_transforms() -> None:
    assert stehfest_weights(2) == (2.0, -2.0)
    t = np.array([0.5, 1.0, 4.0])
    assert np.allclose(invert(lambda p: 1.0 / p, t), 1.0, rtol=1e-6)
    assert np.allclose(invert(lambda p: 1.0 / p**2, t), t, rtol=1e-6)
    assert np.allclose(invert(lambda p: 1.0 / (p + 1.0), t), np.exp(-t), rtol=1e-3)
    with pytest.raises(ValueError, match="even"):
        stehfest_weights(3)
    with pytest.raises(ValueError, match="positive times"):
        invert(lambda p: 1.0 / p, [0.0])


def test_units() -> None:
    assert float(convert(100, "lpm", "m3/d")) == pytest.approx(144.0)
    assert float(convert(1, "gpm", "m3/d")) == pytest.approx(5.450992, rel=1e-6)
    assert float(convert(10, "ft", "m")) == pytest.approx(3.048)
    assert float(convert(90, "min", "d")) == pytest.approx(90 / 1440)
    with pytest.raises(ValueError, match="not both"):
        convert(1, "m", "min")
    with pytest.raises(ValueError, match="unknown unit"):
        convert(1, "furlong", "m")
    depth = drawdown_from_levels([12.0, 13.5], 10.0)
    assert list(depth) == [2.0, 3.5]
    elevation = drawdown_from_levels([98.0], 100.0, below_datum=False)
    assert list(elevation) == [2.0]


def test_log_derivative_is_flat_for_theis() -> None:
    # Late Theis drawdown has ds/d(ln t) = Q / (4 pi T).
    t = np.logspace(-2, 2, 60)
    s = 100.0 * theis_w(1e-3 / t) / (4.0 * math.pi * 50.0)
    times, d = log_derivative(t, s, window=0.1)
    assert times.size == t.size - 2
    late = d[times > 1.0]
    assert np.allclose(late, 100.0 / (4.0 * math.pi * 50.0), rtol=0.01)
    with pytest.raises(ValueError, match="three points"):
        log_derivative(t[:2], s[:2])
    with pytest.raises(ValueError, match="increasing"):
        log_derivative(t[::-1], s)
