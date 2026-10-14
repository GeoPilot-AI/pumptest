"""Automatic fitting: every model recovers parameters it generated.

Synthetic data with known answers is the only way to test the unconfined and
wellbore-storage fits end to end, because the published comparisons for those
solutions involve partial penetration this package does not model yet.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pumptest import (
    HantushJacob,
    Model,
    Neuman,
    Observation,
    PapadopulosCooper,
    PumpingTest,
    Schedule,
    Theis,
    Well,
    compare,
    fit,
)
from pumptest.fit import starting_points

RNG = np.random.default_rng(20261014)


def synthetic(
    model: Model,
    values: dict[str, float],
    *,
    distances: tuple[float, ...] = (30.0, 90.0),
    well: Well | None = None,
    schedule: Schedule | None = None,
    thickness: float | None = 20.0,
    noise: float = 0.0,
    times: np.ndarray | None = None,
) -> PumpingTest:
    t = np.logspace(-4, 0, 40) if times is None else times
    well = well or Well(0.1)
    schedule = schedule or Schedule.constant(500.0)
    template = PumpingTest(
        well,
        schedule,
        [Observation(f"r{d:g}", d, t, np.ones_like(t)) for d in distances],
        thickness=thickness,
    )
    observations = []
    for obs in template.observations:
        s = model.drawdown(values, template, obs)
        observations.append(
            Observation(obs.name, obs.distance, t, s + noise * RNG.standard_normal(t.size))
        )
    return PumpingTest(well, schedule, observations, thickness=thickness)


def test_theis_recovers_its_parameters_from_noisy_data() -> None:
    truth = {"T": 250.0, "S": 2e-4}
    result = fit(synthetic(Theis(), truth, noise=0.002), Theis())
    assert result.values["T"] == pytest.approx(truth["T"], rel=0.02)
    assert result.values["S"] == pytest.approx(truth["S"], rel=0.05)
    assert result.rmse == pytest.approx(0.002, rel=0.3)
    assert result.success
    assert result.correlation.shape == (2, 2)
    assert set(result.residuals) == {"r30", "r90"}


def test_hantush_recovers_leakage() -> None:
    truth = {"T": 800.0, "S": 5e-4, "B": 400.0}
    result = fit(synthetic(HantushJacob(), truth), HantushJacob())
    for name, value in truth.items():
        assert result.values[name] == pytest.approx(value, rel=1e-3)
    assert result.derived["c"] == pytest.approx(400.0**2 / 800.0, rel=2e-3)
    assert result.derived["leakance"] == pytest.approx(800.0 / 400.0**2, rel=2e-3)


def test_neuman_recovers_delayed_yield() -> None:
    truth = {"T": 300.0, "S": 1e-3, "Sy": 0.08, "Kd": 0.2}
    test = synthetic(Neuman(), truth, distances=(15.0, 40.0), times=np.logspace(-5, 1, 30))
    result = fit(test, Neuman())
    for name, value in truth.items():
        assert result.values[name] == pytest.approx(value, rel=0.01)
    assert result.derived["Kz"] == pytest.approx(0.2 * 300.0 / 20.0, rel=0.02)


def test_neuman_piezometer_screen_changes_the_response() -> None:
    truth = {"T": 300.0, "S": 1e-3, "Sy": 0.08, "Kd": 0.2}
    test = synthetic(Neuman(), truth, distances=(15.0,))
    full = test.observations[0]
    deep = Observation("deep", 15.0, full.time, full.drawdown, screen=(0.0, 2.0))
    shallow = Observation("shallow", 15.0, full.time, full.drawdown, screen=(18.0, 18.0))
    s_deep = Neuman().drawdown(truth, test, deep)
    s_shallow = Neuman().drawdown(truth, test, shallow)
    # Near the water table the drawdown lags: the free surface feeds it.
    assert np.all(s_shallow[5:25] < s_deep[5:25])


def test_papadopulos_cooper_fits_a_dug_well() -> None:
    truth = {"T": 50.0, "S": 1e-3}
    well = Well(radius=1.5, casing_radius=1.5, name="Dug well")
    test = synthetic(PapadopulosCooper(), truth, distances=(0.0,), well=well)
    result = fit(test, PapadopulosCooper())
    assert result.values["T"] == pytest.approx(50.0, rel=1e-3)
    assert result.values["S"] == pytest.approx(1e-3, rel=0.02)


def test_papadopulos_cooper_with_skin() -> None:
    truth = {"T": 120.0, "S": 1e-4, "Sk": 3.0}
    test = synthetic(
        PapadopulosCooper(skin=True), truth, distances=(0.0, 20.0), well=Well(0.15, 0.25)
    )
    model = PapadopulosCooper(skin=True)
    assert model.name.endswith("skin")
    result = fit(test, model)
    assert result.values["Sk"] == pytest.approx(3.0, abs=0.05)
    assert result.values["T"] == pytest.approx(120.0, rel=1e-3)


def test_papadopulos_cooper_recovers_the_effective_casing_radius() -> None:
    # A pumped well with skin, observed only in the well, S held at its true
    # value because a single well cannot separate it from the skin.
    truth = {"T": 1800.0, "S": 4e-4, "Sk": 25.0, "rc": 0.09}
    second = 1.0 / 86400.0
    model = PapadopulosCooper(skin=True, casing_radius=True)
    test = synthetic(
        model,
        truth,
        distances=(0.0,),
        well=Well(0.185),
        schedule=Schedule.constant(690.0),
        thickness=None,
        times=np.geomspace(5 * second, 750 * second, 25),
    )
    result = fit(test, model, fixed={"S": 4e-4})
    assert result.values["rc"] == pytest.approx(0.09, rel=1e-3)
    assert result.values["T"] == pytest.approx(1800.0, rel=1e-3)
    assert result.values["Sk"] == pytest.approx(25.0, rel=1e-3)


def test_recovery_is_fitted_through_the_schedule() -> None:
    truth = {"T": 250.0, "S": 2e-4}
    schedule = Schedule.constant(500.0, stop=0.5)
    t = np.linspace(0.01, 1.0, 60)
    test = synthetic(Theis(), truth, schedule=schedule, times=t)
    late = test.observations[0].drawdown[t > 0.6]
    assert np.all(np.diff(late) < 0)
    result = fit(test, Theis())
    assert result.values["T"] == pytest.approx(250.0, rel=1e-4)


def test_fixed_parameters_and_named_observations() -> None:
    truth = {"T": 250.0, "S": 2e-4}
    test = synthetic(Theis(), truth)
    result = fit(test, Theis(), fixed={"S": 2e-4}, observations=["r30"])
    assert result.estimated == ("T",)
    assert result.values["S"] == 2e-4
    assert result.values["T"] == pytest.approx(250.0, rel=1e-6)
    assert "fixed" in result.summary()
    assert set(result.residuals) == {"r30"}


def test_explicit_initial_guesses() -> None:
    truth = {"T": 250.0, "S": 2e-4}
    test = synthetic(Theis(), truth)
    one = fit(test, Theis(), initial={"T": 10.0, "S": 1e-2})
    many = fit(test, Theis(), initial=[{"T": 10.0}, {"T": 1e4, "S": 1e-6}])
    assert one.values["T"] == pytest.approx(250.0, rel=1e-5)
    assert many.values["T"] == pytest.approx(250.0, rel=1e-5)


def test_bad_requests_are_refused() -> None:
    test = synthetic(Theis(), {"T": 250.0, "S": 2e-4})
    with pytest.raises(ValueError, match="no parameter"):
        fit(test, Theis(), fixed={"B": 1.0})
    with pytest.raises(ValueError, match="nothing to estimate"):
        fit(test, Theis(), fixed={"T": 1.0, "S": 1e-4})
    with pytest.raises(ValueError, match="none of the named"):
        fit(test, Theis(), observations=["missing"])
    no_thickness = synthetic(Theis(), {"T": 250.0, "S": 2e-4}, thickness=None)
    with pytest.raises(ValueError, match="saturated thickness"):
        fit(no_thickness, Neuman())


def test_starting_points_survive_a_flat_record() -> None:
    t = np.logspace(-3, 0, 10)
    test = PumpingTest(
        Well(0.1), Schedule.constant(100.0), [Observation("flat", 10.0, t, np.full(10, 0.5))]
    )
    starts = starting_points(test, Theis())
    assert starts[0]["T"] == pytest.approx(100.0 / (4.0 * math.pi * 0.5))


def test_summary_reports_a_failed_fit() -> None:
    test = synthetic(Theis(), {"T": 250.0, "S": 2e-4})
    result = fit(test, Theis())
    failed = type(result)(**{**result.__dict__, "success": False, "message": "stalled"})
    assert "warning: stalled" in failed.summary()


def test_information_criteria_and_compare() -> None:
    truth = {"T": 800.0, "S": 5e-4, "B": 400.0}
    test = synthetic(HantushJacob(), truth, noise=0.001)
    ranked = compare(test, [Theis(), HantushJacob(), Neuman()])
    assert ranked[0].model == "Hantush-Jacob"
    assert [r.aicc for r in ranked] == sorted(r.aicc for r in ranked)
    best = ranked[0]
    assert best.aicc > best.aic
    assert "AICc" in best.summary()
    # Without a thickness the unconfined model is skipped, not fatal.
    flat = synthetic(Theis(), {"T": 250.0, "S": 2e-4}, thickness=None)
    names = [r.model for r in compare(flat, [Theis(), Neuman()], observations=["r30"])]
    assert names == ["Theis"]


def test_aicc_is_infinite_without_spare_points() -> None:
    t = np.array([0.01, 0.1])
    test = PumpingTest(
        Well(0.1), Schedule.constant(100.0), [Observation("two", 10.0, t, np.array([0.1, 0.3]))]
    )
    assert math.isinf(fit(test, Theis()).aicc)


def test_papadopulos_cooper_model_names() -> None:
    assert PapadopulosCooper().name == "Papadopulos-Cooper"
    assert PapadopulosCooper(casing_radius=True).name == "Papadopulos-Cooper with fitted rc"
    both = PapadopulosCooper(skin=True, casing_radius=True)
    assert both.name == "Papadopulos-Cooper with skin and fitted rc"
    assert both.parameter_names == ("T", "S", "Sk", "rc")
