"""The package against welltestpy, on the open Lauswiesen pumping tests.

Mueller et al. (2022) fitted the Theis solution to each test with welltestpy.
welltestpy does not fit the raw records: it normalises the rate, smooths each
record with a low-pass filter and keeps ten times per well, spaced
quadratically from 10 s. Given those same points, pumptest must return their
estimates; given the raw records, it must fit at least as well as their
parameters do.
"""

from __future__ import annotations

import gzip
import math
from importlib import resources

import numpy as np
import pytest
from scipy import signal

from pumptest import Observation, PumpingTest, Schedule, Theis, datasets, fit
from pumptest._types import FloatArray

TESTS = ("lauswiesen_b2", "lauswiesen_b3", "lauswiesen_b4", "lauswiesen_b5")
SECOND = 1.0 / 86400.0


def _raw(name: str) -> FloatArray:
    path = resources.files("pumptest.datasets").joinpath("data", f"{name}.txt.gz")
    with path.open("rb") as fh, gzip.open(fh, "rt") as text:
        table: FloatArray = np.loadtxt(text)
    return table


def _smooth(time: FloatArray, head: FloatArray, tout: FloatArray) -> FloatArray:
    # welltestpy.process.filterdrawdown with its defaults (dxscale 2).
    n = 2 * int((time[-1] - time[0]) / max(float(np.diff(time).min()), 1.0))
    grid = np.linspace(time[0], time[-1], n)
    b, a = signal.butter(1, 0.025)
    filtered = signal.filtfilt(b, a, np.interp(grid, time, head), padlen=150)
    return np.asarray(np.interp(tout, grid, filtered))


def _as_welltestpy_sees_it(name: str) -> PumpingTest:
    test = datasets.load(name)
    raw = _raw(name.replace("lauswiesen_b", "lauswiesen_B"))
    rate = test.schedule.first_rate * SECOND
    tmin = max(10.0, float(raw[0, 0]))
    tout = np.linspace(math.sqrt(tmin), math.sqrt(float(raw[-1, 0])), 10) ** 2
    obs = [
        Observation(o.name, o.distance, tout * SECOND, _smooth(raw[:, 0], raw[:, i] / rate, tout))
        for i, o in enumerate(test.observations, start=1)
    ]
    return PumpingTest(test.well, Schedule.constant(86400.0), obs, test.thickness)


def _rmse(test: PumpingTest, values: dict[str, float]) -> float:
    model = Theis()
    res = np.concatenate([o.drawdown - model.drawdown(values, test, o) for o in test.observations])
    return float(np.sqrt(np.mean(res**2)))


@pytest.mark.parametrize("name", TESTS)
def test_reproduces_welltestpy_on_its_own_points(name: str) -> None:
    published = datasets.published(name).values
    result = fit(_as_welltestpy_sees_it(name), Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=1e-3)
    assert result.values["S"] == pytest.approx(published["S"], rel=5e-3)


@pytest.mark.parametrize("name", TESTS)
def test_raw_records_fit_at_least_as_well(name: str) -> None:
    test, published = datasets.load(name), datasets.published(name).values
    result = fit(test, Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.04)
    assert result.rmse <= _rmse(test, published)


def test_lauswiesen_geometry() -> None:
    test = datasets.load("lauswiesen_b3")
    assert [o.name for o in test.observations] == ["B1", "B2", "B3", "B4", "B5"]
    assert test.observations[2].distance == 0.0
    assert test.observations[1].distance == pytest.approx(9.77, abs=0.01)
