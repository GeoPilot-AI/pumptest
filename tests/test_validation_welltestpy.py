"""The package against welltestpy, on the open Lauswiesen pumping tests.

Mueller et al. (2022) fitted the Theis solution to each test with welltestpy.
welltestpy does not fit the raw records: it normalises the rate, smooths each
record with a low-pass filter and keeps ten times per well, spaced
quadratically from 10 s. Given those same points, pumptest must return their
estimates; given the raw records, it must fit at least as well as their
parameters do. The preprocessing lives in the validation report, so the page
and this test use one copy of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pumptest import PumpingTest, Theis, datasets, fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validation_report import WELLTESTPY, as_welltestpy_sees_it


def _rmse(test: PumpingTest, values: dict[str, float]) -> float:
    model = Theis()
    res = np.concatenate([o.drawdown - model.drawdown(values, test, o) for o in test.observations])
    return float(np.sqrt(np.mean(res**2)))


@pytest.mark.parametrize("name", WELLTESTPY)
def test_reproduces_welltestpy_on_its_own_points(name: str) -> None:
    published = datasets.published(name).values
    result = fit(as_welltestpy_sees_it(name), Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=1e-3)
    assert result.values["S"] == pytest.approx(published["S"], rel=5e-3)


@pytest.mark.parametrize("name", WELLTESTPY)
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
