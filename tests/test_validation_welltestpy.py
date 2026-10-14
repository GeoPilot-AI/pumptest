"""The package against welltestpy, on the open Lauswiesen pumping tests.

Mueller et al. (2022) fitted the Theis solution to each test with welltestpy,
a different open-source program with a stochastic optimiser. The same two
criteria as for AQTESOLV apply: transmissivity close to theirs, and a fit at
least as good as their parameters give on the same data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumptest import PumpingTest, Theis, datasets, fit

TESTS = ("lauswiesen_b2", "lauswiesen_b3", "lauswiesen_b4", "lauswiesen_b5")


def _rmse(test: PumpingTest, values: dict[str, float]) -> float:
    res = np.concatenate(
        [o.drawdown - Theis().drawdown(values, test, o) for o in test.observations]
    )
    return float(np.sqrt(np.mean(res**2)))


@pytest.mark.parametrize("name", TESTS)
def test_lauswiesen_matches_welltestpy(name: str) -> None:
    test, published = datasets.load(name), datasets.published(name).values
    result = fit(test, Theis())
    assert result.values["T"] == pytest.approx(published["T"], rel=0.04)
    assert result.values["S"] == pytest.approx(published["S"], rel=0.06)
    assert result.rmse <= _rmse(test, published)


def test_lauswiesen_geometry() -> None:
    test = datasets.load("lauswiesen_b3")
    assert [o.name for o in test.observations] == ["B1", "B2", "B3", "B4", "B5"]
    assert test.observations[2].distance == 0.0
    assert test.observations[1].distance == pytest.approx(9.77, abs=0.01)
