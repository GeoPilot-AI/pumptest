"""Cooper-Jacob, Theis recovery and step-drawdown, on data with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pumptest import cooper_jacob, step_drawdown, theis_recovery, theis_w

T, S, Q, R = 400.0, 2e-4, 800.0, 30.0


def theis(t: np.ndarray, rate: float = Q) -> np.ndarray:
    return rate * theis_w(R * R * S / (4.0 * T * t)) / (4.0 * math.pi * T)


def test_cooper_jacob_walks_to_small_u() -> None:
    t = np.logspace(-4, 0, 30)
    result = cooper_jacob(t, theis(t), Q, R)
    assert result.valid
    assert result.u_max <= 0.05
    assert result.transmissivity == pytest.approx(T, rel=0.02)
    assert result.storativity == pytest.approx(S, rel=0.1)
    assert result.line.zero_drawdown_time > 0


def test_cooper_jacob_with_a_chosen_start() -> None:
    t = np.logspace(-4, 0, 30)
    result = cooper_jacob(t, theis(t), Q, R, start=0.01)
    assert result.transmissivity == pytest.approx(T, rel=0.01)
    with pytest.raises(ValueError, match="fewer than two"):
        cooper_jacob(t, theis(t), Q, R, start=5.0)


def test_cooper_jacob_refuses_bad_input() -> None:
    t = np.logspace(-2, 0, 5)
    with pytest.raises(ValueError, match="three points"):
        cooper_jacob(t[:2], theis(t[:2]), Q, R)
    with pytest.raises(ValueError, match="positive"):
        cooper_jacob(t, theis(t), 0.0, R)
    with pytest.raises(ValueError, match="does not increase"):
        cooper_jacob(t, -theis(t), Q, R)


def test_theis_recovery() -> None:
    stop = 1.0
    t = np.linspace(1.05, 3.0, 40)
    residual = theis(t) - theis(t - stop)
    result = theis_recovery(t, residual, Q, stop)
    assert result.transmissivity == pytest.approx(T, rel=0.02)
    with pytest.raises(ValueError, match="two readings"):
        theis_recovery(t, residual, Q, 10.0)
    with pytest.raises(ValueError, match="does not fall"):
        theis_recovery(t, -residual, Q, stop)


def test_step_drawdown_recovers_jacob_coefficients() -> None:
    b, c = 0.004, 2e-6
    rates = np.array([300.0, 600.0, 900.0, 1200.0])
    result = step_drawdown(rates, b * rates + c * rates**2)
    assert result.aquifer_loss_coefficient == pytest.approx(b, rel=1e-9)
    assert result.well_loss_coefficient == pytest.approx(c, rel=1e-9)
    assert result.drawdown(1500.0) == pytest.approx(b * 1500 + c * 1500**2, rel=1e-9)
    assert result.efficiencies[0] > result.efficiencies[-1]
    assert result.efficiencies[0] == pytest.approx(b / (b + c * 300.0), rel=1e-9)


def test_step_drawdown_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="two steps"):
        step_drawdown([100.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        step_drawdown([0.0, 100.0], [0.0, 1.0])
