"""Well functions against independent evaluations of the published solutions.

The library computes the leaky, large-diameter and unconfined solutions in
the Laplace domain and inverts numerically. Here each is compared with the
same solution evaluated another way: the Hantush integral by direct
quadrature, and the Papadopulos-Cooper (1967) and Neuman (1975) real-time
integrals by ``scripts/reference_values.py``, whose output is stored below
because it takes minutes to compute. The two routes share no code beyond
SciPy's special functions, so agreement checks both.
"""

from __future__ import annotations

import itertools
import math
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy import integrate, special

from pumptest import hantush_w, theis_w
from pumptest.laplace import invert
from pumptest.neuman import neuman_drawdown_laplace
from pumptest.radial import radial_drawdown_laplace

# Output of scripts/reference_values.py.
PAPADOPULOS_COOPER = [  # (alpha, 1/u_w, F)
    (0.1, 1, 0.09191299086),
    (0.1, 100, 3.275303386),
    (0.1, 10000, 8.61765422),
    (0.1, 1e06, 13.23805857),
    (0.001, 100, 0.09834135113),
    (0.001, 10000, 5.52587682),
    (0.001, 1e06, 13.21139383),
    (1e-05, 10000, 0.09930946876),
    (1e-05, 1e06, 6.777686485),
    (1e-05, 1e08, 17.80689962),
]
NEUMAN = [  # (sigma, beta, ts, s_D)
    (0.001, 0.01, 1, 0.9644477625),
    (0.001, 0.01, 100, 3.43588377),
    (0.001, 0.01, 10000, 3.895679522),
    (0.01, 0.1, 10, 1.552027184),
    (0.01, 0.1, 1000, 3.17071694),
    (0.1, 1, 1, 0.3451598033),
    (0.1, 1, 100, 3.044749636),
    (0.01, 4, 10, 0.1460410526),
    (0.001, 0.001, 10000, 5.693081747),
]
TOLERANCE = 1e-4
"""Stehfest inversion with 16 terms and the quadrature both resolve these to
better than about 2e-5; 1e-4 leaves room for library and platform changes."""


def _e1_series(u: str) -> float:
    """E1 from -gamma - ln u - sum((-u)**k / (k k!)), in 60-digit decimals."""

    getcontext().prec = 60
    x = Decimal(u)
    gamma = Decimal("0.577215664901532860606512090082402431042159335939923598805767")
    total, term = Decimal(0), Decimal(1)
    for k in range(1, 400):
        term = term * (-x) / k
        total += term / k
    return float(-gamma - x.ln() - total)


def _hantush_quadrature(u: float, rho: float) -> float:
    """W(u, r/B) = int_u^inf exp(-y - rho^2 / (4 y)) / y dy, by direct quadrature."""

    def f(y: float) -> float:
        return math.exp(-y - rho * rho / (4.0 * y)) / y

    pieces = np.concatenate([[u], np.geomspace(max(u, 1e-12) * 1.0001, 100.0, 60)])
    return float(sum(integrate.quad(f, a, b, limit=200)[0] for a, b in itertools.pairwise(pieces)))


def _invert(transform, t: float) -> float:  # type: ignore[no-untyped-def]
    return float(invert(transform, [t])[0])


def test_theis_w_against_its_power_series() -> None:
    cases = ["1e-4", "1e-2", "0.1", "1", "5"]
    expected = [_e1_series(u) for u in cases]
    assert np.allclose(theis_w([float(u) for u in cases]), expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("u", "rho"),
    [(1e-4, 0.02), (4e-4, 0.07), (1e-3, 0.05), (2e-3, 0.08), (1e-2, 0.3), (0.1, 1.0), (1.0, 2.0)],
)
def test_hantush_laplace_kernel_against_quadrature(u: float, rho: float) -> None:
    # The fitted Hantush-Jacob model runs through the Laplace kernel. Unit T, r, S.
    def transform(p: np.ndarray) -> np.ndarray:
        return radial_drawdown_laplace(p, 1.0, 1.0, 1.0, leakage_factor=1.0 / rho)

    expected = _hantush_quadrature(u, rho)
    assert 4.0 * math.pi * _invert(transform, 1.0 / (4.0 * u)) == pytest.approx(
        expected, rel=TOLERANCE
    )
    assert hantush_w(u, rho)[()] == pytest.approx(expected, rel=1e-8)


def test_hantush_limits() -> None:
    u = np.array([1e-4, 1e-2, 1.0])
    assert np.allclose(hantush_w(u, 0.0), theis_w(u), rtol=1e-12)
    assert hantush_w(0.0, 0.5)[()] == pytest.approx(2.0 * special.k0(0.5), rel=1e-12)
    # Late time levels off at the steady state 2 K0(r/B).
    assert hantush_w(1e-8, 0.5)[()] == pytest.approx(2.0 * special.k0(0.5), rel=1e-6)


@pytest.mark.parametrize(("alpha", "inv_u", "expected"), PAPADOPULOS_COOPER)
def test_papadopulos_cooper_against_its_integral(
    alpha: float, inv_u: float, expected: float
) -> None:
    # Drawdown in the pumped well; unit T and radii, so storativity is alpha.
    def transform(p: np.ndarray) -> np.ndarray:
        return radial_drawdown_laplace(
            p, 1.0, 1.0, alpha, well_radius=1.0, casing_radius=1.0, at_pumping_well=True
        )

    f = 4.0 * math.pi * _invert(transform, inv_u * alpha / 4.0)
    assert f == pytest.approx(expected, rel=TOLERANCE)


@pytest.mark.parametrize(("sigma", "beta", "ts", "expected"), NEUMAN)
def test_neuman_against_its_integral(sigma: float, beta: float, ts: float, expected: float) -> None:
    # Unit T, r and Kd; beta = r**2 / b**2 fixes b; S = sigma Sy; ts = t / S.
    sy = 0.1
    s = sigma * sy

    def transform(p: np.ndarray) -> np.ndarray:
        return neuman_drawdown_laplace(p, 1.0, 1.0, s, sy, 1.0, 1.0 / math.sqrt(beta))

    assert 4.0 * math.pi * _invert(transform, ts * s) == pytest.approx(expected, rel=TOLERANCE)


def test_neuman_limits() -> None:
    # Without specific yield the aquifer is confined: Theis with S.
    def no_yield(p: np.ndarray) -> np.ndarray:
        return neuman_drawdown_laplace(p, 1.0, 1.0, 1e-4, 1e-12, 0.1, 10.0)

    for t in (1e-3, 1e-1, 10.0):
        w = 4.0 * math.pi * _invert(no_yield, t)
        assert w == pytest.approx(float(theis_w(1e-4 / (4.0 * t))), rel=1e-6)

    # At late time the water table has drained: Theis with S + Sy.
    def drained(p: np.ndarray) -> np.ndarray:
        return neuman_drawdown_laplace(p, 1.0, 1.0, 1e-4, 0.1, 0.1, 10.0)

    for t in (1e5, 1e7):
        w = 4.0 * math.pi * _invert(drained, t)
        assert w == pytest.approx(float(theis_w((1e-4 + 0.1) / (4.0 * t))), rel=1e-5)
