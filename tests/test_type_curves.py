"""Well functions against the tables of Kruseman and de Ridder (1994).

Values transcribed from Annex 5.1 (Neuman 1975) and Annex 15.1 (Papadopulos
and Cooper 1967) of *Analysis and evaluation of pumping test data*, ILRI
Publication 47, 2nd edition. The tables carry three significant figures, so a
0.6 percent tolerance is the rounding plus Stehfest error, not slack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special

from pumptest import hantush_w, theis_w
from pumptest.laplace import invert
from pumptest.neuman import neuman_drawdown_laplace
from pumptest.radial import radial_drawdown_laplace

TABLE_TOLERANCE = 0.006

# (beta, 1/u_A, W): Neuman type A curves, early time, sigma = S/Sy -> 0.
NEUMAN_A = [
    (0.001, 0.4, 2.48e-2), (0.001, 4, 1.02), (0.001, 40, 2.97), (0.001, 400, 4.77),
    (0.001, 1.4e4, 5.62), (0.01, 0.4, 2.41e-2), (0.01, 4, 9.63e-1), (0.01, 40, 2.61),
    (0.01, 1.4e3, 3.46), (0.1, 0.4, 2.24e-2), (0.1, 4, 7.92e-1), (0.1, 14, 1.34),
    (0.1, 80, 1.55), (0.6, 0.4, 1.88e-2), (0.6, 4, 4.38e-1), (0.6, 14, 5.07e-1),
    (1.0, 0.4, 1.70e-2), (1.0, 4, 3.00e-1), (1.0, 8, 3.17e-1), (4.0, 0.4, 9.33e-3),
    (4.0, 4, 4.78e-2),
]  # fmt: skip

# (beta, 1/u_B, W): Neuman type B curves, late time.
NEUMAN_B = [
    (0.001, 0.8, 5.62), (0.001, 14, 5.65), (0.001, 140, 5.85), (0.001, 400, 6.16),
    (0.01, 0.8, 3.46), (0.01, 4, 3.51), (0.01, 40, 3.90), (0.01, 400, 5.46),
    (0.1, 0.08, 1.56), (0.1, 0.8, 1.61), (0.1, 4, 1.83), (0.1, 40, 3.18), (0.1, 400, 5.42),
    (0.6, 0.004, 5.08e-1), (0.6, 0.4, 5.89e-1), (0.6, 4, 1.20), (0.6, 40, 3.14),
    (0.6, 400, 5.42),
]  # fmt: skip

# (alpha, 1/u_w, F): Papadopulos-Cooper drawdown inside a large-diameter well.
PAPADOPULOS_COOPER = [
    (1e-1, 0.1, 9.75e-3), (1e-1, 1, 9.19e-2), (1e-1, 10, 7.34e-1), (1e-1, 100, 3.28),
    (1e-1, 1e3, 6.21), (1e-1, 1e4, 8.57), (1e-1, 1e5, 10.9), (1e-1, 1e9, 20.1),
    (1e-3, 1, 9.99e-4), (1e-3, 100, 9.83e-2), (1e-3, 1e3, 9.07e-1), (1e-3, 1e4, 5.53),
    (1e-3, 1e6, 13.2), (1e-5, 1e4, 9.93e-2), (1e-5, 1e6, 6.78), (1e-5, 1e8, 17.8),
]  # fmt: skip


def neuman_w(beta: float, t: float, s: float, sy: float) -> float:
    # Unit T, r and Kd; beta = r**2 / b**2 then fixes the thickness.
    b = 1.0 / math.sqrt(beta)
    value = invert(lambda p: neuman_drawdown_laplace(p, 1.0, 1.0, s, sy, 1.0, b), [t])
    return float(4.0 * math.pi * value[0])


@pytest.mark.parametrize(("beta", "inv_u", "expected"), NEUMAN_A)
def test_neuman_type_a(beta: float, inv_u: float, expected: float) -> None:
    sy = 0.1
    s = 1e-7 * sy
    w = neuman_w(beta, inv_u * s / 4.0, s, sy)
    assert w == pytest.approx(expected, rel=TABLE_TOLERANCE)


@pytest.mark.parametrize(("beta", "inv_u", "expected"), NEUMAN_B)
def test_neuman_type_b(beta: float, inv_u: float, expected: float) -> None:
    sy = 0.1
    w = neuman_w(beta, inv_u * sy / 4.0, 1e-7 * sy, sy)
    assert w == pytest.approx(expected, rel=TABLE_TOLERANCE)


@pytest.mark.parametrize(("alpha", "inv_u", "expected"), PAPADOPULOS_COOPER)
def test_papadopulos_cooper_table(alpha: float, inv_u: float, expected: float) -> None:
    # Unit T and radii, so storativity is alpha itself.
    def transform(p: np.ndarray) -> np.ndarray:
        return radial_drawdown_laplace(
            p, 1.0, 1.0, alpha, well_radius=1.0, casing_radius=1.0, at_pumping_well=True
        )

    f = 4.0 * math.pi * float(invert(transform, [inv_u * alpha / 4.0])[0])
    assert f == pytest.approx(expected, rel=TABLE_TOLERANCE)


def test_theis_w_known_values() -> None:
    # E1 to nine figures (Abramowitz and Stegun, Table 5.1).
    u = [1e-4, 1e-2, 0.1, 1.0, 5.0]
    expected = [8.633224705, 4.037929577, 1.822923958, 0.219383934, 0.001148296]
    assert np.allclose(theis_w(u), expected, rtol=1e-8)


def test_hantush_limits() -> None:
    u = np.array([1e-4, 1e-2, 1.0])
    assert np.allclose(hantush_w(u, 0.0), theis_w(u), rtol=1e-12)
    assert hantush_w(0.0, 0.5)[()] == pytest.approx(2.0 * special.k0(0.5), rel=1e-12)
    # Late time levels off at the steady state 2 K0(r/B).
    assert hantush_w(1e-8, 0.5)[()] == pytest.approx(2.0 * special.k0(0.5), rel=1e-6)


def test_hantush_table_values() -> None:
    # Annex 4.2 (after Hantush 1956), W(u, r/L), three figures. The annex prints
    # 4.89 at u = 2e-3, r/L = 0.08 between neighbours 5.12 and 4.85 that this
    # function reproduces exactly; 4.98 is the value, so that cell is skipped.
    cases = [
        (0.0, 0.01, 9.44), (1e-4, 0.02, 7.84), (4e-4, 0.07, 5.55),
        (1e-3, 0.05, 5.80), (2e-3, 0.09, 4.85), (1e-2, 0.09, 3.85),
    ]  # fmt: skip
    for u, rho, expected in cases:
        assert hantush_w(u, rho)[()] == pytest.approx(expected, rel=TABLE_TOLERANCE)
