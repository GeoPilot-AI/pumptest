"""Reference values for the well functions, computed without the Laplace transform.

    python scripts/reference_values.py > reference.txt

The library evaluates the Papadopulos-Cooper and Neuman solutions in the
Laplace domain and inverts numerically. This script evaluates the same two
solutions from their real-time integral forms, by direct quadrature, so the
two routes share no code beyond SciPy's Bessel functions. The output is the
table that ``tests/test_type_curves.py`` checks against. It takes several
minutes, which is why the values are stored rather than recomputed per test.

Papadopulos and Cooper (1967), drawdown in the pumped well:

    F(u_w, alpha) = 32 alpha^2 / pi^2 int_0^inf (1 - exp(-b^2 / (4 u_w))) / (b^3 D(b)) db
    D(b) = (b J0(b) - 2 alpha J1(b))^2 + (b Y0(b) - 2 alpha Y1(b))^2

Neuman (1975), fully penetrating well and piezometer, s = Q / (4 pi T) s_D:

    s_D = int_0^inf 4 y J0(y sqrt(beta)) (u0(y) + sum_n un(y)) dy
    u0 = (1 - exp(-ts beta (y^2 - g0^2))) tanh(g0)
         / ((y^2 + (1 + sigma) g0^2 - (y^2 - g0^2)^2 / sigma) g0)
    un = (1 - exp(-ts beta (y^2 + gn^2))) tan(gn)
         / ((y^2 - (1 + sigma) gn^2 - (y^2 + gn^2)^2 / sigma) gn)
    sigma g0 tanh(g0) = y^2 - g0^2,                  g0^2 < y^2
    sigma gn sin(gn) + (y^2 + gn^2) cos(gn) = 0,      (2n - 1) pi / 2 < gn < n pi

with ts = T t / (S r^2), sigma = S / Sy and beta = Kd r^2 / b^2.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from scipy import integrate, optimize, special

PAPADOPULOS_COOPER_CASES = [
    (1e-1, 1.0), (1e-1, 1e2), (1e-1, 1e4), (1e-1, 1e6),
    (1e-3, 1e2), (1e-3, 1e4), (1e-3, 1e6),
    (1e-5, 1e4), (1e-5, 1e6), (1e-5, 1e8),
]  # fmt: skip
"""(alpha, 1/u_w)."""

NEUMAN_CASES = [
    (1e-3, 0.01, 1.0), (1e-3, 0.01, 1e2), (1e-3, 0.01, 1e4),
    (1e-2, 0.1, 10.0), (1e-2, 0.1, 1e3),
    (0.1, 1.0, 1.0), (0.1, 1.0, 1e2),
    (1e-2, 4.0, 10.0), (1e-3, 0.001, 1e4),
]  # fmt: skip
"""(sigma, beta, ts)."""

UPPER = 2.0e4
"""Upper limit of the Neuman integral; the tail beyond it is below 1e-4 relative."""


def _pieces(f: Callable[[float], float], points: np.ndarray) -> float:
    total = 0.0
    for a, b in itertools.pairwise(points):
        total += integrate.quad(f, a, b, limit=400, epsabs=1e-14, epsrel=1e-11)[0]
    return total


def papadopulos_cooper(alpha: float, uw: float) -> float:
    def g(b: float) -> float:
        d = (b * special.j0(b) - 2 * alpha * special.j1(b)) ** 2 + (
            b * special.y0(b) - 2 * alpha * special.y1(b)
        ) ** 2
        return float(-math.expm1(-b * b / (4 * uw)) / (b**3 * d))

    return 32 * alpha**2 / math.pi**2 * _pieces(g, np.geomspace(1e-9, 1e3, 400))


def _g0(y: float, sigma: float) -> float:
    def f(g: float) -> float:
        return sigma * g * math.tanh(g) - (y * y - g * g)

    hi = y * (1 - 1e-15)
    return hi if f(hi) <= 0 else float(optimize.brentq(f, 0.0, hi, xtol=1e-15))


def _gn(n: int, y: float, sigma: float) -> float:
    def f(g: float) -> float:
        return sigma * g * math.sin(g) + (y * y + g * g) * math.cos(g)

    a, b = (2 * n - 1) * math.pi / 2, n * math.pi
    return float(optimize.brentq(f, a + 1e-12, b - 1e-12, xtol=1e-15))


def neuman(sigma: float, beta: float, ts: float) -> float:
    def integrand(y: float) -> float:
        if y == 0.0:
            return 0.0
        g0 = _g0(y, sigma)
        u = 0.0
        if g0 > 0:
            d0 = (y * y + (1 + sigma) * g0 * g0 - (y * y - g0 * g0) ** 2 / sigma) * g0
            u = -math.expm1(-ts * beta * (y * y - g0 * g0)) * math.tanh(g0) / d0
        for n in range(1, 400):
            gn = _gn(n, y, sigma)
            dn = (y * y - (1 + sigma) * gn * gn - (y * y + gn * gn) ** 2 / sigma) * gn
            un = -math.expm1(-ts * beta * (y * y + gn * gn)) * math.tan(gn) / dn
            u += un
            if abs(un) < 1e-14 * max(abs(u), 1e-30):
                break
        return float(4.0 * y * special.j0(y * math.sqrt(beta)) * u)

    return _pieces(integrand, np.concatenate([[0.0], np.geomspace(1e-6, UPPER, 200)]))


def main() -> int:
    print("PAPADOPULOS_COOPER = [  # (alpha, 1/u_w, F)")
    for alpha, inv_u in PAPADOPULOS_COOPER_CASES:
        print(f"    ({alpha:g}, {inv_u:g}, {papadopulos_cooper(alpha, 1 / inv_u):.10g}),")
    print("]")
    print("NEUMAN = [  # (sigma, beta, ts, s_D)")
    for sigma, beta, ts in NEUMAN_CASES:
        print(f"    ({sigma:g}, {beta:g}, {ts:g}, {neuman(sigma, beta, ts):.10g}),")
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
