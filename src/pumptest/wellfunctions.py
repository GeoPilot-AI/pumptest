"""Dimensionless well functions: the type curves of the classic solutions.

These are the tabulated functions of the textbooks (Kruseman and de Ridder,
1994), exposed so a user can draw type curves or check a value against a
table. The fitted models use them where a closed form exists and the Laplace
kernels in :mod:`pumptest.radial` and :mod:`pumptest.neuman` where it does not.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import integrate, special

from pumptest._types import FloatArray


def theis_w(u: npt.ArrayLike) -> FloatArray:
    """Theis well function W(u), the exponential integral E1(u)."""

    values: FloatArray = special.exp1(np.asarray(u, dtype=np.float64))
    return values


def _hantush_scalar(u: float, rho: float) -> float:
    if u <= 0:
        # Steady state: the integral from zero is 2 K0(r/B).
        return 2.0 * float(special.k0(rho))
    if rho == 0:
        return float(special.exp1(u))
    integrand = lambda y: math.exp(-y - rho * rho / (4.0 * y)) / y  # noqa: E731
    # The integrand peaks near y = rho / 2; splitting there keeps quad accurate
    # for the small u and large r/B corners of the type-curve family.
    peak = max(u, rho / 2.0)
    head, _ = integrate.quad(integrand, u, peak, limit=200) if peak > u else (0.0, 0.0)
    tail, _ = integrate.quad(integrand, peak, math.inf, limit=200)
    return float(head + tail)


def hantush_w(u: npt.ArrayLike, r_over_b: npt.ArrayLike) -> FloatArray:
    """Hantush and Jacob (1955) leaky well function W(u, r/B).

    Evaluated by quadrature, so it is the reference the Laplace kernel is
    tested against rather than the path the fitting takes.
    """

    uu, rr = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64), np.asarray(r_over_b, dtype=np.float64)
    )
    out = np.empty(uu.shape, dtype=np.float64)
    for index in np.ndindex(uu.shape):
        out[index] = _hantush_scalar(float(uu[index]), float(rr[index]))
    return out
