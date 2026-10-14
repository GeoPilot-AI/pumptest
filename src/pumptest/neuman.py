"""Neuman (1974) delayed water-table response, in the Laplace domain.

A fully penetrating well pumps an anisotropic unconfined aquifer of saturated
thickness b. The water table is a linearised free surface that releases
specific yield instantly (Neuman's assumption; Moench's gradual drainage is a
later extension). Separating variables in the vertical gives eigenvalues
eps_n, the roots of ``eps tan(eps) = p Sy b / Kz``, and

    s(r, z, p) = 1 / (2 pi T p) * sum_n a_n K0(q_n r) phi_n(z),

with ``a_n = 2 sin(eps_n) / (eps_n + sin(eps_n) cos(eps_n))``,
``(q_n r)**2 = r**2 S p / T + beta eps_n**2`` and ``beta = Kd r**2 / b**2``.
``phi_n`` is ``cos(eps_n z / b)`` averaged over the observation screen.

Two limits pin the derivation and are tested: with Sy -> 0 the series
collapses to Theis with S, and at late time to Theis with S + Sy.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from pumptest._types import FloatArray

MAX_TERMS = 400
"""A pumping well (r tiny against b) needs the most terms. The coefficients
fall as c**2 / eps**4 once eps passes c, so 400 terms already agree with 2000
to six figures at r = 0.1 m in a 20 m aquifer; the tests pin that."""

_CHUNK = 200_000
"""Root arrays are rows times terms; chunking bounds the memory per call."""


def eigenvalues(c: FloatArray, n_terms: int) -> FloatArray:
    """Roots of ``eps tan(eps) = c`` for each ``c``, the first ``n_terms`` each.

    Root n lies in (n pi, n pi + pi/2). Newton on ``eps sin - c cos`` (no
    poles), kept inside that bracket by bisection, converges in a few steps.
    """

    cc = np.asarray(c, dtype=np.float64)[:, None]
    n = np.arange(n_terms, dtype=np.float64)[None, :]
    lo = np.broadcast_to(n * math.pi, (cc.shape[0], n_terms)).copy()
    hi = lo + math.pi / 2.0
    # For n >= 1, tan(eps) = c / eps gives eps ~ n pi + atan(c / (n pi)).
    base = np.where(n > 0, n * math.pi, 1.0)
    eps = np.where(n > 0, lo + np.arctan(cc / base), np.arctan(np.sqrt(cc)) * 1.0)
    eps = np.clip(eps, lo, hi)
    for _ in range(60):
        sin, cos = np.sin(eps), np.cos(eps)
        g = eps * sin - cc * cos
        # g rises through the bracket for even n and falls for odd n.
        rising = (n % 2) == 0
        below = np.where(rising, g < 0, g > 0)
        lo = np.where(below, eps, lo)
        hi = np.where(below, hi, eps)
        slope = (1.0 + cc) * sin + eps * cos
        with np.errstate(divide="ignore", invalid="ignore"):
            step = eps - g / slope
        inside = (step > lo) & (step < hi) & np.isfinite(step)
        new = np.where(inside, step, 0.5 * (lo + hi))
        if np.max(np.abs(new - eps)) < 1e-14 * max(1.0, float(np.max(new))):
            eps = new
            break
        eps = new
    roots: FloatArray = eps
    return roots


def _terms_needed(beta: float) -> int:
    # K0(sqrt(beta) eps) is below 1e-13 once its argument passes about 30.
    if beta <= 0:
        return MAX_TERMS
    return int(min(MAX_TERMS, max(20, math.ceil(30.0 / (math.pi * math.sqrt(beta))) + 5)))


def _screen_average(eps: FloatArray, z_low: float, z_high: float) -> FloatArray:
    if z_high - z_low <= 0:
        point: FloatArray = np.cos(eps * z_low)
        return point
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (np.sin(eps * z_high) - np.sin(eps * z_low)) / (eps * (z_high - z_low))
    averaged: FloatArray = np.where(eps == 0, 1.0, mean)
    return averaged


def neuman_drawdown_laplace(
    p: FloatArray,
    r: float,
    transmissivity: float,
    storativity: float,
    specific_yield: float,
    anisotropy: float,
    thickness: float,
    *,
    screen: tuple[float, float] | None = None,
) -> FloatArray:
    """Laplace transform of drawdown per unit rate in an unconfined aquifer.

    ``anisotropy`` is Kd = Kz / Kr. ``screen`` is the observation interval as
    heights above the aquifer base, in the units of ``thickness``; ``None``
    means a fully penetrating observation well and a point is ``(z, z)``.
    """

    t, s, b = transmissivity, storativity, thickness
    beta = anisotropy * r * r / (b * b)
    z_low, z_high = (0.0, 1.0) if screen is None else (screen[0] / b, screen[1] / b)
    shape = p.shape
    flat = p.reshape(-1)
    c_all = flat * specific_yield * b * b / (anisotropy * t)
    n_terms = _terms_needed(beta)
    out = np.empty_like(flat)
    rows = max(1, _CHUNK // n_terms)
    for start in range(0, flat.size, rows):
        pp = flat[start : start + rows][:, None]
        eps = eigenvalues(c_all[start : start + rows], n_terms)
        sin = np.sin(eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 2.0 * sin / (eps + sin * np.cos(eps))
        a = np.where(eps == 0, 1.0, a)
        arg = np.sqrt(r * r * s * pp / t + beta * eps * eps)
        terms = a * special.k0(arg) * _screen_average(eps, z_low, z_high)
        out[start : start + rows] = terms.sum(axis=1) / (2.0 * math.pi * t * pp[:, 0])
    result: FloatArray = out.reshape(shape)
    return result
