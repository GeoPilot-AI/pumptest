"""Numerical inversion of Laplace transforms by the Stehfest (1970) algorithm.

Most well-hydraulics solutions beyond Theis have no closed form in time but a
simple one in the Laplace domain. Stehfest's method samples the transform at
``n`` real points per time and needs no complex arithmetic, which is why WTAQ
and AQTESOLV use it too. It is accurate for the smooth, monotone responses of
pumping tests; it is not suited to oscillating functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

import numpy as np
import numpy.typing as npt

from pumptest._types import FloatArray

DEFAULT_TERMS = 16
"""Sixteen terms holds Theis to 4e-5 wherever u <= 2, which covers every
drawdown large enough to measure. Eighteen starts to add round-off at small u;
twelve leaves 0.1 percent error near u = 2."""


@cache
def stehfest_weights(n: int) -> tuple[float, ...]:
    """The ``n`` Stehfest coefficients V_i (``n`` must be even)."""

    if n <= 0 or n % 2:
        raise ValueError(f"Stehfest needs an even, positive number of terms, got {n}")
    half = n // 2
    weights = []
    for i in range(1, n + 1):
        total = 0.0
        for k in range((i + 1) // 2, min(i, half) + 1):
            total += (
                k**half
                * math.factorial(2 * k)
                / (
                    math.factorial(half - k)
                    * math.factorial(k)
                    * math.factorial(k - 1)
                    * math.factorial(i - k)
                    * math.factorial(2 * k - i)
                )
            )
        weights.append((-1) ** (i + half) * total)
    return tuple(weights)


def invert(
    transform: Callable[[FloatArray], FloatArray],
    t: npt.ArrayLike,
    n: int = DEFAULT_TERMS,
) -> FloatArray:
    """Invert ``transform`` at the positive times ``t``.

    ``transform`` receives a 2-D array of Laplace variables, one row per time,
    and must return an array of the same shape. Evaluating every row in one call
    is what keeps fitting fast: the Bessel functions run vectorised.
    """

    times = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if np.any(times <= 0):
        raise ValueError("Laplace inversion needs strictly positive times")
    weights = np.asarray(stehfest_weights(n))
    ln2 = math.log(2.0)
    p = np.outer(ln2 / times, np.arange(1, n + 1, dtype=np.float64))
    values = np.asarray(transform(p), dtype=np.float64)
    result: FloatArray = (values @ weights) * ln2 / times
    return result
