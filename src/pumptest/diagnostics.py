"""Diagnostic plots' data: the derivative that tells one aquifer from another.

On a log-log plot the drawdown derivative ds/d(ln t) is flat for radial flow
in a confined aquifer, falls to zero for leakage or a recharge boundary, dips
for delayed yield and rises for a barrier. Reading it before fitting is what
stops a leaky test being forced through Theis.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pumptest._types import FloatArray


def log_derivative(
    time: npt.ArrayLike, drawdown: npt.ArrayLike, *, window: float = 0.2
) -> tuple[FloatArray, FloatArray]:
    """Bourdet et al. (1989) derivative ds/d(ln t), smoothed over ``window``.

    For each point the neighbours used are the first ones at least ``window``
    (in natural-log units of time) to each side; field data needs 0.1 to 0.5.
    Returns the times that have neighbours on both sides and the derivative.
    """

    t = np.asarray(time, dtype=np.float64)
    s = np.asarray(drawdown, dtype=np.float64)
    if t.shape != s.shape or t.size < 3:
        raise ValueError("need at least three points, with matching time and drawdown")
    if np.any(np.diff(t) <= 0) or np.any(t <= 0):
        raise ValueError("times must be positive and strictly increasing")
    x = np.log(t)
    times: list[float] = []
    values: list[float] = []
    for i in range(1, t.size - 1):
        left = np.nonzero(x[i] - x[:i] >= window)[0]
        right = np.nonzero(x[i + 1 :] - x[i] >= window)[0]
        j = int(left[-1]) if left.size else i - 1
        k = int(right[0]) + i + 1 if right.size else i + 1
        dl, dr = x[i] - x[j], x[k] - x[i]
        slope_l = (s[i] - s[j]) / dl
        slope_r = (s[k] - s[i]) / dr
        times.append(float(t[i]))
        values.append(float((slope_l * dr + slope_r * dl) / (dl + dr)))
    return np.asarray(times), np.asarray(values)
