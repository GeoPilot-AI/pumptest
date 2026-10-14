"""Straight-line methods: the hand analyses every hydrogeologist checks first.

- Cooper and Jacob (1946) time-drawdown, valid once u is small.
- Theis (1935) recovery, on residual drawdown against t / t'.
- Step-drawdown well losses, s / Q = B + C Q (Jacob 1947; Hantush-Bierschenk).

They also seed the automatic fit, which starts from the Cooper-Jacob line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LN10 = math.log(10.0)
JACOB_U_LIMIT = 0.05
"""Below this u the Cooper-Jacob line is within about 2 percent of Theis."""


@dataclass(frozen=True)
class LineFit:
    """A straight line through drawdown against log10 of time."""

    slope: float
    """Drawdown per log cycle."""
    intercept: float
    """Drawdown where log10(t) is zero."""
    used: int
    """How many points the line went through."""

    @property
    def zero_drawdown_time(self) -> float:
        """t0, where the line crosses zero drawdown."""

        return float(10.0 ** (-self.intercept / self.slope))


@dataclass(frozen=True)
class CooperJacobResult:
    transmissivity: float
    storativity: float
    line: LineFit
    u_max: float
    """The largest u among the points used; above 0.05 the result is suspect."""

    @property
    def valid(self) -> bool:
        return self.u_max <= JACOB_U_LIMIT


def _line(log_t: npt.NDArray[np.float64], s: npt.NDArray[np.float64]) -> LineFit:
    slope, intercept = np.polyfit(log_t, s, 1)
    return LineFit(float(slope), float(intercept), int(log_t.size))


def cooper_jacob(
    time: npt.ArrayLike,
    drawdown: npt.ArrayLike,
    rate: float,
    distance: float,
    *,
    start: float | None = None,
) -> CooperJacobResult:
    """Cooper-Jacob analysis of a constant-rate test at one well.

    With ``start`` the line uses times from ``start`` on. Without it the line
    starts on the later half of the record and walks the start later until
    every point used has u <= 0.05, or until three points are left.
    """

    t = np.asarray(time, dtype=np.float64)
    s = np.asarray(drawdown, dtype=np.float64)
    if t.size < 3:
        raise ValueError("Cooper-Jacob needs at least three points")
    if rate <= 0 or distance <= 0:
        raise ValueError("rate and distance must be positive")

    def solve(mask: npt.NDArray[np.bool_]) -> CooperJacobResult:
        line = _line(np.log10(t[mask]), s[mask])
        # A flat record fits a slope of round-off size, which would give an
        # absurd transmissivity rather than an error.
        if line.slope <= 1e-9 * float(np.max(np.abs(s[mask]))):
            raise ValueError("drawdown does not increase with time over the chosen points")
        trans = 2.3026 * rate / (4.0 * math.pi * line.slope)
        stor = 2.25 * trans * line.zero_drawdown_time / distance**2
        u_max = float(distance**2 * stor / (4.0 * trans * t[mask].min()))
        return CooperJacobResult(trans, stor, line, u_max)

    if start is not None:
        mask = t >= start
        if int(mask.sum()) < 2:
            raise ValueError("fewer than two points after the chosen start time")
        return solve(mask)

    first = t.size // 2
    result = solve(np.arange(t.size) >= first)
    while not result.valid and t.size - first > 3:
        first += 1
        result = solve(np.arange(t.size) >= first)
    return result


@dataclass(frozen=True)
class RecoveryResult:
    transmissivity: float
    line: LineFit
    """Residual drawdown against log10(t / t')."""


def theis_recovery(
    time_since_start: npt.ArrayLike,
    residual_drawdown: npt.ArrayLike,
    rate: float,
    pumping_duration: float,
) -> RecoveryResult:
    """Theis recovery: T from residual drawdown per log cycle of t / t'.

    ``time_since_start`` is t, measured from the start of pumping; t' is the
    time since pumping stopped. Only recovery points (t > duration) are used.
    """

    t = np.asarray(time_since_start, dtype=np.float64)
    s = np.asarray(residual_drawdown, dtype=np.float64)
    after = t > pumping_duration
    if int(after.sum()) < 2:
        raise ValueError("need at least two readings after pumping stopped")
    ratio = t[after] / (t[after] - pumping_duration)
    line = _line(np.log10(ratio), s[after])
    if line.slope <= 0:
        raise ValueError("residual drawdown does not fall as the well recovers")
    return RecoveryResult(2.3026 * rate / (4.0 * math.pi * line.slope), line)


@dataclass(frozen=True)
class StepTestResult:
    """Well losses from a step-drawdown test, s = B Q + C Q**2."""

    aquifer_loss_coefficient: float
    """B: drawdown per unit rate from the formation (laminar)."""
    well_loss_coefficient: float
    """C: the nonlinear well loss (Jacob's C)."""
    rates: tuple[float, ...]
    efficiencies: tuple[float, ...]
    """B Q / (B Q + C Q**2) at each step: the share of drawdown the aquifer causes."""

    def drawdown(self, rate: float) -> float:
        """Predicted drawdown at ``rate``, for choosing a production rate."""

        return self.aquifer_loss_coefficient * rate + self.well_loss_coefficient * rate**2


def step_drawdown(rates: npt.ArrayLike, drawdowns: npt.ArrayLike) -> StepTestResult:
    """Fit s = B Q + C Q**2 to the drawdown reached in each step.

    Give the drawdown each step reached after the same elapsed time in the
    step, or the cumulative drawdown at the end of equal-length steps (the
    Hantush-Bierschenk increments are summed the same way).

    The fit minimises the error in drawdown, which is what was measured.
    The textbook plot of s / Q against Q weights the low-rate steps up by
    1 / Q; on Clark's "Well 1" that line misses the drawdowns by 0.119 m
    where this one misses by 0.096 m.
    """

    q = np.asarray(rates, dtype=np.float64)
    s = np.asarray(drawdowns, dtype=np.float64)
    if q.size < 2 or q.shape != s.shape:
        raise ValueError("need at least two steps, each with a rate and a drawdown")
    if np.any(q <= 0):
        raise ValueError("step rates must be positive")
    (b, c), *_ = np.linalg.lstsq(np.column_stack([q, q * q]), s, rcond=None)
    eff = tuple(float(b * x / (b * x + c * x * x)) for x in q)
    return StepTestResult(float(b), float(c), tuple(float(x) for x in q), eff)
