"""What a pumping test is: a well, a pumping schedule and the observations.

Units are never converted inside the package. Give lengths, times and rates in
one consistent set (metres, days and cubic metres per day is the usual one) and
the fitted parameters come back in it: transmissivity in square metres per
day. :mod:`pumptest.units` converts field units before they get here.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pumptest._types import FloatArray


@dataclass(frozen=True)
class Schedule:
    """Pumping rate as a step function: ``(start_time, rate)`` pairs.

    Recovery is a step to zero. A step-drawdown test is a list of rising steps.
    Rates are positive for pumping (abstraction) and the first step starts at
    the time origin of the observations.
    """

    steps: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a schedule needs at least one step")
        starts = [t for t, _ in self.steps]
        if any(b <= a for a, b in itertools.pairwise(starts)):
            raise ValueError("schedule steps must start at strictly increasing times")

    @classmethod
    def constant(cls, rate: float, *, stop: float | None = None) -> Schedule:
        """A constant-rate test, optionally followed by recovery from ``stop``."""

        steps = [(0.0, rate)] if stop is None else [(0.0, rate), (stop, 0.0)]
        return cls(tuple(steps))

    @property
    def first_rate(self) -> float:
        return self.steps[0][1]

    def rate_at(self, t: float) -> float:
        """The rate in force at time ``t`` (zero before the first step)."""

        rate = 0.0
        for start, q in self.steps:
            if t >= start:
                rate = q
        return rate

    def superpose(
        self, unit_response: Callable[[FloatArray], FloatArray], t: npt.ArrayLike
    ) -> FloatArray:
        """Drawdown at ``t`` from the drawdown per unit rate of a step at time zero.

        Every solution here is linear in the rate, so a schedule is the sum of
        its rate changes, each delayed to its own start.
        """

        times = np.asarray(t, dtype=np.float64)
        total = np.zeros_like(times)
        previous = 0.0
        for start, rate in self.steps:
            change = rate - previous
            previous = rate
            if change == 0.0:
                continue
            after = times > start
            if np.any(after):
                total[after] += change * unit_response(times[after] - start)
        return total


@dataclass(frozen=True)
class Well:
    """The pumping well. ``casing_radius`` sets wellbore storage (defaults to ``radius``)."""

    radius: float
    casing_radius: float | None = None
    name: str = "Pumping well"

    @property
    def storage_radius(self) -> float:
        return self.radius if self.casing_radius is None else self.casing_radius


@dataclass(frozen=True)
class Observation:
    """Drawdown measured at one place: an observation well or the pumping well.

    ``distance`` zero means the pumping well itself. ``screen`` is the
    observation interval as heights above the aquifer base, used only by the
    unconfined solution; ``None`` is a fully penetrating well.
    """

    name: str
    distance: float
    time: FloatArray
    drawdown: FloatArray
    screen: tuple[float, float] | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        t = np.asarray(self.time, dtype=np.float64)
        s = np.asarray(self.drawdown, dtype=np.float64)
        if t.shape != s.shape or t.ndim != 1:
            raise ValueError(f"{self.name}: time and drawdown must be 1-D and the same length")
        if np.any(t <= 0):
            raise ValueError(f"{self.name}: times must be after the start of pumping")
        if self.distance < 0:
            raise ValueError(f"{self.name}: distance cannot be negative")
        order = np.argsort(t)
        object.__setattr__(self, "time", t[order])
        object.__setattr__(self, "drawdown", s[order])

    @property
    def at_pumping_well(self) -> bool:
        return self.distance == 0.0


@dataclass(frozen=True)
class PumpingTest:
    """A whole test. ``thickness`` is required by the unconfined solution only."""

    well: Well
    schedule: Schedule
    observations: tuple[Observation, ...]
    thickness: float | None = None
    name: str = "Pumping test"
    notes: dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        well: Well,
        schedule: Schedule,
        observations: Sequence[Observation],
        thickness: float | None = None,
        name: str = "Pumping test",
        notes: dict[str, str] | None = None,
    ) -> None:
        if not observations:
            raise ValueError("a pumping test needs at least one observation")
        names = [o.name for o in observations]
        if len(set(names)) != len(names):
            raise ValueError("observation names must be unique")
        object.__setattr__(self, "well", well)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "observations", tuple(observations))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "notes", dict(notes or {}))

    def radius_of(self, observation: Observation) -> float:
        """The radial distance a solution should use for ``observation``."""

        return self.well.radius if observation.at_pumping_well else observation.distance
