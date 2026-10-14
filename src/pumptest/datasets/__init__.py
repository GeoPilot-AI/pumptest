"""Published pumping tests, bundled so examples and checks run anywhere.

    >>> from pumptest import datasets, fit, Theis
    >>> test = datasets.load("oude_korendijk")
    >>> datasets.published("oude_korendijk").values["T"]
    462.602

Every test is in metres, days and cubic metres per day. Provenance and
licences are in ``data/README.md`` next to this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from pumptest._types import FloatArray
from pumptest.data import Observation, PumpingTest, Schedule, Well

FT = 0.3048
MINUTE = 1.0 / 1440.0
SECOND = 1.0 / 86400.0


@dataclass(frozen=True)
class Published:
    """A published analysis of a bundled test, to compare a fit against."""

    source: str
    model: str
    values: dict[str, float]
    note: str = ""
    derived: dict[str, float] = field(default_factory=dict)


def _read(name: str) -> FloatArray:
    with resources.files(__package__).joinpath("data", name).open("r", encoding="utf-8") as fh:
        table: FloatArray = np.loadtxt(fh, comments="#")
    return table


def _series(name: str, *, scale: float = 1.0) -> tuple[FloatArray, FloatArray]:
    raw = _read(name)
    raw = raw[raw[:, 0] > 0]
    return raw[:, 0] * scale, np.abs(raw[:, 1])


def _oude_korendijk() -> PumpingTest:
    obs = [
        Observation("H30", 30.0, *_series("oudekorendijk_h30.dat", scale=MINUTE)),
        Observation("H90", 90.0, *_series("oudekorendijk_h90.dat", scale=MINUTE)),
    ]
    return PumpingTest(Well(0.2), Schedule.constant(788.0), obs, 7.0, "Oude Korendijk")


def _gridley() -> PumpingTest:
    obs = [
        Observation("Well 1", 251.1552, *_series("gridley_well_1.txt")),
        Observation("Well 3 (pumped)", 0.0, *_series("gridley_well_3.txt")),
    ]
    return PumpingTest(Well(0.1524), Schedule.constant(1199.218), obs, 5.4846, "Gridley")


def _sioux_flats() -> PumpingTest:
    obs = [Observation(f"OW{d}", d * FT, *_series(f"sioux{d}.txt")) for d in (100, 200, 400)]
    return PumpingTest(Well(0.1524), Schedule.constant(6605.754), obs, 15.24, "Sioux Flats")


def _texas_hill() -> PumpingTest:
    obs = [
        Observation("OW1", 12.191, *_series("texas40.txt")),
        Observation("OW2", 24.383, *_series("texas80.txt")),
        Observation("OW3", 48.766, *_series("texas160.txt")),
    ]
    return PumpingTest(Well(0.1524), Schedule.constant(24464.06), obs, 15.24, "Texas Hill")


def _dalem() -> PumpingTest:
    obs = [Observation(f"P{d}", float(d), *_series(f"dalem_p{d}.txt")) for d in (30, 60, 90, 120)]
    return PumpingTest(Well(0.1), Schedule.constant(761.0, stop=0.34), obs, 37.0, "Dalem")


def _vennebulten() -> PumpingTest:
    obs = [Observation("WI1/90 deep", 90.0, *_series("venne_deep.txt", scale=MINUTE))]
    return PumpingTest(Well(0.2), Schedule.constant(873.0), obs, 21.0, "Vennebulten")


def _hoogezand() -> PumpingTest:
    raw = _read("hoogezand.txt")
    obs = [Observation("pumped well", 0.0, raw[:, 0] * SECOND, raw[:, 1])]
    return PumpingTest(Well(0.185), Schedule.constant(28.7 * 24.0), obs, None, "Hoogezand")


_LOADERS = {
    "oude_korendijk": _oude_korendijk,
    "gridley": _gridley,
    "sioux_flats": _sioux_flats,
    "texas_hill": _texas_hill,
    "dalem": _dalem,
    "vennebulten": _vennebulten,
    "hoogezand": _hoogezand,
}

_T_TEXAS, _T_DALEM = 224.726 * 15.24, 45.332 * 37.0
_PUBLISHED = {
    "oude_korendijk": Published(
        "AQTESOLV (Yang 2020)", "Theis", {"T": 66.086 * 7.0, "S": 2.541e-5 * 7.0}
    ),
    "gridley": Published(
        "AQTESOLV (Yang 2020)", "Theis", {"T": 37.803 * 5.4846, "S": 1.356e-6 * 5.4846}
    ),
    "sioux_flats": Published(
        "AQTESOLV (Yang 2020)", "Theis", {"T": 282.659 * 15.24, "S": 4.211e-3 * 15.24}
    ),
    "texas_hill": Published(
        "AQTESOLV (Yang 2020)",
        "Hantush-Jacob",
        {"T": _T_TEXAS, "S": 2.125e-4 * 15.24, "B": math.sqrt(_T_TEXAS * 43.964)},
        derived={"c": 43.964},
    ),
    "dalem": Published(
        "Kruseman and de Ridder (1994)",
        "Hantush-Jacob",
        {"T": _T_DALEM, "S": 4.762e-5 * 37.0, "B": math.sqrt(_T_DALEM * 331.141)},
        derived={"c": 331.141},
    ),
    "vennebulten": Published(
        "Kruseman and de Ridder (1994), type curves by eye",
        "Neuman",
        {"T": 1447.0, "S": 5.2e-4, "Sy": 4.9e-3, "Kd": 0.01 * 21.0**2 / 90.0**2},
        note="T is 1447 m2/d from the type-A match and 1616 from type B; the book "
        "calls Sy unreasonably low.",
    ),
    "hoogezand": Published(
        "Kruseman and de Ridder (1994), Jacob line by eye; S after Mulder (1983)",
        "Cooper-Jacob",
        {"T": 1800.0, "S": 4e-4},
        note="Jacob straight line for t > 41 s; leaky aquifer with c = 1000 d.",
    ),
}


def available() -> tuple[str, ...]:
    """Names accepted by :func:`load`."""

    return tuple(_LOADERS)


def load(name: str) -> PumpingTest:
    """A bundled test, in metres, days and cubic metres per day."""

    try:
        return _LOADERS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(_LOADERS)}") from None


def published(name: str) -> Published:
    """The published analysis of a bundled test."""

    if name not in _PUBLISHED:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(_PUBLISHED)}")
    return _PUBLISHED[name]


def step_test() -> tuple[FloatArray, FloatArray]:
    """Clark's step-drawdown test "Well 1": rates (m3/d) and drawdown 100 min into each step.

    Kruseman and de Ridder publish B = 3.26e-3 d/m2 and C = 1.45e-7 d2/m5.
    """

    raw = _read("well1_steps.txt")
    return raw[:, 0], raw[:, 1]
