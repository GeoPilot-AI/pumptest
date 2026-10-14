"""Published pumping tests, bundled so examples and checks run anywhere.

    >>> from pumptest import datasets, fit, Theis
    >>> test = datasets.load("sioux_flats")
    >>> round(datasets.published("sioux_flats").values["T"], 1)
    4307.7

Every test is in metres, days and cubic metres per day. Provenance and
licences are in ``data/README.md`` next to this module.
"""

from __future__ import annotations

import gzip
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from importlib import resources

import numpy as np

from pumptest._types import FloatArray
from pumptest.data import Observation, PumpingTest, Schedule, Well

FT = 0.3048
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


def _series(name: str) -> tuple[FloatArray, FloatArray]:
    raw = _read(name)
    raw = raw[raw[:, 0] > 0]
    return raw[:, 0], np.abs(raw[:, 1])


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


_LAUSWIESEN_XY = {
    "B1": (67.715, 18.863),
    "B2": (63.877, 22.201),
    "B3": (71.009, 28.88),
    "B4": (48.742, 20.038),
    "B5": (48.908, 35.283),
}
_LAUSWIESEN_RATE = {"B2": 0.00653, "B3": 0.00594, "B4": 0.00625, "B5": 0.00604}  # m3/s


def _lauswiesen(test: str) -> PumpingTest:
    path = resources.files(__package__).joinpath("data", f"lauswiesen_{test}.txt.gz")
    with path.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        table: FloatArray = np.loadtxt(fh, comments="#")
    table = table[table[:, 0] > 0]
    x0, y0 = _LAUSWIESEN_XY[test]
    obs = [
        Observation(
            well,
            0.0 if well == test else math.hypot(x - x0, y - y0),
            table[:, 0] * SECOND,
            table[:, column],
        )
        for column, (well, (x, y)) in enumerate(_LAUSWIESEN_XY.items(), start=1)
    ]
    rate = _LAUSWIESEN_RATE[test] * 86400.0
    return PumpingTest(Well(0.0762), Schedule.constant(rate), obs, 5.0, f"Lauswiesen {test}")


_LOADERS: dict[str, Callable[[], PumpingTest]] = {
    "gridley": _gridley,
    "sioux_flats": _sioux_flats,
    "texas_hill": _texas_hill,
    **{f"lauswiesen_{t.lower()}": partial(_lauswiesen, t) for t in _LAUSWIESEN_RATE},
}

# Theis estimates by welltestpy, from the archived workflow of Mueller et al.
# (2022): mean ln T (T in m2/s) and ln S, fitted to every well of each test.
_WELLTESTPY = {
    "B2": (-4.041889357980989, -2.135803320654749),
    "B3": (-4.599175133715375, -2.087130985265375),
    "B4": (-4.469906947665516, -2.558349156725309),
    "B5": (-3.687795829848876, -3.341085358753064),
}

_T_TEXAS = 224.723 * 15.24
_PUBLISHED = {
    "gridley": Published(
        "AQTESOLV (Yang 2020)", "Theis", {"T": 37.803 * 5.4846, "S": 1.356e-6 * 5.4846}
    ),
    "sioux_flats": Published(
        "AQTESOLV (Yang 2020)", "Theis", {"T": 282.659 * 15.24, "S": 4.211e-3 * 15.24}
    ),
    "texas_hill": Published(
        "AQTESOLV (Yang 2020)",
        "Hantush-Jacob",
        {"T": _T_TEXAS, "S": 2.125e-4 * 15.24, "B": math.sqrt(_T_TEXAS * 43.965)},
        derived={"c": 43.965},
    ),
    **{
        f"lauswiesen_{t.lower()}": Published(
            "welltestpy (Mueller et al. 2022)",
            "Theis",
            {"T": math.exp(ln_t) * 86400.0, "S": math.exp(ln_s)},
            note="Fitted to all five wells, the pumped well included, after welltestpy's "
            "smoothing and resampling to ten times per well.",
        )
        for t, (ln_t, ln_s) in _WELLTESTPY.items()
    },
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
