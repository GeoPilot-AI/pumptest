"""Field units to one consistent set, and water levels to drawdown.

The solvers never convert units (see :mod:`pumptest.data`). Field sheets do
not arrive consistent: pumping in litres per minute, time in minutes, levels
in metres below ground. Convert here first, then build the test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pumptest._types import FloatArray

LENGTH = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
TIME = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
VOLUME = {
    "m3": 1.0,
    "l": 0.001,
    "ft3": 0.3048**3,
    "gal": 0.003785411784,  # US gallon
    "igal": 0.00454609,  # imperial gallon
}
RATE = {
    "m3/d": VOLUME["m3"] / TIME["d"],
    "m3/h": VOLUME["m3"] / TIME["h"],
    "m3/s": 1.0,
    "l/s": VOLUME["l"],
    "lps": VOLUME["l"],
    "l/min": VOLUME["l"] / TIME["min"],
    "lpm": VOLUME["l"] / TIME["min"],
    "l/h": VOLUME["l"] / TIME["h"],
    "lph": VOLUME["l"] / TIME["h"],
    "gpm": VOLUME["gal"] / TIME["min"],
    "igpm": VOLUME["igal"] / TIME["min"],
    "ft3/s": VOLUME["ft3"],
    "cfs": VOLUME["ft3"],
    "ft3/d": VOLUME["ft3"] / TIME["d"],
}
"""Each rate in cubic metres per second."""

_TABLES = (("length", LENGTH), ("time", TIME), ("rate", RATE))


def convert(value: npt.ArrayLike, source: str, target: str) -> FloatArray:
    """Convert ``value`` between two units of the same kind.

    >>> float(convert(100, "lpm", "m3/d"))
    144.0
    """

    known = {unit for _, table in _TABLES for unit in table}
    for unit in (source, target):
        if unit not in known:
            raise ValueError(f"unknown unit {unit!r}")
    for kind, table in _TABLES:
        if source in table and target in table:
            factor = table[source] / table[target]
            converted: FloatArray = np.asarray(value, dtype=np.float64) * factor
            return converted
        if source in table or target in table:
            raise ValueError(f"cannot convert {source!r} to {target!r}: not both {kind} units")
    raise AssertionError("unreachable: both units are known")  # pragma: no cover


def drawdown_from_levels(
    levels: npt.ArrayLike, static_level: float, *, below_datum: bool = True
) -> FloatArray:
    """Drawdown from water levels.

    With ``below_datum`` (the usual depth to water, measured down from ground
    or a measuring point) drawdown is level minus static level. With levels as
    elevations above a datum it is static level minus level.
    """

    lv = np.asarray(levels, dtype=np.float64)
    result: FloatArray = lv - static_level if below_datum else static_level - lv
    return result
