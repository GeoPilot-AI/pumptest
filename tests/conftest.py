"""Loaders for the published pumping tests in ``tests/data/ttim``.

The data files come from the TTim documentation (MIT licence, see
``tests/data/ttim/README.md``). Heads there are negative drawdowns; times are
in days except where a loader converts minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pumptest import Observation, PumpingTest, Schedule, Well

DATA = Path(__file__).parent / "data" / "ttim"
FT = 0.3048


def series(name: str, *, minutes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(DATA / name, comments="#")
    t = raw[:, 0] / 1440.0 if minutes else raw[:, 0]
    return t, np.abs(raw[:, 1])


@pytest.fixture
def oude_korendijk() -> PumpingTest:
    obs = [
        Observation("H30", 30.0, *series("oudekorendijk_h30.dat", minutes=True)),
        Observation("H90", 90.0, *series("oudekorendijk_h90.dat", minutes=True)),
    ]
    return PumpingTest(Well(0.2), Schedule.constant(788.0), obs, thickness=7.0)


@pytest.fixture
def gridley() -> PumpingTest:
    obs = [
        Observation("Well 1", 251.1552, *series("gridley_well_1.txt")),
        Observation("Well 3 (pumped)", 0.0, *series("gridley_well_3.txt")),
    ]
    return PumpingTest(Well(0.1524), Schedule.constant(1199.218), obs, thickness=5.4846)


@pytest.fixture
def sioux() -> PumpingTest:
    obs = [Observation(f"OW{d}", d * FT, *series(f"sioux{d}.txt")) for d in (100, 200, 400)]
    return PumpingTest(Well(0.1524), Schedule.constant(6605.754), obs, thickness=15.24)


@pytest.fixture
def texas_hill() -> PumpingTest:
    obs = [
        Observation("OW1", 12.191, *series("texas40.txt")),
        Observation("OW2", 24.383, *series("texas80.txt")),
        Observation("OW3", 48.766, *series("texas160.txt")),
    ]
    return PumpingTest(Well(0.1524), Schedule.constant(24464.06), obs, thickness=15.24)


@pytest.fixture
def dalem() -> PumpingTest:
    obs = [Observation(f"P{d}", float(d), *series(f"dalem_p{d}.txt")) for d in (30, 60, 90, 120)]
    return PumpingTest(Well(0.1), Schedule.constant(761.0, stop=0.34), obs, thickness=37.0)
