"""Fixtures: the published pumping tests bundled in :mod:`pumptest.datasets`."""

from __future__ import annotations

import pytest

from pumptest import PumpingTest, datasets


@pytest.fixture
def oude_korendijk() -> PumpingTest:
    return datasets.load("oude_korendijk")


@pytest.fixture
def gridley() -> PumpingTest:
    return datasets.load("gridley")


@pytest.fixture
def sioux() -> PumpingTest:
    return datasets.load("sioux_flats")


@pytest.fixture
def texas_hill() -> PumpingTest:
    return datasets.load("texas_hill")


@pytest.fixture
def dalem() -> PumpingTest:
    return datasets.load("dalem")


@pytest.fixture
def vennebulten() -> PumpingTest:
    return datasets.load("vennebulten")


@pytest.fixture
def hoogezand() -> PumpingTest:
    return datasets.load("hoogezand")
