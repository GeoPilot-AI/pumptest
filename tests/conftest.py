"""Fixtures: the published pumping tests bundled in :mod:`pumptest.datasets`."""

from __future__ import annotations

import pytest

from pumptest import PumpingTest, datasets


@pytest.fixture
def gridley() -> PumpingTest:
    return datasets.load("gridley")


@pytest.fixture
def sioux() -> PumpingTest:
    return datasets.load("sioux_flats")


@pytest.fixture
def texas_hill() -> PumpingTest:
    return datasets.load("texas_hill")
