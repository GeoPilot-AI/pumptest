"""Pumping test analysis: aquifer parameters from drawdown.

    >>> from pumptest import PumpingTest, Well, Schedule, Observation, Theis, fit
    >>> result = fit(test, Theis())  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP

Units are the caller's to keep consistent; see :mod:`pumptest.units`.
"""

from pumptest import datasets
from pumptest.data import Observation, PumpingTest, Schedule, Well
from pumptest.diagnostics import log_derivative
from pumptest.fit import FitResult, compare, fit
from pumptest.models import (
    MODELS,
    HantushJacob,
    Model,
    Neuman,
    PapadopulosCooper,
    Parameter,
    Theis,
)
from pumptest.straightline import cooper_jacob, step_drawdown, theis_recovery
from pumptest.wellfunctions import hantush_w, theis_w

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "FitResult",
    "HantushJacob",
    "Model",
    "Neuman",
    "Observation",
    "PapadopulosCooper",
    "Parameter",
    "PumpingTest",
    "Schedule",
    "Theis",
    "Well",
    "__version__",
    "compare",
    "cooper_jacob",
    "datasets",
    "fit",
    "hantush_w",
    "log_derivative",
    "step_drawdown",
    "theis_recovery",
    "theis_w",
]
