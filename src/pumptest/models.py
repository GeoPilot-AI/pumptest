"""The aquifer models a test is fitted to.

Each model knows its parameters, their bounds, and the drawdown per unit rate
at one observation point. :meth:`Model.drawdown` superposes that over the
pumping schedule, so recovery and step tests need nothing model-specific.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pumptest import laplace
from pumptest._types import FloatArray
from pumptest.data import Observation, PumpingTest
from pumptest.neuman import neuman_drawdown_laplace
from pumptest.radial import radial_drawdown_laplace
from pumptest.wellfunctions import theis_w

Response = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class Parameter:
    """A fitted quantity. ``log`` parameters are estimated on a log10 scale."""

    name: str
    description: str
    dimension: str
    lower: float
    upper: float
    log: bool = True


TRANSMISSIVITY = Parameter("T", "Transmissivity", "L2/T", 1e-8, 1e8)
STORATIVITY = Parameter("S", "Storativity (elastic)", "-", 1e-10, 1.0)


class Model(ABC):
    """An analytical solution for drawdown around a pumped well."""

    name: str
    reference: str
    parameters: tuple[Parameter, ...]

    @abstractmethod
    def unit_response(
        self, values: Mapping[str, float], test: PumpingTest, observation: Observation
    ) -> Response:
        """Drawdown per unit rate, for a rate switched on at time zero."""

    def check(self, test: PumpingTest) -> None:  # noqa: B027
        """Raise ``ValueError`` if ``test`` lacks something this model needs.

        Most models need nothing beyond the test itself, so the default accepts.
        """

    def derived(self, values: Mapping[str, float], test: PumpingTest) -> dict[str, float]:
        """Quantities that follow from the fitted ones, such as K = T / b."""

        out: dict[str, float] = {}
        if test.thickness:
            out["K"] = values["T"] / test.thickness
            out["Ss"] = values["S"] / test.thickness
        return out

    def drawdown(
        self,
        values: Mapping[str, float],
        test: PumpingTest,
        observation: Observation,
        t: npt.ArrayLike | None = None,
    ) -> FloatArray:
        """Drawdown at ``observation`` over the whole pumping schedule."""

        times = observation.time if t is None else np.asarray(t, dtype=np.float64)
        response = self.unit_response(values, test, observation)
        return test.schedule.superpose(response, times)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)


class Theis(Model):
    """Confined aquifer, line-sink well: Theis (1935)."""

    name = "Theis"
    reference = "Theis, C.V. (1935). Trans. Am. Geophys. Union 16, 519-524."
    parameters = (TRANSMISSIVITY, STORATIVITY)

    def unit_response(
        self, values: Mapping[str, float], test: PumpingTest, observation: Observation
    ) -> Response:
        t_, s_ = values["T"], values["S"]
        r = test.radius_of(observation)

        def response(t: FloatArray) -> FloatArray:
            return theis_w(r * r * s_ / (4.0 * t_ * t)) / (4.0 * math.pi * t_)

        return response


class HantushJacob(Model):
    """Leaky confined aquifer, aquitard without storage: Hantush and Jacob (1955).

    Leakage is fitted as the leakage factor B; the aquitard resistance
    c = B**2 / T and the leakance K'/b' = 1 / c are reported as derived values.
    """

    name = "Hantush-Jacob"
    reference = "Hantush, M.S. and Jacob, C.E. (1955). Trans. Am. Geophys. Union 36, 95-100."
    parameters = (
        TRANSMISSIVITY,
        STORATIVITY,
        Parameter("B", "Leakage factor", "L", 1e-3, 1e8),
    )

    def unit_response(
        self, values: Mapping[str, float], test: PumpingTest, observation: Observation
    ) -> Response:
        t_, s_, b_ = values["T"], values["S"], values["B"]
        r = test.radius_of(observation)

        def response(t: FloatArray) -> FloatArray:
            return laplace.invert(
                lambda p: radial_drawdown_laplace(p, r, t_, s_, leakage_factor=b_), t
            )

        return response

    def derived(self, values: Mapping[str, float], test: PumpingTest) -> dict[str, float]:
        out = super().derived(values, test)
        resistance = values["B"] ** 2 / values["T"]
        out["c"] = resistance
        out["leakance"] = 1.0 / resistance
        return out


class PapadopulosCooper(Model):
    """Large-diameter well with wellbore storage: Papadopulos and Cooper (1967).

    The usual solution for dug wells. The screen and casing radii come from
    the :class:`~pumptest.data.Well`. With ``skin=True`` a dimensionless skin
    factor (Agarwal et al., 1970) is fitted as well. With
    ``casing_radius=True`` the effective casing radius rc is fitted instead of
    taken from the well: a pump column or riser in the casing shrinks the
    area the water level falls in, which shows as early drawdown faster than
    the nominal casing allows.
    """

    name = "Papadopulos-Cooper"
    reference = "Papadopulos, I.S. and Cooper, H.H. (1967). Water Resour. Res. 3(1), 241-244."

    def __init__(self, *, skin: bool = False, casing_radius: bool = False) -> None:
        self.skin = skin
        self.fit_casing = casing_radius
        extra: tuple[Parameter, ...] = ()
        if skin:
            extra += (Parameter("Sk", "Skin factor", "-", -5.0, 100.0, log=False),)
        if casing_radius:
            extra += (Parameter("rc", "Effective casing radius", "L", 1e-4, 10.0),)
        self.parameters = (TRANSMISSIVITY, STORATIVITY, *extra)
        options = [label for on, label in ((skin, "skin"), (casing_radius, "fitted rc")) if on]
        if options:
            self.name = f"Papadopulos-Cooper with {' and '.join(options)}"

    def unit_response(
        self, values: Mapping[str, float], test: PumpingTest, observation: Observation
    ) -> Response:
        t_, s_ = values["T"], values["S"]
        skin = values.get("Sk", 0.0)
        rw = test.well.radius
        rc = values.get("rc", test.well.storage_radius)
        r = test.radius_of(observation)
        at_well = observation.at_pumping_well

        def response(t: FloatArray) -> FloatArray:
            return laplace.invert(
                lambda p: radial_drawdown_laplace(
                    p,
                    r,
                    t_,
                    s_,
                    well_radius=rw,
                    casing_radius=rc,
                    skin=skin,
                    at_pumping_well=at_well,
                ),
                t,
            )

        return response


class Neuman(Model):
    """Unconfined aquifer with delayed water-table response: Neuman (1974).

    Fits T, the elastic storativity S, specific yield Sy and the anisotropy
    ratio Kd = Kz / Kr. Needs the saturated thickness on the test.
    """

    name = "Neuman"
    reference = "Neuman, S.P. (1974). Water Resour. Res. 10(2), 303-312."
    parameters = (
        TRANSMISSIVITY,
        STORATIVITY,
        Parameter("Sy", "Specific yield", "-", 1e-5, 0.6),
        Parameter("Kd", "Anisotropy ratio Kz/Kr", "-", 1e-6, 1e3),
    )

    def check(self, test: PumpingTest) -> None:
        if not test.thickness or test.thickness <= 0:
            raise ValueError("the Neuman solution needs the saturated thickness of the aquifer")

    def unit_response(
        self, values: Mapping[str, float], test: PumpingTest, observation: Observation
    ) -> Response:
        self.check(test)
        b = float(test.thickness or 0.0)
        t_, s_, sy, kd = values["T"], values["S"], values["Sy"], values["Kd"]
        r = test.radius_of(observation)

        def response(t: FloatArray) -> FloatArray:
            return laplace.invert(
                lambda p: neuman_drawdown_laplace(
                    p, r, t_, s_, sy, kd, b, screen=observation.screen
                ),
                t,
            )

        return response

    def derived(self, values: Mapping[str, float], test: PumpingTest) -> dict[str, float]:
        out = super().derived(values, test)
        if test.thickness:
            out["Kz"] = values["Kd"] * values["T"] / test.thickness
        return out


MODELS: dict[str, Callable[[], Model]] = {
    "theis": Theis,
    "hantush-jacob": HantushJacob,
    "papadopulos-cooper": PapadopulosCooper,
    "neuman": Neuman,
}
"""Model factories by name, for callers that pick a solution from a string."""
