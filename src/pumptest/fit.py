"""Automatic estimation: nonlinear least squares over every observation at once.

Positive parameters are estimated as log10 values, which is what makes one
optimiser work for transmissivities of 1e-3 and storativities of 1e-6 in the
same problem. Several starting points are tried and the lowest residual wins,
because the leaky and unconfined families have shallow valleys where a single
start stops early.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from pumptest._types import FloatArray
from pumptest.data import Observation, PumpingTest
from pumptest.models import Model, Parameter
from pumptest.straightline import cooper_jacob

DIFF_STEP = 1e-4
"""Relative finite-difference step for the Jacobian. Stehfest inversion
carries round-off near 1e-7 relative; scipy's default step (about 1.5e-8) sits
under that noise and the optimiser stops on its first iteration."""


@dataclass(frozen=True)
class FitResult:
    """The estimate and how far to trust it."""

    model: str
    values: dict[str, float]
    standard_errors: dict[str, float]
    """Linearised one-sigma errors in the units of each parameter."""
    derived: dict[str, float]
    rmse: float
    """Root mean square of (observed - computed), unweighted, over all points."""
    residuals: dict[str, FloatArray]
    correlation: FloatArray
    """Correlation of the estimated parameters, in the order of ``estimated``."""
    estimated: tuple[str, ...]
    points: int
    success: bool
    message: str

    def summary(self) -> str:
        """A plain-text table of the result, for a terminal or a report."""

        lines = [f"{self.model} fit to {self.points} points, RMSE {self.rmse:.4g}"]
        for name, value in self.values.items():
            err = self.standard_errors.get(name)
            spread = f" +/- {err:.3g}" if err is not None and math.isfinite(err) else " (fixed)"
            lines.append(f"  {name:>8} = {value:.5g}{spread}")
        for name, value in self.derived.items():
            lines.append(f"  {name:>8} = {value:.5g} (derived)")
        if not self.success:
            lines.append(f"  warning: {self.message}")
        return "\n".join(lines)


def _to_x(parameter: Parameter, value: float) -> float:
    return math.log10(value) if parameter.log else value


def _from_x(parameter: Parameter, x: float) -> float:
    return float(10.0**x) if parameter.log else float(x)


def _guide_observation(test: PumpingTest) -> Observation:
    # An observation well gives a sounder S than the pumping well, whose
    # drawdown carries well loss; among them the longest record is the best.
    away = [o for o in test.observations if not o.at_pumping_well]
    pool = away or list(test.observations)
    return max(pool, key=lambda o: o.time.size)


def starting_points(test: PumpingTest, model: Model) -> list[dict[str, float]]:
    """Initial guesses: Cooper-Jacob for T and S, a spread for the rest."""

    guide = _guide_observation(test)
    rate = test.schedule.first_rate
    try:
        cj = cooper_jacob(guide.time, guide.drawdown, rate, test.radius_of(guide))
        trans, stor = cj.transmissivity, cj.storativity
    except ValueError:
        trans, stor = rate / (4.0 * math.pi * max(float(np.max(guide.drawdown)), 1e-9)), 1e-4
    stor = min(max(stor, 1e-8), 0.3)
    reach = max(test.radius_of(o) for o in test.observations)
    spreads: dict[str, list[float]] = {
        "T": [trans],
        "S": [stor],
        "B": [10.0 * reach, 100.0 * reach, 1000.0 * reach],
        "Sk": [0.0, 5.0],
    }
    if "Sy" in model.parameter_names:
        # On an unconfined record the late Cooper-Jacob storativity is mostly Sy.
        spreads["S"] = [stor / 100.0]
        spreads["Sy"] = [min(max(stor, 1e-3), 0.4), 0.1]
        spreads["Kd"] = [0.01, 0.1, 1.0]
    names = model.parameter_names
    grid = itertools.product(*(spreads[n] for n in names))
    return [dict(zip(names, combo, strict=True)) for combo in grid]


def fit(
    test: PumpingTest,
    model: Model,
    *,
    initial: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    fixed: Mapping[str, float] | None = None,
    observations: Iterable[str] | None = None,
) -> FitResult:
    """Estimate ``model``'s parameters from ``test``.

    ``fixed`` holds parameters at given values. ``initial`` replaces the
    automatic starting points with one guess or a list of them.
    ``observations`` restricts the fit to the named observations.
    """

    model.check(test)
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(model.parameter_names)
    if unknown:
        raise ValueError(f"{model.name} has no parameter(s) {sorted(unknown)}")
    chosen = list(test.observations)
    if observations is not None:
        wanted = set(observations)
        chosen = [o for o in chosen if o.name in wanted]
        if not chosen:
            raise ValueError("none of the named observations are in the test")
    free = [p for p in model.parameters if p.name not in fixed]
    if not free:
        raise ValueError("every parameter is fixed; nothing to estimate")

    if initial is None:
        starts = starting_points(test, model)
    elif isinstance(initial, Mapping):
        starts = [dict(initial)]
    else:
        starts = [dict(g) for g in initial]
    defaults = starting_points(test, model)[0]

    observed = np.concatenate([o.drawdown for o in chosen])
    weights = np.concatenate([np.full(o.time.size, math.sqrt(o.weight)) for o in chosen])

    def values_of(x: FloatArray) -> dict[str, float]:
        values = dict(fixed)
        for parameter, xi in zip(free, x, strict=True):
            values[parameter.name] = _from_x(parameter, float(xi))
        return values

    def computed(values: Mapping[str, float]) -> FloatArray:
        return np.concatenate([model.drawdown(values, test, o) for o in chosen])

    def residual(x: FloatArray) -> FloatArray:
        out: FloatArray = (observed - computed(values_of(x))) * weights
        return np.where(np.isfinite(out), out, 1e6)

    lower = np.array([_to_x(p, p.lower) for p in free])
    upper = np.array([_to_x(p, p.upper) for p in free])
    best: optimize.OptimizeResult | None = None
    for start in starts:
        guess = {**defaults, **start}
        x0 = np.clip([_to_x(p, guess[p.name]) for p in free], lower + 1e-9, upper - 1e-9)
        result = optimize.least_squares(
            residual, x0, bounds=(lower, upper), method="trf", diff_step=DIFF_STEP
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None

    values = values_of(best.x)
    raw = observed - computed(values)
    n, k = observed.size, len(free)
    jac = np.asarray(best.jac, dtype=np.float64)
    sigma2 = 2.0 * float(best.cost) / max(n - k, 1)
    try:
        cov = np.linalg.inv(jac.T @ jac) * sigma2
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    sd = np.sqrt(np.abs(np.diag(cov)))
    errors = {
        p.name: (values[p.name] * math.log(10.0) * float(e) if p.log else float(e))
        for p, e in zip(free, sd, strict=True)
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)

    ordered = {name: values[name] for name in model.parameter_names}
    offsets = np.cumsum([0] + [o.time.size for o in chosen])
    return FitResult(
        model=model.name,
        values=ordered,
        standard_errors=errors,
        derived=model.derived(ordered, test),
        rmse=float(np.sqrt(np.mean(raw**2))),
        residuals={o.name: raw[a:b] for o, a, b in zip(chosen, offsets, offsets[1:], strict=False)},
        correlation=corr,
        estimated=tuple(p.name for p in free),
        points=int(n),
        success=bool(best.success),
        message=str(best.message),
    )
