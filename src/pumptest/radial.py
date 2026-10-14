"""Radial flow to a well in a confined or leaky aquifer, in the Laplace domain.

One kernel covers the family AQTESOLV lists as separate solutions, because each
is a special case of the same boundary-value problem:

- Theis (1935): line sink, no leakage.
- Hantush and Jacob (1955): leakage through an aquitard without storage.
- Papadopulos and Cooper (1967): finite well radius with wellbore storage.
- Agarwal et al. (1970): an infinitesimal skin at the well face.

The kernel returns drawdown per unit pumping rate. Consistent units are the
caller's responsibility: metres and days in, square metres per day out.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from pumptest._types import FloatArray


def radial_drawdown_laplace(
    p: FloatArray,
    r: float,
    transmissivity: float,
    storativity: float,
    *,
    leakage_factor: float = math.inf,
    well_radius: float = 0.0,
    casing_radius: float = 0.0,
    skin: float = 0.0,
    at_pumping_well: bool = False,
) -> FloatArray:
    """Laplace transform of drawdown per unit rate at distance ``r``.

    With ``well_radius`` zero the well is a line sink and wellbore storage and
    skin have no meaning; they are ignored. ``leakage_factor`` is B, with
    B squared equal to T times the aquitard resistance.
    """

    t, s = transmissivity, storativity
    leak = 0.0 if math.isinf(leakage_factor) else 1.0 / (leakage_factor * leakage_factor)
    q = np.sqrt(s * p / t + leak)
    if well_radius <= 0.0:
        line: FloatArray = special.k0(q * r) / (2.0 * math.pi * t * p)
        return line

    rw = well_radius
    # Scaled Bessel functions carry the exp(-x) factor separately, so a distant
    # observation well at early time underflows to zero rather than to 0/0.
    k0w = special.k0e(q * rw)
    k1w = special.k1e(q * rw)
    flux = rw * q * k1w
    face = k0w + skin * flux
    denominator = p * (2.0 * math.pi * t * flux + math.pi * casing_radius**2 * p * face)
    if at_pumping_well:
        well: FloatArray = face / denominator
        return well
    rr = max(r, rw)
    obs: FloatArray = special.k0e(q * rr) * np.exp(-q * (rr - rw)) / denominator
    return obs
