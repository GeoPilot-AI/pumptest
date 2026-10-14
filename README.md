# pumptest

Pumping test analysis in Python: transmissivity, storativity, leakage,
specific yield and well losses from drawdown data.

- **Solutions:** Theis, Cooper-Jacob, Hantush-Jacob (leaky), Papadopulos-Cooper
  (large-diameter wells, with optional skin), Neuman (unconfined, delayed yield).
- **Tests:** constant rate, recovery and variable-rate schedules (by
  superposition), and step-drawdown well losses.
- **Fitting:** automatic least squares across any number of observation wells
  at once, with standard errors and parameter correlations.
- **Diagnostics:** the Bourdet log-derivative.
- **Validation:** checked against AQTESOLV and published textbook analyses of
  five real tests, and against the published type-curve tables. See
  [docs/validation.md](docs/validation.md).

Runtime dependencies are numpy and scipy only. MIT licence.

## Install

```
pip install pumptest
```

The package is not on PyPI yet; until it is, install from a clone with
`pip install -e .`.

## Example

The Oude Korendijk test (Kruseman and de Ridder): 788 cubic metres per day,
drawdown at a piezometer 30 m away, time in minutes.

```python
import numpy as np

from pumptest import Observation, PumpingTest, Schedule, Theis, Well, fit
from pumptest.units import convert

minutes = np.array([0.1, 0.25, 0.5, 0.7, 1.0, 1.4, 1.9, 2.33, 2.8, 3.36, 4.0, 5.35,
                    6.8, 8.3, 8.7, 10.0, 13.1, 18.0, 27.0, 33.0, 41.0, 48.0, 59.0,
                    80.0, 95.0, 139.0, 181.0, 245.0, 300.0, 360.0, 480.0, 600.0,
                    728.0, 830.0])
drawdown_m = np.array([0.04, 0.08, 0.13, 0.18, 0.23, 0.28, 0.33, 0.36, 0.39, 0.42,
                       0.45, 0.50, 0.54, 0.57, 0.58, 0.60, 0.64, 0.68, 0.742, 0.753,
                       0.779, 0.793, 0.819, 0.855, 0.873, 0.915, 0.935, 0.966, 0.990,
                       1.007, 1.050, 1.053, 1.072, 1.088])

test = PumpingTest(
    well=Well(radius=0.2),
    schedule=Schedule.constant(788.0),  # m3/d
    observations=[Observation("H30", 30.0, convert(minutes, "min", "d"), drawdown_m)],
    thickness=7.0,
)
result = fit(test, Theis())
print(result.summary())
```

Units are never converted inside the solvers: give one consistent set
(metres, days and cubic metres per day here) and the results come back in it.

## Development

```
pip install -e ".[dev,docs]"
python scripts/checks.py
```

Design decisions are recorded in [pm/decisions](pm/decisions/README.md).
