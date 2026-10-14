# Getting started

## Install

The package is not on PyPI yet. Install it from GitHub:

```
pip install git+https://github.com/GeoPilot-AI/pumptest
```

It needs Python 3.11 or later, numpy and scipy.

## Units: one consistent set

The solvers work in the units they are given, so lengths, times and rates
must use one consistent set, and the results come back in it. Metres, days and cubic metres
per day is the usual choice, and gives transmissivity in square metres per
day. Convert field units first with `pumptest.units`:

```python
from pumptest.units import convert, drawdown_from_levels

rate = float(convert(250, "lpm", "m3/d"))  # 250 litres per minute = 360 m3/d
minutes = [1, 2, 5, 10, 30, 60]
days = convert(minutes, "min", "d")
drawdown = drawdown_from_levels([12.31, 12.45, 12.70, 12.88, 13.20, 13.41], static_level=12.05)
```

`convert` knows metres, centimetres, millimetres, feet and inches; seconds,
minutes, hours and days; and cubic metres, litres, US and imperial gallons and
cubic feet per second, minute, hour or day (`lpm`, `lps`, `m3/h`, `gpm`,
`cfs` and so on).

## Describe the test

A test is a pumping well, a pumping schedule and one or more observations.

```python
import numpy as np

from pumptest import Observation, PumpingTest, Schedule, Well

t = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])  # days
s = np.array([0.05, 0.11, 0.21, 0.29, 0.37, 0.48, 0.56, 0.64, 0.75])  # metres

test = PumpingTest(
    well=Well(radius=0.1),
    schedule=Schedule.constant(500.0),  # m3/d from time zero
    observations=[Observation("OW-1", distance=25.0, time=t, drawdown=s)],
    thickness=20.0,
)
```

- `distance=0` means the observation is in the pumping well itself.
- `Schedule.constant(500, stop=0.5)` adds recovery from day 0.5;
  `Schedule(((0, 300), (0.1, 600), (0.2, 900)))` is a step test.
- `thickness` is needed by the unconfined solution, and to report K = T/b.

## Fit a model

```python
from pumptest import Theis, fit

result = fit(test, Theis())
print(result.summary())
print(result.values["T"], result.standard_errors["T"])
```

`result.values` holds the estimates, `result.standard_errors` their
one-sigma errors, `result.derived` quantities such as K, `result.rmse` the
misfit in the drawdown unit, and `result.residuals` the misfit at each
observation.

Hold a parameter at a known value with `fixed`, fit only some wells with
`observations`, or give your own starting point with `initial`:

```python
result = fit(test, Theis(), fixed={"S": 2e-4})
```

## Compare models

```python
from pumptest import HantushJacob, Neuman, compare

for r in compare(test, [Theis(), HantushJacob(), Neuman()]):
    print(f"{r.model:15} AICc {r.aicc:8.1f}   RMSE {r.rmse:.4f}")
```

`compare` ranks by AICc, which penalises every extra parameter: a leaky or
unconfined model wins only when it fits enough better to earn its extra
parameters.

## Where next

- [Confined aquifer, several wells](tutorials/confined.md)
- [Choosing the model](tutorials/model-choice.md)
- [Single-well and step tests](tutorials/single-well.md)
- [Your own field data](tutorials/field-data.md)
