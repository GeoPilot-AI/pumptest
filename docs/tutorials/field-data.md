# Your own field data

A typical field sheet: depth to water in metres below ground at irregular
minutes, a rate in litres per minute read off a flow meter, and an
observation well some metres away. This page turns that into a test.

## From a CSV

Suppose `field.csv` holds `minutes,depth_m` with the static level as the
first row. Here the file is written first so the example runs:

```python
from pathlib import Path

Path("field.csv").write_text(
    "minutes,depth_m\n0,8.42\n1,8.61\n2,8.70\n4,8.81\n8,8.93\n15,9.03\n30,9.15\n"
    "60,9.27\n120,9.38\n240,9.50\n480,9.61\n"
)
```

```python
import numpy as np

from pumptest import Observation, PumpingTest, Schedule, Theis, Well, fit
from pumptest.units import convert, drawdown_from_levels

raw = np.loadtxt("field.csv", delimiter=",", skiprows=1)
static = raw[0, 1]
minutes, depth = raw[1:, 0], raw[1:, 1]

test = PumpingTest(
    well=Well(radius=0.075),  # a 150 mm borewell
    schedule=Schedule.constant(float(convert(600, "lpm", "m3/d"))),
    observations=[
        Observation(
            "OW-1",
            distance=15.0,
            time=convert(minutes, "min", "d"),
            drawdown=drawdown_from_levels(depth, static),
        )
    ],
    thickness=30.0,
)
print(fit(test, Theis()).summary())
```

## A rate that changed

Flow meters drift and pumps get throttled. Enter each change as a step: the
solution superposes them.

```python
schedule = Schedule(((0.0, 864.0), (0.05, 820.0), (0.2, 790.0)))
varying = PumpingTest(test.well, schedule, test.observations, thickness=30.0)
print(fit(varying, Theis()).values)
```

## Checklist before you trust a fit

- **Time zero** is the moment pumping started, not the first reading.
- **Drawdown** is positive and measured from the static level before pumping.
- **Units** are one consistent set; the result is in that set.
- **Look at the derivative** before choosing a model
  ([Choosing the model](model-choice.md)).
- **Look at the residuals** (`result.residuals`) after fitting: a pattern in
  them means the model is missing something.
