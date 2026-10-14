# Single-well and step tests

Most tests in practice have no observation well: only the drawdown in the
pumped well. That record carries well losses, skin and the water stored in
the casing, and a single well cannot separate storativity from skin. This
tutorial uses the Hoogezand test (Mulder 1983, in Kruseman and de Ridder 1994)
and Clark's step-drawdown test "Well 1".

## Jacob's straight line

```python
from pumptest import cooper_jacob, datasets

test = datasets.load("hoogezand")
obs = test.observations[0]
second = 1 / 86400
cj = cooper_jacob(obs.time, obs.drawdown, rate=test.schedule.first_rate,
                  distance=0.185, start=41 * second)
print(f"T = {cj.transmissivity:.0f} m2/d from a slope of {cj.line.slope:.4f} m per log cycle")
```

The book reads the slope by eye as 0.07 m per log cycle and gets 1800 m²/d.

## Large-diameter well with skin

The Papadopulos-Cooper solution includes the water stored in the casing.
With `skin=True` it adds a skin factor; with `casing_radius=True` it fits the
effective casing radius, because a pump column in the casing leaves less
area for the water level to fall in. Hold S at an independent value, here
Mulder's 4 × 10⁻⁴:

```python
from pumptest import Observation, PapadopulosCooper, PumpingTest, fit

keep = obs.time >= 5 * second
record = PumpingTest(test.well, test.schedule,
                     [Observation("pumped well", 0.0, obs.time[keep], obs.drawdown[keep])])
model = PapadopulosCooper(skin=True, casing_radius=True)
result = fit(record, model, fixed={"S": 4e-4})
print(result.summary())
```

T comes out at 1817 m²/d against the book's 1800, with a skin near 25 and
an effective casing radius of about 0.09 m, half the 0.185 m casing.

!!! note "Dug wells"
    The same model is the standard one for large-diameter dug wells: give
    the well's radius as both `radius` and `casing_radius`, and fit with the
    observation at `distance=0`.

## Step-drawdown test

Give the drawdown each step reached after the same time in the step, or the
cumulative drawdown at the end of equal steps:

```python
from pumptest import step_drawdown

rates, drawdowns = datasets.step_test()
steps = step_drawdown(rates, drawdowns)
print(f"B = {steps.aquifer_loss_coefficient:.3e} d/m2, C = {steps.well_loss_coefficient:.3e} d2/m5")
for q, e in zip(steps.rates, steps.efficiencies):
    print(f"  {q:6.0f} m3/d: well efficiency {e:.0%}")
print(f"drawdown at 6000 m3/d: {steps.drawdown(6000):.1f} m")
```

pumptest fits s = BQ + CQ² by least squares in drawdown, which fits Clark's
data better than the textbook's line (see
[decision 6](../decisions.md)).

## Recovery

When pumping stops, the residual drawdown gives T again without the well
losses. With a schedule that stops, every model fits the recovery directly:

```python
import numpy as np
from pumptest import Schedule, Theis, Well

truth = {"T": 300.0, "S": 1e-4}
t = np.linspace(0.02, 2.0, 60)
template = PumpingTest(Well(0.1), Schedule.constant(400.0, stop=1.0),
                       [Observation("OW", 20.0, t, np.ones_like(t))])
s = Theis().drawdown(truth, template, template.observations[0])
recovery_test = PumpingTest(template.well, template.schedule, [Observation("OW", 20.0, t, s)])
print(fit(recovery_test, Theis()).values)
```

For the hand method, `theis_recovery(t, residual, rate, pumping_duration)`
fits the straight line of residual drawdown against t/t'.
