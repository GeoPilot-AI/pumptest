# Single-well and step tests

Many tests have no observation well: only the drawdown in the pumped well.
That record carries well losses, skin and the water stored in the casing,
and a single well cannot separate storativity from skin. The examples below
use synthetic records, made with the solutions themselves, so the true
values are known.

## A synthetic single-well test

```python
import numpy as np
from pumptest import Observation, PapadopulosCooper, PumpingTest, Schedule, Well

second = 1 / 86400
truth = {"T": 1800.0, "S": 4e-4, "Sk": 25.0, "rc": 0.09}
t = np.geomspace(5 * second, 750 * second, 25)
model = PapadopulosCooper(skin=True, casing_radius=True)
template = PumpingTest(Well(0.185), Schedule.constant(690.0),
                       [Observation("pumped well", 0.0, t, np.ones_like(t))])
s = model.drawdown(truth, template, template.observations[0])
s = s + np.random.default_rng(2).normal(0.0, 0.003, t.size)  # 3 mm of noise
test = PumpingTest(Well(0.185), Schedule.constant(690.0), [Observation("pumped well", 0.0, t, s)])
```

## Jacob's straight line

```python
from pumptest import cooper_jacob

cj = cooper_jacob(t, s, rate=690.0, distance=0.185, start=60 * second)
print(f"T = {cj.transmissivity:.0f} m2/d from a slope of {cj.line.slope:.4f} m per log cycle")
```

The straight line from 60 s gives 1657 m²/d, 8 percent below the 1800 m²/d
the record was made with.

## Large-diameter well with skin

The Papadopulos-Cooper solution includes the water stored in the casing.
With `skin=True` it adds a skin factor; with `casing_radius=True` it fits the
effective casing radius, because a pump column in the casing leaves less
area for the water level to fall in. Hold S at an independent value:

```python
from pumptest import fit

result = fit(test, model, fixed={"S": 4e-4})
print(result.summary())
```

The fit recovers T = 1822 m²/d, a skin of 25.4 and rc = 0.090 m. The
correlation between T and the skin is 1.00: from a pumped well alone the two
are not separately determined, and the standard errors understate that.

!!! note "Dug wells"
    The same model is the standard one for large-diameter dug wells: give
    the well's radius as both `radius` and `casing_radius`, and fit with the
    observation at `distance=0`.

## Step-drawdown test

Give the drawdown each step reached after the same time in the step, or the
cumulative drawdown at the end of equal steps. Here the drawdowns follow
s = BQ + CQ² with B = 3.2 × 10⁻³ d/m² and C = 1.5 × 10⁻⁷ d²/m⁵, plus 0.1 m of
noise:

```python
from pumptest import step_drawdown

rates = np.array([1300.0, 1700.0, 2400.0, 3300.0, 4100.0, 5000.0])
drawdowns = 3.2e-3 * rates + 1.5e-7 * rates**2
drawdowns = drawdowns + np.random.default_rng(3).normal(0.0, 0.1, rates.size)
steps = step_drawdown(rates, drawdowns)
print(f"B = {steps.aquifer_loss_coefficient:.3e} d/m2, C = {steps.well_loss_coefficient:.3e} d2/m5")
for q, e in zip(steps.rates, steps.efficiencies):
    print(f"  {q:6.0f} m3/d: well efficiency {e:.0%}")
print(f"drawdown at 6000 m3/d: {steps.drawdown(6000):.1f} m")
```

pumptest fits s = BQ + CQ² by least squares in drawdown, the quantity that
was measured, rather than the usual regression of s/Q on Q (see
[decision 6](../decisions.md)).

## Recovery

When pumping stops, the residual drawdown gives T again without the well
losses. With a schedule that stops, every model fits the recovery directly:

```python
from pumptest import Theis

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
