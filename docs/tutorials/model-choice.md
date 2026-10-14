# Choosing the model

Every solution will fit some curve through the data. Which one is right is a
hydrogeological question first and a statistical one second. pumptest gives
you two tools: the derivative plot, and AICc.

This tutorial uses the leaky Texas Hill test from the AQTESOLV manual and,
because no unconfined test with published parameters ships with the package
yet, a synthetic unconfined test made with the Neuman solution itself.

## A synthetic unconfined test

```python
import numpy as np
from pumptest import Neuman, Observation, PumpingTest, Schedule, Well, datasets

truth = {"T": 1300.0, "S": 5e-4, "Sy": 0.08, "Kd": 0.1}
t = np.logspace(-4, 1, 40)
template = PumpingTest(Well(0.15), Schedule.constant(900.0),
                       [Observation("OW 20 m", 20.0, t, np.ones_like(t))], thickness=20.0)
s = Neuman().drawdown(truth, template, template.observations[0])
s = s + np.random.default_rng(1).normal(0.0, 0.002, s.size)  # 2 mm of noise
unconfined = PumpingTest(template.well, template.schedule,
                         [Observation("OW 20 m", 20.0, t, s)], thickness=20.0)
```

## The derivative tells the aquifer type

The drawdown derivative ds/d(ln t) is flat during radial flow in a confined
aquifer. It falls towards zero when leakage or a recharge boundary feeds the
well, and dips then recovers when delayed yield from a water table takes over.

```python
from pumptest import log_derivative

for name, test in (("Texas Hill", datasets.load("texas_hill")), ("unconfined", unconfined)):
    obs = test.observations[0]
    _, d = log_derivative(obs.time, obs.drawdown, window=0.3)
    print(f"{name:12} derivative early {d[1]:.3f}, middle {d[len(d) // 2]:.3f}, "
          f"late {d[-2]:.3f}, lowest {d.min():.4f}")
```

Texas Hill's derivative falls from about 0.59 to 0.14: leakage feeding the
well. The synthetic test's derivative sinks from about 0.03 to near zero and
rises again to about 0.05: the dip of delayed yield. Real derivatives are
noisier; read the trend, not single points.

## AICc ranks the candidates

```python
from pumptest import HantushJacob, PapadopulosCooper, Theis, compare

for name, test in (("Texas Hill", datasets.load("texas_hill")), ("unconfined", unconfined)):
    print(name)
    for r in compare(test, [Theis(), HantushJacob(), Neuman(), PapadopulosCooper()]):
        print(f"   {r.model:20} AICc {r.aicc:8.1f}   RMSE {r.rmse:.4f} m")
```

Hantush-Jacob ranks first on Texas Hill and Neuman on the synthetic test. A
model that adds parameters has to lower the residual enough to pay for them;
AICc makes that trade explicit.

!!! warning "AICc ranks models, it does not validate them"
    The best model in the list can still be wrong. If no candidate follows
    the derivative, the aquifer may have a feature that the candidates do not
    describe, such as a boundary, partial penetration or double porosity.
    Look at the residuals. AICc also assumes independent residuals, which
    drawdown records rarely have, so small differences in AICc mean little.

## Leaky aquifer: Texas Hill

```python
from pumptest import fit

result = fit(datasets.load("texas_hill"), HantushJacob())
print(result.summary())
```

The leakage factor B is fitted; the aquitard resistance c = B²/T and the
leakance K'/b' = 1/c are reported as derived values. AQTESOLV gives
c = 43.96 d on the same data; pumptest gives 43.89 d.

## Unconfined aquifer

```python
result = fit(unconfined, Neuman())
print(result.summary())
```

Neuman's solution fits transmissivity, the elastic storativity S, specific
yield Sy and the anisotropy ratio Kd = Kz/Kr, and needs the saturated
thickness. On the synthetic record it recovers the values it was made from
(T = 1295 against 1300 m²/d, Sy = 0.081 against 0.08, Kd = 0.101 against
0.1). Field records are shorter and noisier, and specific yield is usually
the least certain of the four.
