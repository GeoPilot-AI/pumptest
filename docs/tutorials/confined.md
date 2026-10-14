# Confined aquifer, several wells

The Oude Korendijk test (Kruseman and de Ridder, 1994) pumped a 7 m confined
aquifer of coarse sand at 788 m³/d for about 14 hours, with piezometers 30 m
and 90 m from the well. It ships with the package.

## Load and look

```python
from pumptest import datasets

test = datasets.load("oude_korendijk")
for obs in test.observations:
    print(obs.name, obs.distance, "m:", obs.time.size, "readings, last", obs.drawdown[-1], "m")
```

## A quick Cooper-Jacob check

The straight-line method gives a first answer and tells you whether the late
data are in the range where it is valid (u ≤ 0.05):

```python
from pumptest import cooper_jacob

h30 = test.observations[0]
cj = cooper_jacob(h30.time, h30.drawdown, rate=788.0, distance=30.0)
print(f"T = {cj.transmissivity:.0f} m2/d, S = {cj.storativity:.2e}, u_max = {cj.u_max:.3f}")
```

## Fit Theis to both piezometers at once

Fitting every observation together gives one set of aquifer parameters that
must explain all of them, which is what AQTESOLV calls a composite fit:

```python
from pumptest import Theis, fit

result = fit(test, Theis())
print(result.summary())
```

The result matches AQTESOLV: T = 462.6 m²/d, S = 1.78 × 10⁻⁴, RMSE 0.050 m
(see [Validation](../validation.md)).

## Fit each piezometer on its own

```python
for name in ("H30", "H90"):
    single = fit(test, Theis(), observations=[name])
    print(name, round(single.values["T"]), f"{single.values['S']:.2e}")
```

Fitted alone, the two piezometers agree on T to within about 5 percent
(480 and 501 m²/d) but differ by a factor of about two in S. That is
typical: storativity depends on the early data and on the exact distance,
and is usually the less certain of the two. The composite fit is the
defensible single answer; the spread between the single-well fits is a fair
statement of its uncertainty.

## Plot it

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
for obs in test.observations:
    line = np.logspace(np.log10(obs.time.min()), np.log10(obs.time.max()), 200)
    ax.loglog(obs.time, obs.drawdown, "o", mfc="none", label=f"{obs.name} measured")
    ax.loglog(line, Theis().drawdown(result.values, test, obs, line), label=f"{obs.name} Theis")
ax.set_xlabel("time since pumping started (d)")
ax.set_ylabel("drawdown (m)")
ax.legend()
fig.savefig("oude_korendijk.png", dpi=150)
```

`Model.drawdown(values, test, observation, t)` evaluates any model at any
times, with the whole pumping schedule superposed.
