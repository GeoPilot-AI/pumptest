# Several observation wells

The Sioux Flats test from the AQTESOLV manual (Duffield, 2007) pumped
6606 m³/d for about 1.4 days, with observation wells 30.5 m, 61 m and 122 m
from the pumped well. It ships with the package.

## Load and look

```python
from pumptest import datasets

test = datasets.load("sioux_flats")
for obs in test.observations:
    print(obs.name, round(obs.distance, 1), "m:", obs.time.size, "readings, last", obs.drawdown[-1], "m")
```

## A quick Cooper-Jacob check

The straight-line method gives a first answer and tells you whether the late
data are in the range where it is valid (u ≤ 0.05):

```python
from pumptest import cooper_jacob

ow100 = test.observations[0]
cj = cooper_jacob(ow100.time, ow100.drawdown, rate=6605.754, distance=ow100.distance)
print(f"T = {cj.transmissivity:.0f} m2/d, S = {cj.storativity:.2e}, u_max = {cj.u_max:.3f}")
```

## Fit Theis to all three wells at once

Fitting every observation together gives one set of aquifer parameters that
must explain all of them:

```python
from pumptest import Theis, fit

result = fit(test, Theis())
print(result.summary())
```

The result matches the AQTESOLV analysis: T = 4310 m²/d against 4308, and
S = 0.0641 against 0.0642 (see [Validation](../validation.md)).

## Fit each well on its own

```python
for name in ("OW100", "OW200", "OW400"):
    single = fit(test, Theis(), observations=[name])
    print(name, round(single.values["T"]), f"{single.values['S']:.3e}")
```

Fitted alone, the three wells give transmissivities between 4278 and
4301 m²/d and storativities between 0.062 and 0.067. When single-well fits
agree this closely, one Theis solution describes the whole test; when they
do not, the composite fit is a compromise and the residuals show where it
fails.

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
fig.savefig("sioux_flats.png", dpi=150)
```

`Model.drawdown(values, test, observation, t)` evaluates any model at any
times, with the whole pumping schedule superposed.
