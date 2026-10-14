# Choosing the model

Every solution will fit some curve through the data. Which one is right is a
hydrogeological question first and a statistical one second. pumptest gives
you two tools: the derivative plot, and AICc.

## The derivative tells the aquifer type

The drawdown derivative ds/d(ln t) is flat during radial flow in a confined
aquifer. It falls towards zero when leakage or a recharge boundary feeds the
well, and dips then recovers when delayed yield from a water table takes over.

```python
from pumptest import datasets, log_derivative

for name in ("oude_korendijk", "texas_hill", "vennebulten"):
    obs = datasets.load(name).observations[0]
    t, d = log_derivative(obs.time, obs.drawdown, window=0.3)
    print(f"{name:15} derivative early {d[1]:.3f}, middle {d[len(d) // 2]:.3f}, late {d[-2]:.3f}")
```

Texas Hill's derivative falls from about 0.57 to 0.11 (leakage feeding the
well). Vennebulten's is flat near 0.035, then sinks to about 0.006 and starts
to rise again at the end of the record: the dip of delayed yield. Oude
Korendijk's also drifts down, from about 0.16 to 0.10, which is one reason
its two piezometers do not give quite the same storativity. Real derivatives
are noisy; read the trend, not single points.

## AICc decides between candidates

```python
from pumptest import HantushJacob, Neuman, PapadopulosCooper, Theis, compare

for name in ("texas_hill", "vennebulten"):
    test = datasets.load(name)
    print(name)
    for r in compare(test, [Theis(), HantushJacob(), Neuman(), PapadopulosCooper()]):
        print(f"   {r.model:20} AICc {r.aicc:8.1f}   RMSE {r.rmse:.4f} m")
```

On Vennebulten, Neuman wins clearly over Hantush-Jacob and Theis. A model
that adds parameters has to lower the residual enough to pay for them; AICc
makes that trade explicit.

!!! warning "AICc ranks models, it does not validate them"
    The best model in the list can still be wrong. If none of the candidates
    follows the derivative, the aquifer has a feature none of them describes
    (a boundary, partial penetration, double porosity). Look at the residuals.

## Leaky aquifer: Texas Hill

```python
from pumptest import fit

result = fit(datasets.load("texas_hill"), HantushJacob())
print(result.summary())
```

The leakage factor B is fitted; the aquitard resistance c = B²/T and the
leakance K'/b' = 1/c are reported as derived values. AQTESOLV gives
c = 43.96 d on the same data; pumptest gives 43.9 d.

## Unconfined aquifer: Vennebulten

```python
result = fit(datasets.load("vennebulten"), Neuman())
print(result.summary())
```

Neuman's solution fits transmissivity, the elastic storativity S, specific
yield Sy and the anisotropy ratio Kd = Kz/Kr. It needs the saturated thickness
on the test. Specific yield from a test this short is poorly determined; the
textbook's own estimate of 4.9 × 10⁻³ is one it calls "unreasonably low".
