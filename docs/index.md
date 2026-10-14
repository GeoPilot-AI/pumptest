# pumptest

**Pumping test analysis in Python.** Transmissivity, storativity, leakage,
specific yield, skin and well losses from drawdown data, by the classic
analytical solutions, with automatic fitting across any number of wells.

![Measured drawdown and fitted solutions for published tests](assets/fits.png)

## What it does

| | |
|---|---|
| **Solutions** | Theis, Hantush-Jacob (leaky), Papadopulos-Cooper (large-diameter wells, with optional skin and fitted casing radius), Neuman (unconfined, delayed yield) |
| **Straight-line methods** | Cooper-Jacob, Theis recovery, step-drawdown well losses |
| **Test types** | constant rate, recovery, and any variable-rate schedule, by superposition |
| **Fitting** | least squares over every observation at once, with standard errors, correlations, RMSE and AICc for choosing between models |
| **Diagnostics** | the Bourdet log-derivative |
| **Example data** | seven published tests bundled in `pumptest.datasets`, with the published estimates for each |
| **Dependencies** | numpy and scipy only |

## How far to trust it

On three tests from the AQTESOLV manual and four open pumping tests analysed
with welltestpy, the fitted parameters match the published ones, and on the
same data the fit is at least as good. The well functions are checked
against independent evaluations and published tables. Every number on
the [validation](validation.md) page is regenerated from the code, and a test
fails if the page and the code disagree.

![Published against pumptest parameters](assets/parity.png)

## Thirty seconds

```python
from pumptest import Theis, datasets, fit

test = datasets.load("sioux_flats")  # three observation wells, from the AQTESOLV manual
result = fit(test, Theis())
print(result.summary())
```

Next: [Getting started](getting-started.md).

## Licence

MIT. Copyright 2026 Piezoline Technologies Private Limited. The bundled
example data come from the TTim documentation (MIT) and from the Lauswiesen
research site (CC BY-SA 4.0); see [Citing](citing.md).
