# 6. Step-drawdown coefficients by least squares in drawdown

Date: 2026-10-14

Status: accepted

## Context

The textbook analysis of a step-drawdown test (Hantush-Bierschenk) draws a
straight line through s/Q against Q. Fitting that line by least squares
weights each step by 1/Q, so the low-rate steps dominate. On Clark's "Well 1"
(Kruseman and de Ridder, Table 14.2) the s/Q regression missed the measured
drawdowns by 0.119 m RMSE, worse than the book's hand-drawn line at 0.105 m.

## Decision

`step_drawdown` fits s = B Q + C Q**2 by linear least squares on drawdown, the
quantity that was measured. On "Well 1" this misses by 0.096 m, better than
both.

## Consequences

- B and C differ slightly from a textbook s/Q regression on the same data;
  the docstring and the validation page say why.
- The validation test asserts the drawdown fit beats the published line,
  not that it reproduces it.
