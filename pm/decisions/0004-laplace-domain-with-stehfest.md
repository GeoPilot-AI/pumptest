# 4. Solutions in the Laplace domain, inverted by Stehfest

Date: 2026-10-14

Status: accepted

## Context

Only Theis has a convenient closed form in time. Leakage, wellbore storage,
skin and the delayed water-table response all have simple closed forms in the
Laplace domain. WTAQ and AQTESOLV invert those numerically.

## Decision

Theis is evaluated directly from the exponential integral. Every other model
is a Laplace-domain kernel inverted by the Stehfest algorithm with 16 terms,
which holds Theis to 4e-5 relative wherever u <= 2. Confined, leaky,
wellbore-storage and skin cases share one radial kernel. The Neuman kernel sums
its vertical eigenfunction series to at most 400 terms.

The fitting Jacobian uses a relative finite-difference step of 1e-4, because
Stehfest round-off near 1e-7 relative swamps scipy's default step and stops
the optimiser on its first iteration (found while validating Texas Hill).

## Consequences

- A new solution is a new kernel, tested against its published table.
- Stehfest is unsuited to oscillating responses; none of these solutions oscillate.
- Relative accuracy is poor where drawdown is negligibly small (u above about
  5), which does not affect fits because those drawdowns are tiny in absolute terms.
