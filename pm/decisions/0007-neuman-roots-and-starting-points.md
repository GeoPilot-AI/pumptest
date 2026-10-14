# 7. Neuman roots by offset; delayed-yield starts from early and late Theis fits

Date: 2026-10-14

Status: accepted

## Context

Fitting Neuman to Texas Hill took about three minutes, 95 percent of it in
the eigenvalue solver, which iterated on the root itself from a rough guess
and fell back to bisection. Replacing it with a fast solver exposed a second
fault: every Neuman starting point began with S near 1e-7, and on Vennebulten
all of them converged to a Theis-like answer (RMSE 0.0101 m instead of 0.0030 m).
The slow solver had only been escaping that by accident.

## Decision

- Solve each root as n pi + d, Newton on the offset d from two fixed-point
  steps, which converges in a few iterations at full precision.
- Start the delayed-yield fit from Theis fits to the first 40 and last 40
  percent of each record: T and the elastic S from early time, Sy from late
  time, the hand method of matching type-A and type-B curves.
- With more than three candidate starts, optimise only from the three with the
  lowest initial residual.

## Consequences

- Vennebulten fits in under 2 s and reaches the Neuman minimum.
- A Neuman fit pushed onto leaky data (Kd near its lower bound) still needs
  the full 400-term series and takes about 16 s. Accelerating the series tail
  is the next step for speed; nothing does it yet.
