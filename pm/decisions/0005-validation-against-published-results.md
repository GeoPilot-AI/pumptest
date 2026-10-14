# 5. Validation against published fits and published tables

Date: 2026-10-14

Status: accepted

## Context

An engine that claims to replace AQTESOLV has to show it gets the same answers.
The TTim documentation (MIT) publishes real tests with the parameters AQTESOLV,
MLU and Kruseman and de Ridder obtained. Kruseman and de Ridder (1994) tabulate
the well functions themselves.

## Decision

The suite carries two kinds of answer key, both as committed tests:

1. Five published tests (Oude Korendijk, Gridley, Sioux Flats, Texas Hill,
   Dalem). Fitted parameters must be close to the published ones, and our fit
   must be at least as good as the published parameters evaluated on the same
   data. Printed RMSE values are not compared digit for digit, because they were
   computed on each program's own copy of the data.
2. Values transcribed from the Kruseman and de Ridder annexes for Theis,
   Hantush, Neuman and Papadopulos-Cooper, within 0.6 percent (the tables carry
   three significant figures).

The unconfined and wellbore-storage fits are tested end to end on synthetic
data, because the published comparisons for those cases use partial
penetration, which is not modelled yet.

## Consequences

- "Matches AQTESOLV" is a test result, not a claim.
- One table cell (Annex 4.2, u = 2e-3, r/L = 0.08) is a misprint in the book and
  is recorded as such in the test.
