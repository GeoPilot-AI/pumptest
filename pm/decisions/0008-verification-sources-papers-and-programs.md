# 8. Verify against papers and published program results, not textbook examples

Date: 2026-10-14

Status: accepted

Supersedes the textbook part of record 5.

## Context

Record 5 compared the package with AQTESOLV results and with the worked
examples and type-curve annexes of Kruseman and de Ridder (1994). A review of
those comparisons found one parameter set attributed to the textbook (Dalem:
T 1677 m2/d, c 331 d) that the book does not contain; the book's own three
analyses give 1665 to 1883 m2/d and 216 to 578 d. The textbook answers are
also graphical, so they bound a fit only loosely, and transcribed tables
raise a permission question for redistribution.

## Decision

- Field comparisons use only data and parameters published in papers or by
  the programs that produced them: the AQTESOLV results reported by Yang
  (2020) and the welltestpy results of Mueller et al. (2022) on the open
  Lauswiesen data.
- The Oude Korendijk, Dalem, Vennebulten, Hoogezand and step-drawdown "Well 1"
  records are removed from the package, the tests and the documentation.
- Tutorials that need a test type no bundled record covers use synthetic
  records made with the model itself, and say so.
- Well functions are checked against independent evaluations of the formulas
  in the original papers; the textbook annex tables are replaced in a
  following change.

## Consequences

- The Neuman and Papadopulos-Cooper models have no published field test until
  one from a paper is found; synthetic recovery tests cover their fitting.
- The validation page and figures show seven tests: three AQTESOLV, four
  Lauswiesen.
