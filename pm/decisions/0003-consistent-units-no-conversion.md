# 3. Consistent units, never converted inside the solvers

Date: 2026-10-14

Status: accepted

## Context

Field data arrives in mixed units: rates in litres per minute or gallons per
minute, time in minutes, levels in feet or metres below ground. A solver that
converts silently has to guess which unit each number is in, and a wrong guess
is a result that is wrong by a constant factor and still looks plausible.

## Decision

Every solver takes and returns numbers in one consistent set chosen by the
caller (metres, days and cubic metres per day is the usual one). Conversion is
an explicit, separate step through `pumptest.units`.

## Consequences

- Transmissivity comes back in the caller's length squared per time unit.
- The examples and tests state their units at the point of use.
- A unit mistake shows up at the conversion call, where the unit names are
  written down, rather than inside a fit.
