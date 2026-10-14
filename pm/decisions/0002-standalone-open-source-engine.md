# 2. A standalone open-source engine, numpy and scipy only

Date: 2026-10-14

Status: accepted

## Context

PumpPilot, the pumping test product, needs an analysis engine. The commercial
tools (AQTESOLV, AquiferTest, MLU) are closed. The open ones are either a
MATLAB toolbox no longer maintained (Hytool), a Fortran program (USGS WTAQ),
or general transient-flow libraries (TTim) whose interface is built around
layered models rather than around a pumping test. The plan is to release the
engine on its own first and build the product on it later.

## Decision

`pumptest` is a separate repository and package under the MIT licence. Its
runtime dependencies are numpy and scipy and nothing else. It does no input or
output beyond what a caller hands it and never opens a network connection, so
the egress gate is not installed.

Both dependencies are available in Pyodide, which is the intended route for
running the same code in a browser later. Nothing runs it in a browser yet.

## Consequences

- Anyone can use, audit and cite the engine, which is the point of releasing it.
- The product cannot rely on anything here that is not in the open release.
- Reports, plotting and file formats belong to the product or to optional extras,
  not to the core package.
