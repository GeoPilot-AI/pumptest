"""Convert the Lauswiesen pumping tests from welltestpy's campaign file.

    python scripts/import_lauswiesen.py path/to/lauswiesen.cmp

The campaign file is ``data/lauswiesen.cmp`` in the archived workflow of
Mueller et al. (2022), https://doi.org/10.5281/zenodo.4724118. It is a zip of
zips holding small CSV files, so it is read here with the standard library
rather than with welltestpy. Writes one gzipped table per test into the
datasets folder (time in seconds, then drawdown in metres at B1 to B5) and
prints the well coordinates and rates that the loader needs.
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "pumptest" / "datasets" / "data"
WELLS = ("B1", "B2", "B3", "B4", "B5")
TESTS = ("B2", "B3", "B4", "B5")


def _variable(blob: bytes) -> list[float]:
    """The numbers in a welltestpy .var file: a scalar ``value`` or a ``values`` list."""

    rows = list(csv.reader(io.StringIO(blob.decode("utf-8"))))
    for i, row in enumerate(rows):
        if row and row[0] == "value":
            return [float(row[1])]
        if row and row[0] == "values":
            return [float(r[0]) for r in rows[i + 1 : i + 1 + int(row[1])]]
    raise ValueError("no value in variable file")


def _member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    (name,) = [n for n in archive.namelist() if n.endswith(suffix)]
    return archive.read(name)


def main() -> int:
    campaign = zipfile.ZipFile(sys.argv[1])
    for well in WELLS:
        inner = zipfile.ZipFile(io.BytesIO(_member(campaign, f"_{well}_Well.wel")))
        x, y = _variable(_member(inner, "_CooVar.var"))
        radius = _variable(_member(inner, "_RadVar.var"))[0]
        print(f"well {well}: x {x}, y {y}, radius {radius}")
    for test in TESTS:
        inner = zipfile.ZipFile(io.BytesIO(_member(campaign, f"_{test}_Test.tst")))
        rate = _variable(_member(inner, "_PprVar.var"))[0]
        columns, times = [], []
        for well in WELLS:
            obs = zipfile.ZipFile(io.BytesIO(_member(inner, f"_{well}_Obs.obs")))
            times.append(np.array(_variable(_member(obs, "_TimVar.var"))))
            columns.append(np.array(_variable(_member(obs, "_ObsVar.var"))))
        time = times[0]
        if any(not np.array_equal(time, t) for t in times):
            raise ValueError(f"test {test}: wells were logged on different clocks")
        table = np.column_stack([time, *columns])
        path = OUT / f"lauswiesen_{test}.txt.gz"
        header = f"Lauswiesen test {test}, rate {rate} m3/s\ntime_s " + " ".join(WELLS)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            np.savetxt(fh, table, fmt="%.6g", header=header)
        print(f"test {test}: rate {rate} m3/s, {len(time)} rows -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
