"""The documentation's numbers and examples are executed, not trusted."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NUMBER_TOLERANCE = 5e-3
"""Fits land a few parts in ten thousand apart across platforms and scipy
versions (Dalem's aquitard resistance: 331.0 on Windows, 331.3 on Linux), so
numbers are compared to this relative tolerance and words exactly."""


def _same(a: str, b: str) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return math.isclose(x, y, rel_tol=NUMBER_TOLERANCE)


def test_validation_page_matches_a_fresh_run() -> None:
    fresh = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validation_report.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    committed = (REPO / "docs" / "validation.md").read_text(encoding="utf-8")
    new, old = re.split(r"[\s|]+", fresh.strip()), re.split(r"[\s|]+", committed.strip())
    hint = "run: python scripts/validation_report.py --write"
    assert len(new) == len(old), hint
    drift = [(a, b) for a, b in zip(old, new, strict=True) if not _same(a, b)]
    assert not drift, f"{hint}; differs at {drift[:5]}"


def test_number_comparison_is_tolerant_but_not_blind() -> None:
    assert _same("331", "331.3")
    assert not _same("331", "340")
    assert not _same("Theis", "Neuman")


def test_readme_examples_run() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert blocks, "the README should carry at least one runnable example"
    namespace: dict[str, object] = {}
    for block in blocks:
        exec(block, namespace)
