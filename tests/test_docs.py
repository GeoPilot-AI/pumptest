"""The documentation's numbers and examples are executed, not trusted."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

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


PAGES = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(REPO)))
def test_documentation_examples_run(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each page is one script, run top to bottom in a scratch directory, the
    # way a reader would paste it.
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), flags=re.S)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for block in blocks:
        exec(block, namespace)


def test_readme_and_every_tutorial_have_examples() -> None:
    for page in [REPO / "README.md", *(REPO / "docs" / "tutorials").glob("*.md")]:
        assert "```python" in page.read_text(encoding="utf-8"), page.name
