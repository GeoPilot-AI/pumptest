"""The documentation's numbers and examples are executed, not trusted."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_validation_page_matches_a_fresh_run() -> None:
    fresh = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validation_report.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    committed = (REPO / "docs" / "validation.md").read_text(encoding="utf-8")
    assert fresh.strip() == committed.strip(), "run: python scripts/validation_report.py --write"


def test_readme_examples_run() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert blocks, "the README should carry at least one runnable example"
    namespace: dict[str, object] = {}
    for block in blocks:
        exec(block, namespace)
