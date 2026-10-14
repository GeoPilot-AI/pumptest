#!/usr/bin/env python3
"""Enforce coverage per module, not only in total.

A total floor is the wrong instrument on its own, and this repository proved it.
`extraction/pdf.py` sat at zero percent for days while the suite reported ninety
percent overall, because eight hundred covered statements elsewhere absorbed it.
That module carries the whole national-source extraction path and had just been
swapped to a different PDF backend for licensing reasons, so it was the single
worst file to be untested and the total said nothing.

The failure mode is arithmetic rather than carelessness: on a codebase of this
size, ninety percent tolerates roughly ninety uncovered statements, which is
several entire modules. A new file arrives, nobody tests it, the total dips by a
point, and the gate stays green.

So two floors. The total catches gradual rot. The per-module floor catches a
file arriving with nothing behind it, which is the case that actually happened.

Reads `coverage.json`, which the test gate writes.

Usage:
    python scripts/check_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "coverage.json"

TOTAL_FLOOR = 90.0

# Deliberately well below the total. This is not a quality bar, it is a tripwire
# for a module nobody wrote a test for. A file genuinely hard to cover gets an
# entry in EXEMPT with a reason, which makes the exception visible in review
# rather than hidden in a falling average.
MODULE_FLOOR = 60.0

EXEMPT: dict[str, str] = {
    # One line per file, with the reason and what closes it. An entry here is a
    # debt with a name, not a permanent exception. Example:
    # "src/pkg/legacy.py": "superseded by the new reader, removed at step 5",
}


def main() -> int:
    if not REPORT.exists():
        print(f"no {REPORT.name}. Run the test gate first, which writes it.")
        return 1

    data = json.loads(REPORT.read_text(encoding="utf-8"))
    total = data["totals"]["percent_covered"]

    failures: list[str] = []
    exempted: list[str] = []

    for path, entry in sorted(data["files"].items()):
        rel = Path(path).as_posix()
        percent = entry["summary"]["percent_covered"]
        statements = entry["summary"]["num_statements"]
        if statements == 0:
            continue
        if rel in EXEMPT:
            exempted.append(f"{rel} at {percent:.0f}%: {EXEMPT[rel]}")
            continue
        if percent < MODULE_FLOOR:
            failures.append(
                f"{rel}: {percent:.0f}% of {statements} statements, floor is {MODULE_FLOOR:.0f}%"
            )

    print(f"Coverage: {total:.1f}% total, floor {TOTAL_FLOOR:.0f}%")
    for line in exempted:
        print(f"  exempt  {line}")

    if total < TOTAL_FLOOR:
        failures.append(f"total {total:.1f}% is below the floor of {TOTAL_FLOOR:.0f}%")

    if failures:
        print("\nCOVERAGE FLOOR NOT MET")
        for line in failures:
            print("  " + line)
        print(
            "\nA module below the floor is usually a module nobody wrote a test for. "
            "If it is genuinely hard to cover, add it to EXEMPT with a reason, so the "
            "exception is visible in review rather than absorbed by the average."
        )
        return 1

    print(f"  every module at or above {MODULE_FLOOR:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
