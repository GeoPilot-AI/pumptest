#!/usr/bin/env python3
"""Every quality gate, in one command.

This is the single entry point for continuous integration. The gates live here
rather than in a pipeline definition so that the pipeline is thin and
replaceable: Section 7 of the terms of reference requires us to adopt IWMI's
DevOps rather than run a parallel toolchain, and when we move into their
pipeline it should have to call one command, not reimplement five.

Run the same thing everywhere:

    python scripts/checks.py            # all gates
    python scripts/checks.py --fast     # skip the slow ones
    python scripts/checks.py lint tests # named gates only

Exit code is zero only if every gate passed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A floor, not a target. Set just below where the suite actually sits, so it
# catches a module arriving untested rather than punishing an honest refactor.
# Raise it when the number moves up and stays there; never lower it to make a
# build pass.
COVERAGE_FLOOR = 90
PY = sys.executable


PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"

# Python says this when an entry point is invoked with -m and is not installed.
_NO_MODULE = re.compile(r"No module named (\S+)")


def _package() -> str:
    """The importable package under src/, for the coverage flag.

    This was hardcoded to the package name of the repository the gates were
    first written in. Every repository installed from the skill therefore
    measured coverage of a package it did not contain, which pytest-cov reports
    as zero percent rather than as an error, so the coverage floor passed while
    measuring nothing. The same failure as every other one recorded here: a
    control present, wired to the wrong thing, reporting success.
    """

    candidates = [d.name for d in (REPO / "src").glob("*") if (d / "__init__.py").exists()]
    if len(candidates) == 1:
        return candidates[0]
    # Two packages or none is a judgement, not a default. Measure the source
    # tree by path rather than guess a name.
    return "src"


def _tracked(*patterns: str) -> list[str]:
    """Files git knows about or would accept, so generated output is never linted.

    ``--others --exclude-standard`` is the load-bearing part. With plain
    ``ls-files`` a module that exists but has never been committed was not passed
    to the checker at all, and the gate reported PASS having examined nothing.
    That is the same failure this repository has now shipped twice: a control
    implemented, tested, and not wired to what it is meant to cover. The
    exclusion keeps gitignored output out, which was the original purpose.

    The exposure was developer-local, since CI runs on a pushed tree where
    everything is tracked. The local run is exactly what the gate exists to make
    trustworthy.
    """

    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *patterns],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted({line for line in out.stdout.splitlines() if line.strip()})


# name -> (description, argv, slow)
def _gates() -> dict[str, tuple[str, list[str], bool]]:
    return {
        "lint": (
            "ruff lint",
            [PY, "-m", "ruff", "check", "src", "tests", "scripts"],
            False,
        ),
        "format": (
            "ruff format, check only",
            [PY, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
            False,
        ),
        "style": (
            "house prose rules",
            [
                PY,
                "scripts/check_house_style.py",
                *_tracked("*.py", "*.md", "*.yaml", "*.yml", "*.toml"),
            ],
            False,
        ),
        # No egress gate: pumptest is pure computation and never opens a network
        # connection (decision 0002). Install the gate if that ever changes.
        "licences": (
            "no copyleft runtime dependency",
            [PY, "scripts/check_licences.py"],
            False,
        ),
        "types": (
            "mypy over src",
            [PY, "-m", "mypy"],
            False,
        ),
        "tests": (
            "pytest with a coverage floor, network tests excluded",
            [
                PY,
                "-m",
                "pytest",
                f"--cov={_package()}",
                "--cov-report=term-missing:skip-covered",
                "--cov-report=json",
            ],
            False,
        ),
        "docs": (
            "documentation site builds, strict on broken links",
            [PY, "-m", "mkdocs", "build", "--quiet", "--strict"],
            False,
        ),
        "coverage": (
            "per-module coverage floor, not only the total",
            [PY, "scripts/check_coverage.py"],
            False,
        ),
        "decisions": (
            "architecture decision records are numbered, complete and indexed",
            [PY, "scripts/check_decisions.py"],
            False,
        ),
        "network": (
            "pytest, live external services",
            [PY, "-m", "pytest", "-m", "network"],
            True,
        ),
    }


def run(name: str, description: str, argv: list[str]) -> tuple[str, float]:
    """Run one gate and report PASS, FAIL or MISSING.

    MISSING is its own outcome, and it is not a pass. A gate whose tool is not
    installed did not check anything, and reporting that as a failure alongside
    a real one wastes the reader's time working out which they have. The docs
    gate spent four merges red in CI for exactly this reason, because mkdocs was
    on the developer machine and not in the workflow's extras.
    """

    started = time.time()
    print(f"\n--- {name}: {description}")

    # A gate whose own script is absent is not a tool that failed to install, it
    # is a gate this repository never opted into. Say so in words a reader can
    # act on, and still refuse to pass: a check that quietly reports nothing is
    # the exact failure these gates exist to prevent.
    script = next((a for a in argv if a.startswith("scripts/")), None)
    if script is not None and not (REPO / script).exists():
        print(f"{script} is not installed in this repository.")
        print(
            f"Either install it, or delete the '{name}' entry from _gates() in "
            "this file. Do not leave a gate listed that checks nothing."
        )
        return MISSING, time.time() - started

    try:
        result = subprocess.run(argv, cwd=REPO, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"tool not found: {argv[0]}")
        return MISSING, time.time() - started

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    combined = result.stdout + result.stderr
    if result.returncode != 0 and _NO_MODULE.search(combined):
        return MISSING, time.time() - started
    return (PASS if result.returncode == 0 else FAIL), time.time() - started


def main() -> int:
    gates = _gates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=[*gates, []], help="gates to run")
    parser.add_argument("--fast", action="store_true", help="skip slow gates")
    args = parser.parse_args()

    selected = args.names or [
        n for n, (_, _, slow) in gates.items() if not (slow and args.fast) and not slow
    ]

    results = []
    for name in selected:
        description, argv, _ = gates[name]
        outcome, took = run(name, description, argv)
        results.append((name, outcome, took))

    print("\n" + "=" * 60)
    for name, outcome, took in results:
        print(f"  {outcome:<7} {name:<10} {took:5.1f}s")
    print("=" * 60)

    failed = [n for n, o, _ in results if o == FAIL]
    missing = [n for n, o, _ in results if o == MISSING]

    if missing:
        print(
            f"COULD NOT RUN: {', '.join(missing)}. These gates checked nothing, "
            "which is never a pass. Each one printed above which of the two "
            "cases it is: a tool that is not installed, or a gate script this "
            "repository does not have.\n"
            '  missing tool:   pip install -e ".[dev,docs]"\n'
            "  missing script: install it, or remove the gate from _gates()"
        )
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    if failed or missing:
        return 1
    print(f"All {len(results)} gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
