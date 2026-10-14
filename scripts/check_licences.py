#!/usr/bin/env python3
"""Fail if a runtime dependency carries a copyleft licence.

Why this exists, specifically. Section 6 of the terms of reference assigns the
deliverables to IWMI and leaves the licensing decision to IWMI alone. A copyleft
runtime dependency constrains that decision, because it can attach conditions to
the work it is distributed with. Our own background intellectual property
disclosure also states affirmatively that the package contains no AGPL, GPL or
LGPL component, and that sentence has to be true on the day it is signed.

It exists because the check was needed once. PyMuPDF (AGPL-3.0 or commercial)
was a declared runtime dependency and nobody noticed until a licence review
looked. The swap to pypdfium2 was a ten-minute change; finding it the week of
signature would not have been. So the check is automated rather than left to
whoever remembers.

Development and project-management extras are not checked. They are not
distributed with the deliverable, so their licences bind our use and do not
attach to IWMI's.

Usage:
    python scripts/check_licences.py            # check, print a table
    python scripts/check_licences.py --sbom     # emit the bill of materials
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Substrings that mark a licence as copyleft for our purposes. Matched
# case-insensitively against both the License field and the classifiers.
DENY = ("agpl", "affero", "gpl", "lgpl", "sspl", "osl", "eupl", "cecill")

# "GPL" appears inside these permissive names, so match them first and allow.
ALLOW_CONTAINING_GPL = ("lgpl-compatible",)


def _runtime_requirements() -> list[str]:
    """The distribution names declared as runtime dependencies."""

    with (REPO / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    names = []
    for spec in data.get("project", {}).get("dependencies", []):
        name = spec.split(";")[0]
        for sep in (">=", "<=", "==", "~=", "!=", ">", "<", "["):
            name = name.split(sep)[0]
        names.append(name.strip())
    return names


def _closure(roots: list[str]) -> set[str]:
    """Every distribution reachable from the runtime requirements."""

    seen: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen.add(key)
        for req in dist.requires or []:
            # Skip optional extras: they are not installed for the runtime path.
            if "extra ==" in req:
                continue
            dep = req.split(";")[0].strip()
            for sep in (">=", "<=", "==", "~=", "!=", ">", "<", "[", " "):
                dep = dep.split(sep)[0]
            if dep:
                queue.append(dep.strip())
    return seen


def _licence_of(name: str) -> str:
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    return licence_from(meta)


def licence_from(meta: Any) -> str:
    """Resolve a distribution's licence from its metadata, best field first.

    Three fields can carry it, and they arrived in this order historically:

        License-Expression   PEP 639, an SPDX expression. Authoritative.
        License ::           classifiers, a fixed vocabulary. Reliable.
        License              free text, sometimes the entire licence body.

    License-Expression is read first because PEP 639 deprecates the other two,
    and packaging tools have begun rejecting a distribution that sets both an
    expression and a classifier. So a package that migrates drops its
    classifiers, and a gate reading only classifiers goes quietly blind on it.

    That is not cosmetic. This gate exists to refuse copyleft in a deliverable
    the client owns, and an unread licence resolves to "unstated", which is not
    copyleft, which passes. An AGPL dependency declaring itself the modern way
    would have gone straight through. Found when the first PEP 639 dependency
    arrived in a closure and the gate printed it as unstated despite it
    declaring BSD-3-Clause.
    """

    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression

    stated = (meta.get("License") or "").strip()
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        derived = "; ".join(c.split("::")[-1].strip() for c in classifiers)
        # A classifier is more reliable than a free-text License field, which is
        # sometimes the entire licence text.
        return derived if not stated or len(stated) > 60 else f"{derived} ({stated})"
    return stated or "unstated"


def _is_copyleft(licence: str) -> bool:
    low = licence.lower()
    if any(ok in low for ok in ALLOW_CONTAINING_GPL):
        return False
    return any(bad in low for bad in DENY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbom", action="store_true", help="emit the bill of materials as JSON")
    args = parser.parse_args()

    roots = _runtime_requirements()
    dists = sorted(_closure(roots))
    rows = [(d, metadata.version(d) if _installed(d) else "", _licence_of(d)) for d in dists]

    if args.sbom:
        print(
            json.dumps([{"name": n, "version": v, "licence": lic} for n, v, lic in rows], indent=2)
        )
        return 0

    offenders = [(n, v, lic) for n, v, lic in rows if _is_copyleft(lic)]
    unstated = [(n, v, lic) for n, v, lic in rows if lic in ("unstated", "not installed")]

    width = max((len(n) for n, _, _ in rows), default=10)
    print(f"Runtime dependency closure: {len(rows)} distributions")
    for n, v, lic in rows:
        mark = "  FAIL" if _is_copyleft(lic) else ("  ?   " if lic == "unstated" else "  ok  ")
        print(f"{mark} {n:<{width}}  {v:<10}  {lic}")

    if unstated:
        print(
            f"\n{len(unstated)} distribution(s) state no licence. "
            "Record these by hand in the background IP disclosure."
        )

    if offenders:
        print("\nCOPYLEFT RUNTIME DEPENDENCY FOUND")
        for n, v, lic in offenders:
            print(f"  {n} {v}: {lic}")
        print(
            "\nThis constrains IWMI's licensing decision, which Section 6 reserves to "
            "IWMI alone, and contradicts the background IP disclosure. Replace it, or "
            "obtain a commercial licence and record it before the disclosure is signed."
        )
        return 1

    print("\nNo copyleft runtime dependency. Safe against the Section 6 position.")
    return 0


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main())
