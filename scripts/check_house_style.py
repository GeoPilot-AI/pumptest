#!/usr/bin/env python3
"""Enforce the house prose rules on files about to be committed.

One rule, for one good reason. Em-dashes and en-dashes mojibake through the
Word table-of-contents step in the document toolchain, where U+2014 has come
back as a EUR sequence in a client deliverable. They also read as
machine-written. Use a colon, a comma or parentheses.

The check is here rather than in a style guide because a style guide is a
request and a hook is a rule, and this one has already cost a rebuild once.

Two notes on the implementation, both learned the hard way when this file was
first run across the whole repository.

The banned characters are written as escape sequences rather than literally, so
this file does not contain the characters it forbids and therefore does not
report itself. An exemption for its own path would have worked and would have
been worse: the rule would then have had a hole in it shaped exactly like the
one file most likely to be copied as an example.

The offending character is never echoed. Printing it to a console using a
legacy code page raises UnicodeEncodeError, so the first genuine violation
would have crashed the tool instead of reporting it. Output is plain ASCII, and
the character is shown as a named marker in place.

Usage:
    python scripts/check_house_style.py FILE [FILE ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

BANNED = {
    "\u2014": "em-dash",
    "\u2013": "en-dash",
    "\u2012": "figure dash",
    "\u2015": "horizontal bar",
}

# Paths where the character is data rather than prose, or is not ours to change.
EXEMPT_PARTS = {".git", "node_modules", ".venv", "out"}

_MAX_EXCERPT = 90


def _safe(text: str) -> str:
    """An ASCII-only rendering, with each banned character named in place.

    Keeps the report readable on any console and keeps the tool from failing on
    the one input it exists to handle.
    """

    for char, name in BANNED.items():
        text = text.replace(char, f"[{name}]")
    return text.encode("ascii", "replace").decode("ascii")


def check(path: Path) -> list[str]:
    if any(part in EXEMPT_PARTS for part in path.parts):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, PermissionError):
        return []

    problems = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for char, name in BANNED.items():
            col = line.find(char)
            if col < 0:
                continue
            excerpt = line.strip()
            if len(excerpt) > _MAX_EXCERPT:
                excerpt = excerpt[: _MAX_EXCERPT - 3] + "..."
            problems.append(f"{path}:{lineno}:{col + 1}  {name}  |  {_safe(excerpt)}")
    return problems


def main(argv: list[str]) -> int:
    problems: list[str] = []
    for arg in argv:
        problems.extend(check(Path(arg)))

    if not problems:
        return 0

    print("House style: dashes that break the Word export and read as machine-written.")
    for p in problems:
        print("  " + p)
    print("\nReplace with a colon, a comma or parentheses.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
