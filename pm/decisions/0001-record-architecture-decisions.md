# 1. Record architecture decisions

Date: 2026-10-14

Status: accepted

## Context

A new person, or the same team months later, needs to know not just what the
architecture is but why it is that way. Decisions made in chat, in a proposal or
in a pull request comment are lost to the repository.

## Decision

Architecture decision records live in `pm/decisions/`, one short file per
decision, numbered in sequence. Each states the context, the decision and its
consequences. A record is written when a decision shapes the architecture or a
delivery boundary.

The index at `pm/decisions/README.md` is generated, not written:

    python scripts/check_decisions.py --write

## Consequences

- The reasoning behind the codebase is discoverable in the repository itself.
- Records are immutable once accepted. A decision that changes course gets its
  own record and supersedes the earlier one by reference, rather than the
  earlier one being edited to agree with the present.
- The format stays short, so writing one is never a barrier.
- The `decisions` gate fails the build when the numbering, the required
  sections or the index drift.
