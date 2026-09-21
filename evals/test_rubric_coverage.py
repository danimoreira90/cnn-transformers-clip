"""Eval R3 — every rubric line is backed by a real artefact containing its anchor.

This is a regression eval, not a unit test. It is red from the first commit and only
turns green in Task 7.4, when all 26 rubric lines have earned evidence. It lives in
`evals/` rather than `tests/` so the unit suite stays honestly green while this one
stays honestly red.
"""

from __future__ import annotations

from pathlib import Path

from cv_mba.rubric import load_evidence, missing_evidence

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_every_rubric_line_has_evidence() -> None:
    evidence = load_evidence(REPO_ROOT / "evidence.yaml")

    gaps = missing_evidence(evidence, REPO_ROOT)

    assert gaps == [], "\n" + "\n".join(str(gap) for gap in gaps)
