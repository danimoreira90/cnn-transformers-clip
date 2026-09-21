"""Unit tests for the rubric registry and evidence checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_mba.rubric import (
    RUBRIC,
    Evidence,
    Gap,
    RubricError,
    load_evidence,
    missing_evidence,
)

SECTION_SIZES = {"R1": 5, "R2": 5, "R3": 6, "R4": 4, "R5": 6}


def test_registry_holds_every_rubric_line() -> None:
    assert len(RUBRIC) == 26


def test_registry_section_sizes_match_the_marking_sheet() -> None:
    counts = {section: 0 for section in SECTION_SIZES}
    for rubric_id in RUBRIC:
        counts[rubric_id.split(".")[0]] += 1

    assert counts == SECTION_SIZES


def test_registry_text_is_never_empty() -> None:
    blank = [rubric_id for rubric_id, text in RUBRIC.items() if not text.strip()]

    assert blank == []


def test_load_evidence_skips_lines_not_yet_recorded(tmp_path: Path) -> None:
    path = tmp_path / "evidence.yaml"
    path.write_text("R1.1:\n  artifact: a.md\n  anchor: hello\nR1.2:\n")

    evidence = load_evidence(path)

    assert evidence == {"R1.1": Evidence(artifact="a.md", anchor="hello")}


def test_load_evidence_rejects_an_unknown_rubric_id(tmp_path: Path) -> None:
    path = tmp_path / "evidence.yaml"
    path.write_text("R9.9:\n  artifact: a.md\n  anchor: hello\n")

    with pytest.raises(RubricError, match="R9.9"):
        load_evidence(path)


def test_load_evidence_rejects_an_entry_without_an_anchor(tmp_path: Path) -> None:
    path = tmp_path / "evidence.yaml"
    path.write_text("R1.1:\n  artifact: a.md\n")

    with pytest.raises(RubricError, match="anchor"):
        load_evidence(path)


def test_missing_evidence_reports_lines_with_no_entry(tmp_path: Path) -> None:
    gaps = missing_evidence({}, tmp_path)

    assert len(gaps) == 26
    assert all(gap.reason == "no evidence recorded" for gap in gaps)


def test_missing_evidence_reports_an_artifact_that_does_not_exist(tmp_path: Path) -> None:
    evidence = {"R1.1": Evidence(artifact="gone.md", anchor="hello")}

    gaps = missing_evidence(evidence, tmp_path)
    r1_1 = [gap for gap in gaps if gap.rubric_id == "R1.1"]

    assert len(r1_1) == 1
    assert "gone.md" in r1_1[0].reason
    assert "not found" in r1_1[0].reason


def test_missing_evidence_reports_an_anchor_absent_from_the_artifact(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("this file says nothing relevant")
    evidence = {"R1.1": Evidence(artifact="present.md", anchor="crack detection")}

    gaps = missing_evidence(evidence, tmp_path)
    r1_1 = [gap for gap in gaps if gap.rubric_id == "R1.1"]

    assert len(r1_1) == 1
    assert "crack detection" in r1_1[0].reason


def test_missing_evidence_is_empty_when_every_line_is_backed(tmp_path: Path) -> None:
    (tmp_path / "all.md").write_text("\n".join(f"marker {rid}" for rid in RUBRIC))
    evidence = {
        rubric_id: Evidence(artifact="all.md", anchor=f"marker {rubric_id}")
        for rubric_id in RUBRIC
    }

    assert missing_evidence(evidence, tmp_path) == []


def test_gap_renders_as_one_readable_line() -> None:
    gap = Gap(rubric_id="R2.1", reason="no evidence recorded")

    assert str(gap) == "R2.1  no evidence recorded"
