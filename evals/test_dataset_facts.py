"""Eval: the real archives on disk still hold the numbers SPEC.md was written against.

Every figure here was measured on 2026-09-20 and then used to make design decisions.
If an upstream archive changes, a decision silently stops being justified. This eval is
what notices.

Skipped with an explicit reason when the archive is absent, naming the command that
fetches it. A skip here means the data is not present; it never means a failure was
stepped over.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cv_mba.data import DuplicateCorpusWarning, a3_index, elpv_index, stratified_split

DATA_ROOT = Path(os.environ.get("CV_MBA_DATA", Path(__file__).resolve().parent.parent / "data" / "raw"))
ELPV_ROOT = DATA_ROOT / "elpv"

FETCH = "git clone https://github.com/zae-bayern/elpv-dataset.git data/raw/elpv"


@pytest.fixture(scope="module")
def elpv():
    if not (ELPV_ROOT / "src" / "elpv_dataset" / "data" / "labels.csv").is_file():
        pytest.skip(f"ELPV archive not at {ELPV_ROOT}. Fetch it with:  {FETCH}")
    return elpv_index(ELPV_ROOT)


def test_elpv_still_holds_2624_cells(elpv) -> None:
    assert len(elpv) == 2624


def test_elpv_annotator_agreement_buckets_are_unchanged(elpv) -> None:
    counts = elpv["defect_probability"].round(3).value_counts().to_dict()

    assert counts == {0.0: 1508, 0.333: 295, 0.667: 106, 1.0: 715}


def test_elpv_binary_split_at_the_threshold_is_1803_to_821(elpv) -> None:
    counts = elpv["label"].value_counts().to_dict()

    assert counts == {0: 1803, 1: 821}


def test_elpv_wafer_types_are_unchanged(elpv) -> None:
    counts = elpv["wafer_type"].value_counts().to_dict()

    assert counts == {"poly": 1550, "mono": 1074}


def test_defect_rate_still_differs_by_wafer_type(elpv) -> None:
    # This difference is the reason splits are stratified on label and wafer type
    # together rather than on label alone. If it ever disappears, the joint
    # stratification stops being justified and SPEC.md needs revisiting.
    rate = elpv.groupby("wafer_type")["label"].mean()

    assert rate["mono"] == pytest.approx(369 / 1074, abs=0.001)
    assert rate["poly"] == pytest.approx(452 / 1550, abs=0.001)
    assert rate["mono"] > rate["poly"]


def test_a_real_split_preserves_every_stratum_and_shares_no_image(elpv) -> None:
    train, test = stratified_split(elpv, by=["label", "wafer_type"], test_size=0.2, seed=0)

    assert len(train) + len(test) == 2624
    assert set(train["path"]).isdisjoint(set(test["path"]))
    for keys, group in elpv.groupby(["label", "wafer_type"]):
        chosen = test[(test["label"] == keys[0]) & (test["wafer_type"] == keys[1])]
        assert len(chosen) / len(group) == pytest.approx(0.2, abs=0.01)


A3_ROOT = DATA_ROOT / "a3" / "data"

A3_FETCH = "uv run kaggle datasets download -d pavansanagapati/images-dataset -p data/raw/a3 --unzip"

A3_EXPECTED = {
    "bike": 365,
    "cars": 420,
    "cats": 202,
    "dogs": 202,
    "flowers": 210,
    "horses": 202,
    "human": 202,
}


@pytest.fixture(scope="module")
def a3():
    if not (A3_ROOT / "cats").is_dir():
        pytest.skip(f"A3 archive not at {A3_ROOT}. Fetch it with:  {A3_FETCH}")
    with pytest.warns(DuplicateCorpusWarning):
        return a3_index(A3_ROOT)


def test_a3_holds_1803_unique_images_not_3606(a3) -> None:
    assert len(a3) == 1803
    assert len(a3) == sum(A3_EXPECTED.values())


def test_a3_class_counts_are_unchanged(a3) -> None:
    counts = a3["class_name"].value_counts().to_dict()

    assert counts == A3_EXPECTED


def test_a3_has_seven_classes_and_no_phantom_eighth(a3) -> None:
    # A recursive load invents a class named "data" from the archive's copy of itself.
    assert sorted(a3["class_name"].unique()) == sorted(A3_EXPECTED)
    assert "data" not in set(a3["class_name"])


def test_a3_contains_no_duplicate_paths(a3) -> None:
    assert a3["path"].duplicated().sum() == 0


def test_the_duplicate_is_still_there_and_still_excluded(a3) -> None:
    # If Kaggle ever fixes the archive, the first assertion fails and the guard stops
    # being needed. Until then: prove the copy exists, prove it is a full copy, and
    # prove indexing the real root still returns 1,803 rather than 3,606.
    nested = A3_ROOT / "data"

    assert nested.is_dir(), "the nested copy is gone; revisit SPEC.md"
    assert len(a3_index(nested)) == 1803, "the nested directory holds a full duplicate"
    assert len(a3) == 1803, "indexing the real root must not pick the duplicate up"

    every_image = sum(1 for _ in A3_ROOT.rglob("*") if _.suffix.lower() in {".jpg", ".png", ".bmp"})
    assert every_image == 3606, "a recursive loader here would see twice the dataset"
