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

from cv_mba.data import (
    FAILED_PROJECT_COUNTS,
    DuplicateCorpusWarning,
    a3_index,
    ads16_index,
    covid_subsample,
    elpv_index,
    stratified_split,
)

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


COVID_ROOT = DATA_ROOT / "covid" / "COVID-19_Radiography_Dataset"
ADS16_ROOT = DATA_ROOT / "ads16"

COVID_FETCH = "uv run kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p data/raw/covid --unzip"
ADS16_FETCH = "uv run kaggle datasets download -d groffo/ads16-dataset -p data/raw/ads16 --unzip"


@pytest.fixture(scope="module")
def covid():
    if not (COVID_ROOT / "COVID").is_dir():
        pytest.skip(f"COVID archive not at {COVID_ROOT}. Fetch it with:  {COVID_FETCH}")
    return covid_subsample(COVID_ROOT, seed=0)


@pytest.fixture(scope="module")
def ads16():
    if not any(ADS16_ROOT.glob("ADS16_Benchmark_part*")):
        pytest.skip(f"ADS-16 archive not at {ADS16_ROOT}. Fetch it with:  {ADS16_FETCH}")
    return ads16_index(ADS16_ROOT)


def test_covid_subsample_reproduces_the_failed_projects_split(covid) -> None:
    assert covid["class_name"].value_counts().to_dict() == FAILED_PROJECT_COUNTS
    assert len(covid) == 1200


def test_covid_subsample_is_identical_across_runs(covid) -> None:
    again = covid_subsample(COVID_ROOT, seed=0)

    assert list(again["path"]) == list(covid["path"])


def test_the_archive_still_has_enough_images_for_the_split(covid) -> None:
    # 3,616 COVID images exist; the failed project used 120. If the archive ever shrinks
    # below the requested counts, covid_subsample raises and this fails first.
    generous = covid_subsample(COVID_ROOT, counts={"COVID": 1000}, seed=0)

    assert len(generous) == 1000


def test_ads16_holds_301_advertisements_among_2697_images(ads16) -> None:
    counts = ads16["partition"].value_counts().to_dict()

    assert counts == {"corpus": 2396, "ads": 301}
    assert len(ads16) == 2697


def test_ads16_advertisements_sit_in_twenty_numbered_folders(ads16) -> None:
    # The brief says "16 categorias de produto". The archive ships 20 numbered folders
    # and no file naming any of them. Measured, not assumed: 16 images in folder 1 and
    # 15 in each of folders 2 to 20, totalling 301. The discrepancy is recorded in
    # SPEC.md and belongs in the report's critical analysis; inventing category labels
    # to make the data match the brief would be the wrong repair.
    folders = ads16.query("partition == 'ads'")["group"]

    assert len(folders.unique()) == 20
    assert sorted(folders.value_counts().unique()) == [15, 16]


def test_ads16_corpus_covers_all_120_survey_participants(ads16) -> None:
    corpus = ads16.query("partition == 'corpus'")

    assert len(corpus["group"].unique()) == 120


def test_ads16_participant_pictures_split_into_positive_and_negative(ads16) -> None:
    corpus = ads16.query("partition == 'corpus'")

    assert sorted(corpus["subset"].unique()) == ["NEG", "POS"]


def test_ads16_participants_contribute_between_10_and_30_pictures(ads16) -> None:
    # Not the uniform 10 per participant the source paper implies. Most contribute 20.
    per_participant = ads16.query("partition == 'corpus'").groupby("group").size()

    assert per_participant.min() == 10
    assert per_participant.max() == 30
    assert per_participant.sum() == 2396


def test_ads16_ad_count_falls_short_of_the_briefs_500_threshold(ads16) -> None:
    # The brief offers "a subset of at least 500 representative images". The advertising
    # corpus is 301. This is the measurement behind SPEC.md F2.1, which resolves the gap
    # by reporting the ad and corpus partitions separately rather than padding the ads
    # with participant pictures to clear a number.
    assert len(ads16.query("partition == 'ads'")) < 500
