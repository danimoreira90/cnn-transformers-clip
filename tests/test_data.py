"""Unit tests for dataset indexing and splitting.

These run against synthetic fixtures, never the real archives. Facts about the real
datasets are asserted in `evals/test_dataset_facts.py`, which needs the downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_mba.data import (
    DatasetNotFoundError,
    DuplicateCorpusWarning,
    StratificationError,
    a3_index,
    ads16_index,
    covid_subsample,
    elpv_index,
    stratified_split,
)

NESTED = Path("src/elpv_dataset/data")


def write_elpv(root: Path, rows: list[tuple[str, float, str]], where: Path = NESTED) -> Path:
    """Build a miniature ELPV clone: a labels file plus the image files it points at."""
    data_dir = root / where
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for name, probability, wafer in rows:
        (data_dir / "images" / name).write_bytes(b"not a real png")
        lines.append(f"images/{name}  {probability}  {wafer}")
    (data_dir / "labels.csv").write_text("\n".join(lines) + "\n")
    return data_dir


BALANCED = [
    (f"cell{i:04d}.png", probability, wafer)
    for i, (probability, wafer) in enumerate(
        [(0.0, "mono"), (0.0, "poly"), (1.0, "mono"), (1.0, "poly")] * 25
    )
]


def test_index_finds_the_labels_file_in_the_nested_package_layout(tmp_path: Path) -> None:
    write_elpv(tmp_path, [("cell0001.png", 0.0, "mono")])

    frame = elpv_index(tmp_path)

    assert len(frame) == 1


def test_index_finds_the_labels_file_at_the_repository_root(tmp_path: Path) -> None:
    # The layout before the upstream repo was restructured. Both must work, because a
    # cached clone on Colab may predate the move.
    write_elpv(tmp_path, [("cell0001.png", 0.0, "mono")], where=Path("."))

    frame = elpv_index(tmp_path)

    assert len(frame) == 1


def test_index_raises_naming_every_path_it_searched(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError) as caught:
        elpv_index(tmp_path)

    message = str(caught.value)
    assert "labels.csv" in message
    assert "src/elpv_dataset/data" in message.replace("\\", "/")


def test_index_labels_anything_from_half_upwards_as_defective(tmp_path: Path) -> None:
    write_elpv(
        tmp_path,
        [
            ("cell0001.png", 0.0, "mono"),
            ("cell0002.png", 1 / 3, "mono"),
            ("cell0003.png", 0.5, "poly"),
            ("cell0004.png", 2 / 3, "poly"),
            ("cell0005.png", 1.0, "mono"),
        ],
    )

    frame = elpv_index(tmp_path)

    assert list(frame["label"]) == [0, 0, 1, 1, 1]


def test_index_returns_absolute_image_paths_that_exist(tmp_path: Path) -> None:
    write_elpv(tmp_path, [("cell0001.png", 0.0, "mono")])

    frame = elpv_index(tmp_path)

    path = Path(frame["path"].iloc[0])
    assert path.is_absolute()
    assert path.is_file()


def test_index_raises_when_a_referenced_image_is_missing(tmp_path: Path) -> None:
    data_dir = write_elpv(tmp_path, [("cell0001.png", 0.0, "mono")])
    (data_dir / "images" / "cell0001.png").unlink()

    with pytest.raises(DatasetNotFoundError, match="cell0001.png"):
        elpv_index(tmp_path)


def test_split_keeps_each_stratum_in_proportion(tmp_path: Path) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    train, test = stratified_split(frame, by=["label", "wafer_type"], test_size=0.2, seed=0)

    for keys, group in frame.groupby(["label", "wafer_type"]):
        in_test = len(test[(test["label"] == keys[0]) & (test["wafer_type"] == keys[1])])
        assert in_test / len(group) == pytest.approx(0.2, abs=0.01)


def test_split_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    first, _ = stratified_split(frame, by=["label", "wafer_type"], seed=7)
    second, _ = stratified_split(frame, by=["label", "wafer_type"], seed=7)

    assert list(first["path"]) == list(second["path"])


def test_split_changes_with_the_seed(tmp_path: Path) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    first, _ = stratified_split(frame, by=["label", "wafer_type"], seed=7)
    second, _ = stratified_split(frame, by=["label", "wafer_type"], seed=8)

    assert list(first["path"]) != list(second["path"])


def test_split_never_puts_the_same_image_in_both_halves(tmp_path: Path) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    train, test = stratified_split(frame, by=["label", "wafer_type"], seed=0)

    assert set(train["path"]).isdisjoint(set(test["path"]))
    assert len(train) + len(test) == len(frame)


def test_split_refuses_a_stratum_too_small_to_appear_in_both_halves(tmp_path: Path) -> None:
    # Every stratum is comfortably large except one lone cell of an unseen wafer type.
    # It cannot be in train and test at once, and silently dropping it from the test
    # half is how a group quietly stops being evaluated.
    rows = BALANCED + [("cell9999.png", 1.0, "unique")]
    write_elpv(tmp_path, rows)
    frame = elpv_index(tmp_path)

    with pytest.raises(StratificationError, match="unique"):
        stratified_split(frame, by=["label", "wafer_type"], test_size=0.2, seed=0)


def test_split_rejects_a_column_that_does_not_exist(tmp_path: Path) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    with pytest.raises(StratificationError, match="brightness"):
        stratified_split(frame, by=["label", "brightness"], seed=0)


@pytest.mark.parametrize("bad_size", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_a_test_size_outside_the_open_unit_interval(
    tmp_path: Path, bad_size: float
) -> None:
    write_elpv(tmp_path, BALANCED)
    frame = elpv_index(tmp_path)

    with pytest.raises(StratificationError, match="test_size"):
        stratified_split(frame, by=["label"], test_size=bad_size, seed=0)


# --------------------------------------------------------------------------------------
# A3 — the Kaggle images-dataset, which ships a copy of itself
# --------------------------------------------------------------------------------------

A3_CLASSES = ("bike", "cars", "cats", "dogs", "flowers", "horses", "human")


def write_a3(root: Path, per_class: int = 3, with_duplicate: bool = False) -> Path:
    """Build a miniature A3 archive, optionally including the nested copy of itself."""
    data_dir = root / "data"
    extension = {"bike": "bmp", "cars": "bmp", "flowers": "png"}
    for name in A3_CLASSES:
        folder = data_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            (folder / f"{name}_{i:03d}.{extension.get(name, 'jpg')}").write_bytes(b"img")
    if with_duplicate:
        nested = data_dir / "data"
        for name in A3_CLASSES:
            folder = nested / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                (folder / f"{name}_{i:03d}.{extension.get(name, 'jpg')}").write_bytes(b"img")
    return data_dir


def test_a3_index_finds_every_class_once(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=3)

    frame = a3_index(data_dir)

    assert len(frame) == 21
    assert sorted(frame["class_name"].unique()) == sorted(A3_CLASSES)


def test_a3_labels_are_alphabetical_and_stable(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=1)

    frame = a3_index(data_dir).sort_values("class_name")

    assert list(zip(frame["class_name"], frame["label"])) == [
        ("bike", 0), ("cars", 1), ("cats", 2), ("dogs", 3),
        ("flowers", 4), ("horses", 5), ("human", 6),
    ]


def test_a3_index_reads_bmp_jpg_and_png(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=2)

    frame = a3_index(data_dir)
    suffixes = {Path(path).suffix for path in frame["path"]}

    assert suffixes == {".bmp", ".jpg", ".png"}


def test_a3_index_excludes_the_archives_copy_of_itself(tmp_path: Path) -> None:
    # The real archive holds data/ and data/data/, an exact duplicate. This is the
    # guarantee that matters: the count is identical whether the copy is there or not.
    clean = write_a3(tmp_path / "clean", per_class=3)
    trapped = write_a3(tmp_path / "trapped", per_class=3, with_duplicate=True)

    with pytest.warns(DuplicateCorpusWarning, match="data"):
        indexed = a3_index(trapped)

    assert len(indexed) == len(a3_index(clean)) == 21
    assert not any("trapped" in path and f"data{Path('/').as_posix()}data" in path.replace("\\", "/")
                   for path in indexed["path"])


def test_a3_index_names_the_duplicate_directory_in_the_warning(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=2, with_duplicate=True)

    with pytest.warns(DuplicateCorpusWarning) as caught:
        a3_index(data_dir)

    assert str(data_dir / "data") in str(caught[0].message)


def test_a3_index_points_one_level_down_when_given_the_wrong_root(tmp_path: Path) -> None:
    write_a3(tmp_path, per_class=2)

    with pytest.raises(DatasetNotFoundError, match="data"):
        a3_index(tmp_path)


def test_a3_index_names_the_classes_it_could_not_find(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=2)
    for leftover in (data_dir / "flowers").iterdir():
        leftover.unlink()
    (data_dir / "flowers").rmdir()

    with pytest.raises(DatasetNotFoundError, match="flowers"):
        a3_index(data_dir)


def test_a3_index_does_not_descend_inside_a_class_folder(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=2)
    buried = data_dir / "cats" / "extra"
    buried.mkdir()
    (buried / "cats_999.jpg").write_bytes(b"img")

    frame = a3_index(data_dir)

    assert len(frame[frame["class_name"] == "cats"]) == 2


def test_a3_index_ignores_files_that_are_not_images(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=2)
    (data_dir / "cats" / "Thumbs.db").write_bytes(b"junk")
    (data_dir / "cats" / "notes.txt").write_text("ignore me")

    frame = a3_index(data_dir)

    assert len(frame[frame["class_name"] == "cats"]) == 2


def test_a3_index_returns_absolute_paths_that_exist(tmp_path: Path) -> None:
    data_dir = write_a3(tmp_path, per_class=1)

    frame = a3_index(data_dir)

    for path in frame["path"]:
        assert Path(path).is_absolute()
        assert Path(path).is_file()


# --------------------------------------------------------------------------------------
# A4.1 — the COVID chest X-ray subsample that reproduces the failed project
# --------------------------------------------------------------------------------------

def write_covid(root: Path, counts: dict[str, int]) -> Path:
    """Build a miniature COVID-19 Radiography clone with its images/ subfolders."""
    base = root / "COVID-19_Radiography_Dataset"
    for name, how_many in counts.items():
        folder = base / name / "images"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(how_many):
            (folder / f"{name}-{i:05d}.png").write_bytes(b"img")
    return base


PLENTY = {"Normal": 200, "Viral Pneumonia": 120, "COVID": 90, "Lung_Opacity": 50}
SMALL = {"Normal": 40, "Viral Pneumonia": 20, "COVID": 10}


def test_covid_subsample_takes_exactly_the_counts_asked_for(tmp_path: Path) -> None:
    base = write_covid(tmp_path, PLENTY)

    frame = covid_subsample(base, counts=SMALL, seed=0)

    assert frame["class_name"].value_counts().to_dict() == SMALL


def test_covid_labels_run_normal_pneumonia_covid(tmp_path: Path) -> None:
    # Fixed so a confusion matrix row 2 is always COVID. The metrics tests and the
    # report both read row 2 as the minority class; changing this silently would make
    # every recall figure in activity 4.1 refer to the wrong disease.
    base = write_covid(tmp_path, PLENTY)

    frame = covid_subsample(base, counts=SMALL, seed=0)
    mapping = dict(zip(frame["class_name"], frame["label"]))

    assert mapping == {"Normal": 0, "Viral Pneumonia": 1, "COVID": 2}


def test_covid_subsample_ignores_classes_it_was_not_asked_for(tmp_path: Path) -> None:
    base = write_covid(tmp_path, PLENTY)

    frame = covid_subsample(base, counts=SMALL, seed=0)

    assert "Lung_Opacity" not in set(frame["class_name"])


def test_covid_subsample_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    base = write_covid(tmp_path, PLENTY)

    first = covid_subsample(base, counts=SMALL, seed=3)
    second = covid_subsample(base, counts=SMALL, seed=3)

    assert list(first["path"]) == list(second["path"])


def test_covid_subsample_changes_with_the_seed(tmp_path: Path) -> None:
    base = write_covid(tmp_path, PLENTY)

    first = covid_subsample(base, counts=SMALL, seed=3)
    second = covid_subsample(base, counts=SMALL, seed=4)

    assert list(first["path"]) != list(second["path"])


def test_covid_subsample_holds_no_duplicate_images(tmp_path: Path) -> None:
    base = write_covid(tmp_path, PLENTY)

    frame = covid_subsample(base, counts=SMALL, seed=0)

    assert frame["path"].duplicated().sum() == 0


def test_covid_subsample_refuses_to_pad_a_class_that_is_too_small(tmp_path: Path) -> None:
    base = write_covid(tmp_path, {"Normal": 40, "Viral Pneumonia": 20, "COVID": 5})

    with pytest.raises(DatasetNotFoundError, match="COVID"):
        covid_subsample(base, counts=SMALL, seed=0)


def test_covid_subsample_names_a_class_folder_that_is_absent(tmp_path: Path) -> None:
    base = write_covid(tmp_path, {"Normal": 40, "Viral Pneumonia": 20})

    with pytest.raises(DatasetNotFoundError, match="COVID"):
        covid_subsample(base, counts=SMALL, seed=0)


# --------------------------------------------------------------------------------------
# A2 — ADS-16, where most of the images are not advertisements
# --------------------------------------------------------------------------------------

def write_ads16(root: Path, ads_per_folder: int = 2, pictures_per_bucket: int = 3) -> Path:
    """Build a miniature ADS-16 clone matching the real layout.

    Advertisements sit one level down; participant pictures sit two, split across the
    POS and NEG buckets. That extra level is what the first version of the loader missed.
    """
    for part, folders, participants in (("part1", ["1", "2"], ["U0001", "U0002"]),
                                        ("part2", ["3", "4"], ["U0003"])):
        stem = root / f"ADS16_Benchmark_{part}" / f"ADS16_Benchmark_{part}"
        for folder in folders:
            where = stem / "Ads" / "Ads" / folder
            where.mkdir(parents=True, exist_ok=True)
            for i in range(ads_per_folder):
                (where / f"{i}.png").write_bytes(b"img")
        for participant in participants:
            for bucket in ("POS", "NEG"):
                where = stem / "Corpus" / "Corpus" / participant / f"{participant}-IM-{bucket}"
                where.mkdir(parents=True, exist_ok=True)
                for i in range(pictures_per_bucket):
                    (where / f"{i}.png").write_bytes(b"img")
    return root


def test_ads16_separates_advertisements_from_participant_pictures(tmp_path: Path) -> None:
    write_ads16(tmp_path, ads_per_folder=2, pictures_per_bucket=3)

    frame = ads16_index(tmp_path)

    assert frame["partition"].value_counts().to_dict() == {"corpus": 18, "ads": 8}


def test_ads16_reaches_pictures_two_levels_below_the_participant(tmp_path: Path) -> None:
    # The regression that matters. Advertisements are one level down, participant
    # pictures are two. A loader written for the advertisement layout returns zero
    # corpus rows and says nothing, so the control group silently disappears.
    write_ads16(tmp_path, pictures_per_bucket=3)

    corpus = ads16_index(tmp_path).query("partition == 'corpus'")

    assert len(corpus) == 18


def test_ads16_records_the_positive_and_negative_buckets(tmp_path: Path) -> None:
    write_ads16(tmp_path, pictures_per_bucket=3)

    corpus = ads16_index(tmp_path).query("partition == 'corpus'")

    assert sorted(corpus["subset"].unique()) == ["NEG", "POS"]
    assert corpus["subset"].value_counts().to_dict() == {"POS": 9, "NEG": 9}


def test_ads16_tags_each_advertisement_with_the_folder_it_came_from(tmp_path: Path) -> None:
    write_ads16(tmp_path, ads_per_folder=2)

    ads = ads16_index(tmp_path).query("partition == 'ads'")

    assert sorted(ads["group"].unique()) == ["1", "2", "3", "4"]
    assert set(ads["subset"]) == {""}


def test_ads16_tags_each_picture_with_its_participant(tmp_path: Path) -> None:
    write_ads16(tmp_path)

    corpus = ads16_index(tmp_path).query("partition == 'corpus'")

    assert sorted(corpus["group"].unique()) == ["U0001", "U0002", "U0003"]


def test_ads16_reads_both_benchmark_parts(tmp_path: Path) -> None:
    write_ads16(tmp_path, ads_per_folder=2)

    frame = ads16_index(tmp_path)
    parts = {"part1" if "part1" in path else "part2" for path in frame["path"]}

    assert parts == {"part1", "part2"}


def test_ads16_raises_when_no_benchmark_part_is_present(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="ADS16_Benchmark"):
        ads16_index(tmp_path)


def test_ads16_returns_absolute_paths_with_no_duplicates(tmp_path: Path) -> None:
    write_ads16(tmp_path)

    frame = ads16_index(tmp_path)

    assert frame["path"].duplicated().sum() == 0
    assert all(Path(path).is_absolute() for path in frame["path"])
