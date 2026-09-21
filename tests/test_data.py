"""Unit tests for dataset indexing and splitting.

These run against synthetic fixtures, never the real archives. Facts about the real
datasets are asserted in `evals/test_dataset_facts.py`, which needs the downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_mba.data import (
    DatasetNotFoundError,
    StratificationError,
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
