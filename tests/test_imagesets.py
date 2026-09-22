"""Unit tests for turning an index of file paths into something torch can train on."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from cv_mba.imagesets import ImagePathDataset, evaluation_transform, training_transform

SIZE = 32


def write_images(root: Path, how_many: int, mode: str = "RGB") -> pd.DataFrame:
    rows = []
    for i in range(how_many):
        path = root / f"img_{i}.png"
        Image.new(mode, (40, 55), color=(i * 7 % 255) if mode == "L" else (i, 30, 200)).save(path)
        rows.append({"path": str(path), "label": i % 3})
    return pd.DataFrame(rows)


def test_one_item_per_row(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 5)

    dataset = ImagePathDataset(frame, evaluation_transform(SIZE))

    assert len(dataset) == 5


def test_an_item_is_an_image_tensor_and_an_integer_label(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 3)

    image, label = ImagePathDataset(frame, evaluation_transform(SIZE))[1]

    assert image.shape == (3, SIZE, SIZE)
    assert label.dtype == torch.long
    assert label.item() == 1


def test_greyscale_mode_gives_one_channel(tmp_path: Path) -> None:
    # ELPV cells are single-channel electroluminescence images. A from-scratch ViT can
    # take them as they are; only a pretrained backbone needs them repeated to three.
    frame = write_images(tmp_path, 2, mode="L")

    image, _ = ImagePathDataset(frame, evaluation_transform(SIZE, channels=1), channels=1)[0]

    assert image.shape == (1, SIZE, SIZE)


def test_a_greyscale_file_can_be_repeated_into_three_channels(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 2, mode="L")

    image, _ = ImagePathDataset(frame, evaluation_transform(SIZE, normalise=False), channels=3)[0]

    assert image.shape == (3, SIZE, SIZE)
    assert torch.allclose(image[0], image[1]) and torch.allclose(image[1], image[2])


def test_images_of_different_sizes_all_come_out_the_same(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 2)
    Image.new("RGB", (200, 97)).save(tmp_path / "img_0.png")

    dataset = ImagePathDataset(frame, evaluation_transform(SIZE))

    assert dataset[0][0].shape == dataset[1][0].shape == (3, SIZE, SIZE)


def test_a_missing_file_names_itself(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 2)
    Path(frame["path"].iloc[0]).unlink()

    with pytest.raises(FileNotFoundError, match="img_0.png"):
        ImagePathDataset(frame, evaluation_transform(SIZE))[0]


def test_the_evaluation_transform_is_a_function_not_a_sample(tmp_path: Path) -> None:
    # No randomness on the validation path. An augmented validation set measures a
    # different problem every epoch.
    frame = write_images(tmp_path, 1)
    dataset = ImagePathDataset(frame, evaluation_transform(SIZE))

    assert torch.equal(dataset[0][0], dataset[0][0])


def test_the_training_transform_does_vary(tmp_path: Path) -> None:
    frame = write_images(tmp_path, 1)
    asymmetric = Image.new("RGB", (40, 40), color=(0, 0, 0))
    for x in range(20):
        for y in range(40):
            asymmetric.putpixel((x, y), (255, 255, 255))
    asymmetric.save(frame["path"].iloc[0])
    torch.manual_seed(0)
    dataset = ImagePathDataset(frame, training_transform(SIZE))

    assert not torch.equal(dataset[0][0], dataset[0][0])


def test_imagenet_statistics_are_used_when_asked(tmp_path: Path) -> None:
    # Rubric 1.3 lists normalisation with the pretraining statistics. A pretrained
    # backbone expects inputs centred the way its own training data was.
    frame = write_images(tmp_path, 1)

    plain, _ = ImagePathDataset(frame, evaluation_transform(SIZE, normalise=False))[0]
    normalised, _ = ImagePathDataset(frame, evaluation_transform(SIZE, normalise=True))[0]

    assert plain.min() >= 0.0 and plain.max() <= 1.0
    assert not torch.allclose(plain, normalised)
