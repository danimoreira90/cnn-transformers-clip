"""Turning an index of file paths into batches a model can train on.

The loaders in `cv_mba.data` return a table of paths and labels and stop there, so they
stay testable without torch. This module is the bridge: it opens the files, makes every
image the same size and shape, and hands back tensors.

Two transform pipelines, and the difference matters. The training one varies its output
on purpose; the evaluation one never does. An augmented validation set measures a
slightly different problem every epoch, so its curve stops being comparable with itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""Channel statistics of the ImageNet training set.

A pretrained backbone learned its filters on inputs centred this way. Feeding it raw
0-to-1 pixels shifts every activation and wastes part of what transfer learning is for.
"""

GREYSCALE_MEAN = (0.449,)
GREYSCALE_STD = (0.226,)
"""The same statistics collapsed to one channel, for single-channel work.

Three numbers cannot normalise a one-channel tensor, so the channel count has to reach
the transform. ELPV is single-channel, which is why this is not a hypothetical.
"""


class ImagePathDataset(Dataset):
    """A torch Dataset over a DataFrame of `path` and `label`.

    `channels` decides how each file is opened: 1 for single-channel work, 3 for a
    pretrained backbone. A greyscale file opened as 3 channels is repeated rather than
    colourised, which is how the single-channel ELPV cells reach an ImageNet model.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: transforms.Compose,
        channels: int = 3,
        label_column: str = "label",
    ) -> None:
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")

        self.paths = list(frame["path"])
        self.labels = list(frame[label_column])
        self.transform = transform
        self.mode = "L" if channels == 1 else "RGB"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path = Path(self.paths[index])
        if not path.is_file():
            raise FileNotFoundError(f"image listed in the index is missing: {path}")

        with Image.open(path) as image:
            converted = image.convert(self.mode)

        return self.transform(converted), torch.tensor(self.labels[index], dtype=torch.long)


def evaluation_transform(
    image_size: int, normalise: bool = True, channels: int = 3
) -> transforms.Compose:
    """Deterministic: resize, centre crop, to tensor, optionally normalise."""
    steps: list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalise:
        steps.append(_normalisation(channels))
    return transforms.Compose(steps)


def training_transform(
    image_size: int,
    normalise: bool = True,
    channels: int = 3,
    horizontal_flip: bool = True,
    rotation_degrees: float = 0.0,
    colour_jitter: float = 0.0,
) -> transforms.Compose:
    """Resize and crop with the augmentations that are switched on.

    Every augmentation defaults to off except the horizontal flip. Rubric 1.3 asks which
    augmentations would help and which would harm particular classes, and that argument
    is only worth making if each one can actually be switched independently.
    """
    steps: list = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if rotation_degrees:
        steps.append(transforms.RandomRotation(rotation_degrees))
    if colour_jitter:
        steps.append(transforms.ColorJitter(colour_jitter, colour_jitter, colour_jitter))
    steps.append(transforms.ToTensor())
    if normalise:
        steps.append(_normalisation(channels))
    return transforms.Compose(steps)


def _normalisation(channels: int) -> transforms.Normalize:
    if channels == 1:
        return transforms.Normalize(mean=GREYSCALE_MEAN, std=GREYSCALE_STD)
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
