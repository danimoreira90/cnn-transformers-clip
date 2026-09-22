"""Dataset indexing and splitting.

Two failures this module exists to make impossible:

Silent label noise. ELPV labels are annotator agreement, not verdicts. A cell scored
0.333 is one where roughly two experts in three said it was fine. Folding those into the
defective class puts about a quarter of that class in dispute, and then a weak
from-scratch result cannot be read: you cannot tell data scarcity from contradictory
labels. The threshold is 0.5.

Unstratified splitting. The failed X-ray project in activity 4.1 split 80/20 with no
stratification. Here every split is stratified on the columns given, and a stratum too
small to appear in both halves raises instead of quietly vanishing from evaluation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFECT_THRESHOLD = 0.5
"""Defect probability at or above which a cell counts as defective.

Annotator agreement runs 0.0, 0.333, 0.667, 1.0. Cutting at 0.5 keeps the two confident
buckets apart from the two disputed ones and leaves 1,803 functional against 821
defective.
"""

_ELPV_LABEL_LOCATIONS = (
    Path("src/elpv_dataset/data"),  # current upstream layout
    Path("."),  # layout before the repository was restructured
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset is absent or incomplete, naming what was looked for."""


class StratificationError(ValueError):
    """Raised when a split cannot preserve the composition it was asked to preserve."""


def elpv_index(root: Path | str, verify_files: bool = True) -> pd.DataFrame:
    """Index the ELPV solar cell dataset from a clone at `root`.

    Returns one row per cell with `path`, `defect_probability`, `wafer_type` and the
    binary `label` at DEFECT_THRESHOLD.

    `verify_files` stats every image. A clone whose large files never arrived still has a
    complete labels file, so without this check the dataset looks fine until training
    fails a long way downstream.
    """
    root = Path(root)

    searched = [root / location / "labels.csv" for location in _ELPV_LABEL_LOCATIONS]
    labels_file = next((path for path in searched if path.is_file()), None)
    if labels_file is None:
        listing = "\n  ".join(str(path) for path in searched)
        raise DatasetNotFoundError(f"no ELPV labels.csv found. Looked in:\n  {listing}")

    frame = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None,
        names=["relative_path", "defect_probability", "wafer_type"],
    )

    data_dir = labels_file.parent
    frame["path"] = [
        str((data_dir / relative).resolve()) for relative in frame["relative_path"]
    ]
    frame["label"] = (frame["defect_probability"] >= DEFECT_THRESHOLD).astype(int)

    if verify_files:
        absent = [path for path in frame["path"] if not Path(path).is_file()]
        if absent:
            shown = "\n  ".join(absent[:5])
            raise DatasetNotFoundError(
                f"{len(absent)} image(s) listed in labels.csv are missing, "
                f"starting with:\n  {shown}"
            )

    return frame[["path", "defect_probability", "wafer_type", "label"]]


def stratified_split(
    frame: pd.DataFrame,
    by: Sequence[str],
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `frame` in two, preserving the share of every combination of `by` columns.

    ELPV is stratified on label and wafer type together, not on label alone. Defect rates
    differ by wafer type — 34.4% of mono cells against 29.2% of poly — so a split that
    balances only the label lets wafer type drift between the halves, and part of any
    measured difference between two models becomes which wafers happened to land where.

    Raises StratificationError if any stratum is too small to appear in both halves,
    rather than dropping it from one of them.
    """
    if not 0.0 < test_size < 1.0:
        raise StratificationError(f"test_size must be between 0 and 1, got {test_size}")

    unknown = [column for column in by if column not in frame.columns]
    if unknown:
        raise StratificationError(f"no such column(s) to stratify on: {', '.join(unknown)}")

    rng = np.random.default_rng(seed)
    test_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []

    for keys, group in frame.groupby(list(by), sort=True):
        n_test = int(round(len(group) * test_size))
        if n_test == 0 or n_test == len(group):
            label = ", ".join(f"{column}={value}" for column, value in zip(by, np.atleast_1d(keys)))
            raise StratificationError(
                f"stratum ({label}) has {len(group)} row(s), too few to appear in both "
                f"halves at test_size={test_size}"
            )

        shuffled = group.iloc[rng.permutation(len(group))]
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train = pd.concat(train_parts).sort_index()
    test = pd.concat(test_parts).sort_index()
    return train, test


A3_CLASSES = ("bike", "cars", "cats", "dogs", "flowers", "horses", "human")
"""The seven class folders in the Kaggle images-dataset, in label order.

Labels are the alphabetical index, matching what torchvision's ImageFolder would assign,
so a model trained with either route reads the same.
"""

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
"""Formats present in A3. They vary by class: bmp for bike and cars, png for flowers,
jpg for the rest. Any loader that assumes one format silently loses whole classes."""


class DuplicateCorpusWarning(UserWarning):
    """Warns that a dataset directory holds a second copy of its own images.

    The A3 archive ships `data/` and, inside it, `data/data/` — an exact duplicate. A
    recursive load returns 3,606 images instead of 1,803, invents an eighth class named
    `data`, and puts every picture in the set twice. Split that 80/20 and the same image
    lands in training and validation, so validation accuracy measures memorisation.

    `a3_index` cannot fall into this, because it reads the seven named class folders and
    nothing else. The warning exists because `ImageFolder` and every other recursive
    loader can, and someone reaching for one needs to know the trap is there.
    """


def _images_in(folder: Path) -> list[Path]:
    """Image files directly inside `folder`. Never recurses, by design."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def a3_index(root: Path | str) -> pd.DataFrame:
    """Index the Kaggle images-dataset from the directory holding the class folders.

    Returns one row per image with `path`, `class_name` and `label`.

    Enumerates the seven named class folders, non-recursively, and nothing else. The
    count is therefore unaffected by anything else sitting in `root` — including the
    archive's copy of itself. Any such directory raises a DuplicateCorpusWarning naming
    it, because a recursive loader pointed at the same place would silently double the
    dataset.
    """
    root = Path(root)

    if not root.is_dir():
        raise DatasetNotFoundError(f"not a directory: {root}")

    missing = [name for name in A3_CLASSES if not (root / name).is_dir()]
    if missing:
        one_level_down = root / "data"
        if one_level_down.is_dir() and not [
            name for name in A3_CLASSES if not (one_level_down / name).is_dir()
        ]:
            raise DatasetNotFoundError(
                f"no class folders in {root}, but all seven are in {one_level_down}. "
                f"Point a3_index at {one_level_down}"
            )
        raise DatasetNotFoundError(
            f"missing class folder(s) under {root}: {', '.join(missing)}"
        )

    intruders = [
        folder
        for folder in sorted(root.iterdir())
        if folder.is_dir()
        and folder.name not in A3_CLASSES
        and _has_images_anywhere(folder)
    ]
    if intruders:
        listing = ", ".join(str(folder) for folder in intruders)
        warnings.warn(
            f"directory alongside the class folders also holds images: {listing}. "
            f"Excluded from this index, which reads only {len(A3_CLASSES)} named class "
            f"folders. A recursive loader pointed here would double every image and leak "
            f"it across a train/validation split",
            DuplicateCorpusWarning,
            stacklevel=2,
        )

    rows = [
        {"path": str(path.resolve()), "class_name": name, "label": label}
        for label, name in enumerate(A3_CLASSES)
        for path in _images_in(root / name)
    ]

    return pd.DataFrame(rows, columns=["path", "class_name", "label"])


def _has_images_anywhere(folder: Path) -> bool:
    """True if `folder` contains an image at any depth."""
    return any(
        path.suffix.lower() in _IMAGE_SUFFIXES for path in folder.rglob("*") if path.is_file()
    )


COVID_CLASSES = ("Normal", "Viral Pneumonia", "COVID")
"""The three X-ray classes, in label order.

Fixed deliberately: row 2 of any confusion matrix is COVID, the minority class whose
recall the whole of activity 4.1 turns on. Reordering these would silently re-point every
recall figure in the report at a different disease.
"""

FAILED_PROJECT_COUNTS = {"Normal": 840, "Viral Pneumonia": 240, "COVID": 120}
"""The split the previous group used: 1,200 images, 7 to 2 to 1.

Reproduced exactly rather than described, so every later number is a comparison against a
measured baseline instead of an assertion about one.
"""


def covid_subsample(
    root: Path | str,
    counts: dict[str, int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw a seeded subsample from the COVID-19 Radiography Database.

    Returns `path`, `class_name` and `label`, with labels following COVID_CLASSES.

    Filenames are sorted before sampling so the draw depends only on the seed, never on
    the order the filesystem happens to return. Without that, the "same" baseline differs
    between machines and the with-and-without comparison in F5.4 compares two different
    experiments.

    A class with fewer images than requested raises rather than returning a short class.
    Quietly delivering 90 images where 120 were asked for would change the imbalance
    being studied, which is the one thing activity 4.1 is about.
    """
    root = Path(root)
    wanted = dict(counts if counts is not None else FAILED_PROJECT_COUNTS)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for label, name in enumerate(COVID_CLASSES):
        if name not in wanted:
            continue

        folder = root / name / "images"
        if not folder.is_dir():
            folder = root / name
        if not folder.is_dir():
            raise DatasetNotFoundError(f"no folder for class {name!r} under {root}")

        available = _images_in(folder)
        if len(available) < wanted[name]:
            raise DatasetNotFoundError(
                f"class {name!r} has {len(available)} image(s) but {wanted[name]} were "
                f"requested; sampling cannot invent the difference"
            )

        chosen = rng.choice(len(available), size=wanted[name], replace=False)
        rows.extend(
            {"path": str(available[index].resolve()), "class_name": name, "label": label}
            for index in sorted(chosen)
        )

    return pd.DataFrame(rows, columns=["path", "class_name", "label"])


def ads16_index(root: Path | str) -> pd.DataFrame:
    """Index ADS-16, separating advertisements from participant photographs.

    Returns `path`, `partition` (`ads` or `corpus`), `group` and `subset`:

    - ads      — `group` is the numbered folder the advertisement sits in, `subset` empty
    - corpus   — `group` is the participant (U0001 ... U0120), `subset` is POS or NEG,
                 the two folders each participant's pictures are split across

    The separation is the point. Only 301 of the 2,697 images are advertisements; the
    other 2,396 belong to the 120 survey participants. Treating the whole archive as an
    advertising corpus would rank personal photographs as advertising content. Keeping
    the partitions apart turns that hazard into the control group in SPEC.md F2.1.

    Measured on the real archive, 2026-09-22: 301 advertisements across 20 numbered
    folders, and 2,396 participant images across 120 participants, between 10 and 30
    each. The archive ships no file naming the folders, so `group` is the folder as
    found, never an invented category label.
    """
    root = Path(root)

    parts = sorted(root.glob("ADS16_Benchmark_part*/ADS16_Benchmark_part*"))
    if not parts:
        raise DatasetNotFoundError(f"no ADS16_Benchmark_part* directories under {root}")

    rows: list[dict[str, object]] = []
    for part in parts:
        rows.extend(_ads16_advertisements(part / "Ads" / "Ads"))
        rows.extend(_ads16_participant_images(part / "Corpus" / "Corpus"))

    return pd.DataFrame(rows, columns=["path", "partition", "group", "subset"])


def _ads16_advertisements(container: Path) -> list[dict[str, object]]:
    """Advertisements, one level down: <container>/<folder>/<image>."""
    if not container.is_dir():
        return []

    return [
        {"path": str(image.resolve()), "partition": "ads", "group": folder.name, "subset": ""}
        for folder in sorted(container.iterdir())
        if folder.is_dir()
        for image in _images_in(folder)
    ]


def _ads16_participant_images(container: Path) -> list[dict[str, object]]:
    """Participant pictures, two levels down: <container>/U0001/U0001-IM-POS/<image>.

    The extra level is why a loader written against the advertisement layout returns
    nothing here rather than failing loudly.
    """
    if not container.is_dir():
        return []

    return [
        {
            "path": str(image.resolve()),
            "partition": "corpus",
            "group": participant.name,
            "subset": bucket.name.rsplit("-", 1)[-1],
        }
        for participant in sorted(container.iterdir())
        if participant.is_dir()
        for bucket in sorted(participant.iterdir())
        if bucket.is_dir()
        for image in _images_in(bucket)
    ]
