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
