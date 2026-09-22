"""Making runs repeatable, so a measured difference is a real difference.

Two separate problems hide under the word "reproducible".

*Sampling.* Which images land in a split, how weights are initialised, what order batches
arrive in. Seeding the three random number generators fixes all of that.

*Arithmetic.* On a GPU, floating-point results can differ between two runs of the same
code. cuDNN picks convolution algorithms by benchmarking them, so the winner can change;
some reductions add numbers in whatever order threads happen to finish. The differences
land in the last bit or two, which sounds harmless until something sorts. A concept
ranking sorts similarity scores, and two concepts within a millionth of each other trade
places when those bits move, so a top-five table changes between identical runs.

SPEC.md allows capability evals C1 and C2 a single run rather than three, and that
permission is conditional on eval R4 — a bitwise reproducibility check — passing on the
T4. This module is what R4 tests.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, on CPU and every GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: int = 0, warn_only: bool = True) -> None:
    """Seed everything and ask torch for reproducible arithmetic.

    `warn_only` leaves the default as a warning rather than a hard failure, because a few
    operations have no deterministic implementation and would otherwise abort a training
    run that is reproducible in every way that matters. Anything that warns is recorded
    in the notebook rather than silenced.

    CUBLAS_WORKSPACE_CONFIG is set because cuBLAS matrix multiplication is only
    deterministic when its workspace is fixed, and torch refuses deterministic mode on
    CUDA without it. Setting it after CUDA has initialised has no effect, which is why
    this is called at the top of a notebook rather than halfway down.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
