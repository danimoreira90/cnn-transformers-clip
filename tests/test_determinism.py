"""Unit tests for the determinism harness.

Eval R4 depends on this. C1 and C2 are allowed a single run instead of three only while
R4 passes, so what is tested here decides whether two capability evals are trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cv_mba.determinism import enable_determinism, seed_everything
from cv_mba.vit import VisionTransformer


def test_seeding_makes_torch_numpy_and_python_repeat() -> None:
    import random

    seed_everything(11)
    first = (torch.randn(4), np.random.rand(4), random.random())
    seed_everything(11)
    second = (torch.randn(4), np.random.rand(4), random.random())

    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_different_seeds_give_different_draws() -> None:
    seed_everything(1)
    first = torch.randn(8)
    seed_everything(2)
    second = torch.randn(8)

    assert not torch.equal(first, second)


def test_enabling_determinism_sets_the_flags_that_matter() -> None:
    enable_determinism(0)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_two_identical_forward_passes_agree_bit_for_bit() -> None:
    # The claim behind eval R4. On CPU this is nearly free; the run that counts is on a
    # T4, where cuDNN picks algorithms by benchmarking and some reductions accumulate in
    # thread-completion order. A concept ranking sorts similarity scores, so two scores
    # within 1e-6 of each other swap places when those last bits move.
    enable_determinism(7)
    model = VisionTransformer(32, 8, 1, 2, d_model=24, n_heads=4, n_layers=2).eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        first, second = model(x), model(x)

    assert torch.equal(first, second), "identical inputs produced different outputs"


def test_a_model_built_twice_under_the_same_seed_is_identical() -> None:
    # Weight initialisation is the other half. Without it, "same seed" reproduces the
    # data order but not the model, and a seeded comparison still moves between runs.
    enable_determinism(3)
    first = VisionTransformer(32, 8, 1, 2, d_model=24, n_heads=4, n_layers=2)
    enable_determinism(3)
    second = VisionTransformer(32, 8, 1, 2, d_model=24, n_heads=4, n_layers=2)

    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between two seeded constructions"


@pytest.mark.gpu
def test_the_gpu_agrees_with_itself() -> None:
    # Marked gpu and skipped with a reason on CPU. This is a skip for absent hardware,
    # never for a failing assertion.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; eval R4 is only meaningful on the T4 this runs on")

    enable_determinism(7)
    model = VisionTransformer(32, 8, 1, 2, d_model=24, n_heads=4, n_layers=2).cuda().eval()
    x = torch.randn(2, 1, 32, 32, device="cuda")

    with torch.no_grad():
        first, second = model(x), model(x)

    assert torch.equal(first, second)
