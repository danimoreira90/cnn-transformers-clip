"""Unit tests for the CLIP retrieval logic.

The model calls themselves are exercised in `evals/`, against real CLIP weights. What is
tested here is the arithmetic around them, which is where the reasoning lives: how
similarity is measured, what counts as a concept being present, and how a ranking is
built. All of it runs on hand-made matrices in microseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cv_mba.clip_search import (
    cosine_similarity,
    features_of,
    l2_normalise,
    rank_concepts,
    top_matches,
)


def test_normalising_gives_every_row_unit_length() -> None:
    x = torch.randn(5, 8) * 17

    normalised = l2_normalise(x)

    assert torch.allclose(normalised.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_normalising_keeps_direction() -> None:
    x = torch.randn(3, 6)

    normalised = l2_normalise(x)

    for original, unit in zip(x, normalised):
        assert torch.allclose(unit * original.norm(), original, atol=1e-5)


def test_a_zero_row_does_not_become_nan() -> None:
    x = torch.zeros(2, 4)

    assert torch.isfinite(l2_normalise(x)).all()


def test_similarity_of_identical_directions_is_one() -> None:
    a = torch.randn(4, 8)

    similarity = cosine_similarity(a, a.clone())

    assert torch.allclose(torch.diagonal(similarity), torch.ones(4), atol=1e-5)


def test_similarity_stays_inside_minus_one_and_one() -> None:
    a, b = torch.randn(6, 10), torch.randn(4, 10)

    similarity = cosine_similarity(a, b)

    assert similarity.shape == (6, 4)
    assert similarity.min() >= -1.0 - 1e-5
    assert similarity.max() <= 1.0 + 1e-5


def test_opposite_directions_score_minus_one() -> None:
    a = torch.randn(3, 5)

    similarity = cosine_similarity(a, -a)

    assert torch.allclose(torch.diagonal(similarity), -torch.ones(3), atol=1e-5)


def test_top_matches_returns_the_highest_scores_in_order() -> None:
    scores = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2])

    indices, values = top_matches(scores, k=3)

    assert indices.tolist() == [1, 3, 2]
    assert values.tolist() == pytest.approx([0.9, 0.7, 0.5])


def test_top_matches_never_asks_for_more_than_exists() -> None:
    scores = torch.tensor([0.4, 0.8])

    indices, _ = top_matches(scores, k=10)

    assert len(indices) == 2


# --------------------------------------------------------------------------------------
# Concept ranking
# --------------------------------------------------------------------------------------

CONCEPTS = ["a car", "a person", "text and logo"]


def test_ranking_reports_mean_similarity_and_frequency_per_concept() -> None:
    similarity = torch.tensor([
        [0.9, 0.1, 0.2],
        [0.8, 0.2, 0.1],
        [0.1, 0.9, 0.3],
    ])

    ranking = rank_concepts(similarity, CONCEPTS, threshold=0.5, standardise=False)

    assert list(ranking.columns) == ["concept", "mean_similarity", "frequency", "count"]
    by_concept = ranking.set_index("concept")
    assert by_concept.loc["a car", "count"] == 2
    assert by_concept.loc["a person", "count"] == 1
    assert by_concept.loc["text and logo", "count"] == 0
    assert by_concept.loc["a car", "mean_similarity"] == pytest.approx(0.6, abs=1e-6)


def test_ranking_is_sorted_by_frequency_then_similarity() -> None:
    similarity = torch.tensor([[0.6, 0.9, 0.1], [0.6, 0.9, 0.1]])

    ranking = rank_concepts(similarity, CONCEPTS, threshold=0.5, standardise=False)

    assert list(ranking["concept"]) == ["a person", "a car", "text and logo"]


def test_frequency_is_a_share_of_the_corpus() -> None:
    similarity = torch.tensor([[0.9, 0.1, 0.1]] * 4 + [[0.1, 0.1, 0.1]] * 6)

    ranking = rank_concepts(similarity, CONCEPTS, threshold=0.5, standardise=False)

    assert ranking.set_index("concept").loc["a car", "frequency"] == pytest.approx(0.4)


def test_standardising_compares_concepts_within_one_image_not_across_images() -> None:
    # CLIP's absolute similarity values are not comparable between images: a busy
    # photograph scores lower against everything than a plain one does. A single absolute
    # threshold therefore counts concepts in bright simple pictures and misses them in
    # crowded ones. Standardising each row asks a better question — which concepts stand
    # out *in this image* — and is why the notebook uses it.
    dim = torch.tensor([[0.30, 0.10, 0.11]])   # one concept clearly dominant, low overall
    bright = torch.tensor([[0.80, 0.60, 0.61]])  # same shape, shifted up

    dim_ranking = rank_concepts(dim, CONCEPTS, threshold=1.0, standardise=True)
    bright_ranking = rank_concepts(bright, CONCEPTS, threshold=1.0, standardise=True)

    assert dim_ranking.set_index("concept").loc["a car", "count"] == 1
    assert bright_ranking.set_index("concept").loc["a car", "count"] == 1


def test_an_absolute_threshold_misses_the_dim_image() -> None:
    # The same two images under a fixed cut. This is the failure standardising avoids,
    # kept as a test so the justification in the report is a measurement.
    dim = torch.tensor([[0.30, 0.10, 0.11]])
    bright = torch.tensor([[0.80, 0.60, 0.61]])

    dim_ranking = rank_concepts(dim, CONCEPTS, threshold=0.5, standardise=False)
    bright_ranking = rank_concepts(bright, CONCEPTS, threshold=0.5, standardise=False)

    assert dim_ranking.set_index("concept").loc["a car", "count"] == 0
    assert bright_ranking.set_index("concept").loc["a car", "count"] == 1


def test_a_concept_list_that_does_not_match_the_matrix_is_refused() -> None:
    similarity = torch.randn(4, 3)

    with pytest.raises(ValueError, match="concepts"):
        rank_concepts(similarity, ["only", "two"], threshold=0.5)


def test_ranking_accepts_numpy_as_well_as_torch() -> None:
    similarity = np.array([[0.9, 0.1, 0.2], [0.8, 0.2, 0.1]])

    ranking = rank_concepts(similarity, CONCEPTS, threshold=0.5, standardise=False)

    assert len(ranking) == 3


# --------------------------------------------------------------------------------------
# Reading embeddings out of whichever shape the model returns
# --------------------------------------------------------------------------------------

def test_a_plain_tensor_is_passed_straight_through() -> None:
    x = torch.randn(3, 512)

    assert features_of(x) is x


def test_a_pooled_output_object_gives_up_its_pooler_output() -> None:
    # transformers 5.x wraps the embedding in an output object. A wrapper written for
    # 4.x fails on Colab with an attribute error, which is how this was found.
    class PooledOutput:
        def __init__(self, pooled): self.pooler_output = pooled

    pooled = torch.randn(3, 512)

    assert torch.equal(features_of(PooledOutput(pooled)), pooled)


def test_an_unreadable_output_says_what_it_expected() -> None:
    with pytest.raises(TypeError, match="pooler_output"):
        features_of(object())
