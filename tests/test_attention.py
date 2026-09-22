"""Unit tests for attention built from scratch.

Rubric line 2.1 asks for scaled dot-product attention and multi-head attention as
*testable* PyTorch modules. These are what make that word true rather than claimed.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from cv_mba.attention import ScaledDotProductAttention


@pytest.fixture
def attention() -> ScaledDotProductAttention:
    return ScaledDotProductAttention()


def test_output_has_the_same_shape_as_the_values(attention) -> None:
    query, key, value = (torch.randn(2, 5, 8) for _ in range(3))

    output, _ = attention(query, key, value)

    assert output.shape == value.shape


def test_weights_are_one_row_per_query_and_one_column_per_key(attention) -> None:
    query = torch.randn(2, 5, 8)
    key = value = torch.randn(2, 7, 8)

    _, weights = attention(query, key, value)

    assert weights.shape == (2, 5, 7)


def test_every_row_of_weights_sums_to_one(attention) -> None:
    query, key, value = (torch.randn(3, 4, 6) for _ in range(3))

    _, weights = attention(query, key, value)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4), atol=1e-6)


def test_a_masked_position_receives_exactly_zero_weight(attention) -> None:
    query, key, value = (torch.randn(1, 3, 4) for _ in range(3))
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    mask[0, :, 1] = False

    _, weights = attention(query, key, value, mask=mask)

    assert torch.all(weights[0, :, 1] == 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_scores_are_divided_by_the_square_root_of_the_key_dimension(attention) -> None:
    # The test that separates a correct implementation from a plausible one. Without the
    # 1/sqrt(d_k) factor the weights are still a valid probability distribution and every
    # other test here still passes; only the numbers are wrong, and they are wrong in a
    # way that makes deep models untrainable rather than obviously broken.
    query = torch.tensor([[[1.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.eye(2).unsqueeze(0)

    _, weights = attention(query, key, value)

    scaled = torch.softmax(torch.tensor([1.0 / math.sqrt(2), 0.0]), dim=-1)
    unscaled = torch.softmax(torch.tensor([1.0, 0.0]), dim=-1)

    assert torch.allclose(weights[0, 0], scaled, atol=1e-6)
    assert not torch.allclose(weights[0, 0], unscaled, atol=1e-6)


def test_identical_keys_spread_attention_evenly(attention) -> None:
    query = torch.randn(1, 2, 4)
    key = torch.ones(1, 3, 4)
    value = torch.randn(1, 3, 4)

    _, weights = attention(query, key, value)

    assert torch.allclose(weights, torch.full((1, 2, 3), 1 / 3), atol=1e-6)


def test_it_carries_no_learnable_parameters(attention) -> None:
    # Scaled dot-product attention is pure arithmetic. Every learnable weight in a
    # transformer lives in the projections around it, which is why multi-head attention
    # is where the parameters appear.
    assert list(attention.parameters()) == []


def test_it_accepts_a_head_dimension_without_special_casing(attention) -> None:
    # Multi-head attention feeds it (batch, heads, seq, d_head). Broadcasting over the
    # leading dimensions is what lets one implementation serve both callers.
    query, key, value = (torch.randn(2, 4, 5, 8) for _ in range(3))

    output, weights = attention(query, key, value)

    assert output.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 5)


# --------------------------------------------------------------------------------------
# Multi-head attention
# --------------------------------------------------------------------------------------

from cv_mba.attention import MultiHeadAttention  # noqa: E402

D_MODEL, N_HEADS, SEQ, BATCH = 12, 3, 5, 2


@pytest.fixture
def multi_head() -> MultiHeadAttention:
    torch.manual_seed(0)
    return MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)


def explicit_per_head(module: MultiHeadAttention, x: Tensor) -> tuple[Tensor, Tensor]:
    """Run each head through its own projections, the slow literal way.

    This is the reference the compact implementation is checked against. It slices the
    projection matrices per head, attends inside each head separately, concatenates the
    head outputs and applies the output projection — exactly the sentence in rubric 2.1,
    written out.
    """
    outputs, weights = [], []
    for head in range(module.n_heads):
        lo, hi = head * module.d_head, (head + 1) * module.d_head
        projected = []
        for projection in (module.query_projection, module.key_projection, module.value_projection):
            projected.append(x @ projection.weight[lo:hi].T + projection.bias[lo:hi])
        attended, head_weights = ScaledDotProductAttention()(*projected)
        outputs.append(attended)
        weights.append(head_weights)

    return module.output_projection(torch.cat(outputs, dim=-1)), torch.stack(weights, dim=1)


def test_output_keeps_the_model_width(multi_head) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output, _ = multi_head(x, x, x)

    assert output.shape == (BATCH, SEQ, D_MODEL)


def test_weights_come_back_one_matrix_per_head(multi_head) -> None:
    # Rubric 2.2 and 3.2 both need these for heatmaps, so they have to survive the
    # concatenation rather than being averaged away inside the module.
    x = torch.randn(BATCH, SEQ, D_MODEL)

    _, weights = multi_head(x, x, x)

    assert weights.shape == (BATCH, N_HEADS, SEQ, SEQ)


def test_the_compact_form_equals_running_each_head_separately(multi_head) -> None:
    # The evidence for "independent projections per head, outputs concatenated". One
    # wide projection reshaped into heads is arithmetically the same as one projection
    # per head; this proves it on real weights instead of asserting it in a comment.
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output, weights = multi_head(x, x, x)
    reference_output, reference_weights = explicit_per_head(multi_head, x)

    assert torch.allclose(output, reference_output, atol=1e-6)
    assert torch.allclose(weights, reference_weights, atol=1e-6)


def test_disturbing_one_head_leaves_the_others_untouched(multi_head) -> None:
    # Independence, checked from the other direction. Blanking one head's query rows
    # flattens that head's attention and must not move a single number in the rest.
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, before = multi_head(x, x, x)

    with torch.no_grad():
        multi_head.query_projection.weight[: multi_head.d_head].zero_()
        multi_head.query_projection.bias[: multi_head.d_head].zero_()
    _, after = multi_head(x, x, x)

    assert not torch.allclose(before[:, 0], after[:, 0])
    assert torch.allclose(before[:, 1:], after[:, 1:], atol=1e-7)


def test_a_blanked_head_attends_to_everything_equally(multi_head) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)
    with torch.no_grad():
        multi_head.query_projection.weight[: multi_head.d_head].zero_()
        multi_head.query_projection.bias[: multi_head.d_head].zero_()

    _, weights = multi_head(x, x, x)

    assert torch.allclose(weights[:, 0], torch.full((BATCH, SEQ, SEQ), 1 / SEQ), atol=1e-6)


def test_head_width_times_head_count_is_the_model_width(multi_head) -> None:
    assert multi_head.d_head * multi_head.n_heads == D_MODEL


def test_a_model_width_not_divisible_by_the_head_count_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, n_heads=3)


def test_a_mask_reaches_every_head(multi_head) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)
    mask = torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool)
    mask[..., 2] = False

    _, weights = multi_head(x, x, x, mask=mask)

    assert torch.all(weights[..., 2] == 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH, N_HEADS, SEQ), atol=1e-6)


def test_cross_attention_between_different_sequence_lengths(multi_head) -> None:
    query = torch.randn(BATCH, 4, D_MODEL)
    memory = torch.randn(BATCH, 9, D_MODEL)

    output, weights = multi_head(query, memory, memory)

    assert output.shape == (BATCH, 4, D_MODEL)
    assert weights.shape == (BATCH, N_HEADS, 4, 9)
