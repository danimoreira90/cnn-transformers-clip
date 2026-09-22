"""Unit tests for the transformer encoder block.

Rubric 2.3 asks for a complete block: two-layer feedforward, LayerNorm, residual
connections, used as the base of the ViT. The tests below pin the composition order as
well as the pieces, because a block with all the right parts wired in the wrong order
still runs and still trains — just worse, and silently.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cv_mba.encoder import TransformerEncoderBlock

D_MODEL, N_HEADS, SEQ, BATCH = 12, 3, 5, 2


@pytest.fixture
def block() -> TransformerEncoderBlock:
    torch.manual_seed(0)
    return TransformerEncoderBlock(d_model=D_MODEL, n_heads=N_HEADS).eval()


def silence_both_sublayers(block: TransformerEncoderBlock) -> None:
    """Force both sublayers to output exactly zero, leaving only the residual paths."""
    with torch.no_grad():
        block.attention.output_projection.weight.zero_()
        block.attention.output_projection.bias.zero_()
        block.feedforward[-1].weight.zero_()
        block.feedforward[-1].bias.zero_()


def test_a_block_keeps_the_shape_it_was_given(block) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output, _ = block(x)

    assert output.shape == x.shape


def test_with_both_sublayers_silenced_the_input_comes_back_untouched(block) -> None:
    # The proof that the residual connections are wired, not merely present in the file.
    # If either skip were missing, a silenced block would return zeros or a normalised
    # version of the input instead of the input itself.
    x = torch.randn(BATCH, SEQ, D_MODEL)
    silence_both_sublayers(block)

    output, _ = block(x)

    assert torch.allclose(output, x, atol=1e-6)


def test_normalisation_happens_before_each_sublayer_not_after(block) -> None:
    # Pre-norm against post-norm, pinned by arithmetic. Both orders produce the right
    # shape and both train; only one of them trains a deep stack from scratch reliably.
    x = torch.randn(BATCH, SEQ, D_MODEL)

    with torch.no_grad():
        normed = block.attention_norm(x)
        attended, _ = block.attention(normed, normed, normed)
        after_attention = x + attended
        expected = after_attention + block.feedforward(block.feedforward_norm(after_attention))
        output, _ = block(x)

    assert torch.allclose(output, expected, atol=1e-6)


def test_the_feedforward_has_exactly_two_linear_layers(block) -> None:
    linears = [layer for layer in block.feedforward if isinstance(layer, nn.Linear)]

    assert len(linears) == 2


def test_the_feedforward_widens_then_narrows_back(block) -> None:
    first, second = [layer for layer in block.feedforward if isinstance(layer, nn.Linear)]

    assert first.in_features == D_MODEL
    assert first.out_features == second.in_features == 4 * D_MODEL
    assert second.out_features == D_MODEL


def test_layer_norm_gives_each_token_zero_mean_and_unit_variance(block) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL) * 50 + 17

    normed = block.attention_norm(x)

    assert torch.allclose(normed.mean(dim=-1), torch.zeros(BATCH, SEQ), atol=1e-5)
    assert torch.allclose(normed.std(dim=-1, unbiased=False), torch.ones(BATCH, SEQ), atol=1e-4)


def test_attention_weights_survive_the_block(block) -> None:
    # Rubric 2.2 and 3.2 need per-head weights out of a trained model. A block that
    # swallows them makes those lines impossible to answer later.
    x = torch.randn(BATCH, SEQ, D_MODEL)

    _, weights = block(x)

    assert weights.shape == (BATCH, N_HEADS, SEQ, SEQ)


def test_a_mask_reaches_the_attention_inside(block) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)
    mask = torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool)
    mask[..., 3] = False

    _, weights = block(x, mask=mask)

    assert torch.all(weights[..., 3] == 0.0)


def test_in_eval_mode_the_same_input_gives_the_same_output(block) -> None:
    # Dropout off means the block is a function, not a sample. Every eval in this project
    # depends on that.
    x = torch.randn(BATCH, SEQ, D_MODEL)
    block_with_dropout = TransformerEncoderBlock(D_MODEL, N_HEADS, dropout=0.5).eval()

    first, _ = block_with_dropout(x)
    second, _ = block_with_dropout(x)

    assert torch.allclose(first, second)


def test_blocks_stack_without_adaptation(block) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)
    stack = [TransformerEncoderBlock(D_MODEL, N_HEADS).eval() for _ in range(3)]

    for layer in stack:
        x, _ = layer(x)

    assert x.shape == (BATCH, SEQ, D_MODEL)
    assert torch.isfinite(x).all()


def test_a_custom_feedforward_width_is_respected() -> None:
    block = TransformerEncoderBlock(D_MODEL, N_HEADS, d_ff=7)

    first = [layer for layer in block.feedforward if isinstance(layer, nn.Linear)][0]

    assert first.out_features == 7
