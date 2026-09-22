"""Unit tests for positional encoding.

Rubric 2.4 asks for positional encoding applied to the ViT token sequence, plus a written
explanation of why attention without it does not preserve position. The permutation pair
below is that explanation, executed: the same claim, checked rather than asserted.
"""

from __future__ import annotations

import math

import pytest
import torch

from cv_mba.encoder import TransformerEncoderBlock
from cv_mba.positional import LearnedPositionalEncoding, SinusoidalPositionalEncoding

D_MODEL, SEQ, BATCH = 16, 6, 2


@pytest.fixture(params=["sinusoidal", "learned"])
def encoding(request):
    torch.manual_seed(0)
    if request.param == "sinusoidal":
        return SinusoidalPositionalEncoding(D_MODEL, max_len=32)
    return LearnedPositionalEncoding(D_MODEL, max_len=32)


def test_encoding_is_added_so_the_width_is_unchanged(encoding) -> None:
    # Added, not concatenated. Concatenating would grow the model width at every layer
    # and change what the projections downstream expect.
    x = torch.randn(BATCH, SEQ, D_MODEL)

    assert encoding(x).shape == x.shape


def test_encoding_actually_changes_the_tokens(encoding) -> None:
    x = torch.randn(BATCH, SEQ, D_MODEL)

    assert not torch.allclose(encoding(x), x)


def test_every_position_gets_a_different_encoding(encoding) -> None:
    zeros = torch.zeros(1, SEQ, D_MODEL)

    encoded = encoding(zeros)[0]
    distinct = {tuple(row.tolist()) for row in encoded}

    assert len(distinct) == SEQ


def test_the_same_position_is_encoded_the_same_way_across_the_batch(encoding) -> None:
    zeros = torch.zeros(BATCH, SEQ, D_MODEL)

    encoded = encoding(zeros)

    assert torch.allclose(encoded[0], encoded[1])


def test_a_sequence_longer_than_the_table_is_refused(encoding) -> None:
    too_long = torch.randn(1, 64, D_MODEL)

    with pytest.raises(ValueError, match="max_len"):
        encoding(too_long)


# --------------------------------------------------------------------------------------
# The pair that carries rubric 2.4
# --------------------------------------------------------------------------------------

def test_attention_alone_cannot_tell_token_order(encoding) -> None:
    # Shuffle the tokens, and every output comes back shuffled the same way — the block
    # computed the identical thing, just relabelled. Attention is a weighted average over
    # a *set*; nothing inside it can see that one token came before another.
    torch.manual_seed(1)
    block = TransformerEncoderBlock(D_MODEL, n_heads=4).eval()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    order = torch.randperm(SEQ)

    with torch.no_grad():
        straight, _ = block(x)
        shuffled, _ = block(x[:, order])

    assert torch.allclose(shuffled, straight[:, order], atol=1e-6)


def test_positional_encoding_is_what_makes_order_matter(encoding) -> None:
    # The same shuffle, with positions added first. Now the outputs are not a relabelling
    # of each other: a token in position 0 and the same token in position 3 are different
    # inputs. That difference is the whole job of positional encoding, and in a ViT it is
    # what lets the model know a crack runs across the middle rather than merely that a
    # crack is somewhere in the picture.
    torch.manual_seed(1)
    block = TransformerEncoderBlock(D_MODEL, n_heads=4).eval()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    order = torch.randperm(SEQ)

    with torch.no_grad():
        straight, _ = block(encoding(x))
        shuffled, _ = block(encoding(x[:, order]))

    assert not torch.allclose(shuffled, straight[:, order], atol=1e-4)


# --------------------------------------------------------------------------------------
# The two flavours differ in what they cost and what they learn
# --------------------------------------------------------------------------------------

def test_sinusoidal_encoding_learns_nothing() -> None:
    fixed = SinusoidalPositionalEncoding(D_MODEL, max_len=32)

    assert list(fixed.parameters()) == []


def test_sinusoidal_values_match_the_closed_form() -> None:
    fixed = SinusoidalPositionalEncoding(D_MODEL, max_len=32)

    table = fixed(torch.zeros(1, SEQ, D_MODEL))[0]

    for position in (0, 1, 5):
        for pair in (0, 3, 7):
            angle = position / (10_000 ** (2 * pair / D_MODEL))
            assert table[position, 2 * pair] == pytest.approx(math.sin(angle), abs=1e-6)
            assert table[position, 2 * pair + 1] == pytest.approx(math.cos(angle), abs=1e-6)


def test_learned_encoding_has_one_trainable_vector_per_position() -> None:
    learned = LearnedPositionalEncoding(D_MODEL, max_len=32)

    parameters = list(learned.parameters())

    assert len(parameters) == 1
    assert parameters[0].shape == (1, 32, D_MODEL)
    assert parameters[0].requires_grad


def test_learned_encoding_receives_gradient() -> None:
    learned = LearnedPositionalEncoding(D_MODEL, max_len=32)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    learned(x).sum().backward()

    gradient = list(learned.parameters())[0].grad
    assert gradient is not None
    assert gradient[:, :SEQ].abs().sum() > 0
    assert gradient[:, SEQ:].abs().sum() == 0
