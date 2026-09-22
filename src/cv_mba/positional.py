"""Positional encoding: the only thing in a transformer that knows about order.

Attention computes a weighted average over a *set* of tokens. Nothing inside it can see
that one token came before another — shuffle the input and every output comes back
shuffled the same way, computing the identical thing under different labels. Adding a
distinct vector to each position is what breaks that symmetry, and it is why a Vision
Transformer can tell that a crack runs across the middle of a cell rather than merely
that a crack is somewhere in the picture.

`test_attention_alone_cannot_tell_token_order` and
`test_positional_encoding_is_what_makes_order_matter` are that argument executed.

Two flavours, because the report has to compare them. The original Transformer used a
fixed sinusoidal table; the Vision Transformer learns one. Learned encodings cost one
vector per position and adapt to the data, which is why ViT uses them. Sinusoidal ones
cost nothing and extend to sequences longer than anything seen in training, which a
learned table cannot do.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine and cosine waves of geometrically increasing wavelength.

    Even channels hold `sin(pos / 10000^(2i/d))`, odd channels the matching cosine. Each
    position therefore gets a unique pattern, and because the wavelengths form a
    geometric series the relative offset between two positions is a fixed linear function
    of their encodings — a shift of `k` positions rotates each frequency pair by a
    constant angle, which the model can learn to detect.

    Holds no learnable parameters. The table is registered as a buffer so it travels with
    the model to the GPU and into a checkpoint without being trained.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        pair_index = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle = position / torch.pow(10_000.0, pair_index / d_model)

        table = torch.zeros(1, max_len, d_model)
        table[0, :, 0::2] = torch.sin(angle)
        table[0, :, 1::2] = torch.cos(angle)

        self.max_len = max_len
        self.register_buffer("table", table)

    def forward(self, x: Tensor) -> Tensor:
        """Add the encoding for positions 0..seq-1 to `x`."""
        _check_length(x, self.max_len)
        return x + self.table[:, : x.shape[1]]


class LearnedPositionalEncoding(nn.Module):
    """One trainable vector per position, which is what the Vision Transformer uses.

    The model discovers for itself what "position 17 of 197" should mean, rather than
    being handed a fixed pattern. On a fixed image size that is strictly more flexible.
    The cost is that the table cannot extend past `max_len`: there is no vector for a
    position never seen in training, which is why changing input resolution after
    training requires interpolating this table rather than simply running the model.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.max_len = max_len
        self.table = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add the learned encoding for positions 0..seq-1 to `x`."""
        _check_length(x, self.max_len)
        return x + self.table[:, : x.shape[1]]


def _check_length(x: Tensor, max_len: int) -> None:
    if x.shape[1] > max_len:
        raise ValueError(
            f"sequence of {x.shape[1]} exceeds max_len {max_len}; "
            f"there is no encoding for positions beyond the table"
        )
