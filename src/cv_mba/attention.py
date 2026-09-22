"""Attention, written from scratch as testable PyTorch modules.

Rubric 2.1 asks for scaled dot-product attention and multi-head attention built from
nothing, with independent projections per head and the head outputs concatenated. Nothing
here calls `torch.nn.MultiheadAttention` or `F.scaled_dot_product_attention`; using those
would answer a different question than the one being asked.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Carries no learnable parameters: it is pure arithmetic. Every weight a transformer
    learns sits in the projections around it.

    Operates on the last two dimensions and broadcasts over whatever comes before, so the
    same module serves a plain `(batch, seq, d_k)` caller and multi-head attention's
    `(batch, heads, seq, d_head)` without special-casing either.

    Why the division by sqrt(d_k): the dot product of two `d_k`-dimensional vectors grows
    with `d_k`, so without it the scores reaching softmax get larger as the model gets
    wider. Softmax then saturates, almost all weight lands on one key, and the gradient
    through the rest goes to nearly zero. The model still runs and still produces a valid
    probability distribution, which is why only a numeric test catches its absence.
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the attended values and the attention weights.

        `mask` is broadcast against the score matrix; positions that are False or 0 are
        excluded and come back with exactly zero weight.
        """
        d_k = query.shape[-1]
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return weights @ value, weights


class MultiHeadAttention(nn.Module):
    """Several attention heads over the same sequence, their outputs concatenated.

    One head can only average one way. Multiple heads let the model attend to different
    relationships at once — in a Vision Transformer, one head may track a crack running
    across a cell while another tracks the regular grid of busbars.

    **Independent projections per head.** Each head gets its own query, key and value
    projection. They are stored as one `d_model x d_model` matrix per role and split into
    head-sized slices, which is arithmetically identical to holding `n_heads` separate
    `d_model x d_head` layers: row block `h` of the matrix touches only head `h`, and no
    other head reads it. The compact form is one matrix multiply instead of `n_heads`,
    which matters on a T4. `test_the_compact_form_equals_running_each_head_separately`
    proves the equivalence against a literal per-head implementation rather than leaving
    it as a claim in this docstring.

    Attention weights come back per head, shaped `(batch, heads, queries, keys)`, because
    rubric lines 2.2 and 3.2 need to draw individual heads. Averaging them inside the
    module would destroy exactly what those lines ask you to interpret.
    """

    def __init__(self, d_model: int, n_heads: int, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not divisible by n_heads {n_heads}; "
                f"heads would have to share dimensions"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model, bias=bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x: Tensor) -> Tensor:
        """(batch, seq, d_model) -> (batch, heads, seq, d_head)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

    def _join_heads(self, x: Tensor) -> Tensor:
        """(batch, heads, seq, d_head) -> (batch, seq, d_model). The concatenation."""
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the attended sequence and the per-head attention weights.

        Query, key and value may differ in length, so the same module handles a sequence
        attending to itself and a sequence attending to another one.
        """
        projected_query = self._split_heads(self.query_projection(query))
        projected_key = self._split_heads(self.key_projection(key))
        projected_value = self._split_heads(self.value_projection(value))

        attended, weights = self.attention(
            projected_query, projected_key, projected_value, mask=mask
        )

        return self.output_projection(self._join_heads(attended)), weights
