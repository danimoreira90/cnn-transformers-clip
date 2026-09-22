"""Differential check: our from-scratch attention against torch.nn.MultiheadAttention.

Lives in `evals/` rather than `tests/` for the same reason the sklearn check does. The
unit tests specify behaviour — shapes, masking, the scaling factor, head independence.
This file checks the arithmetic against an implementation nobody here wrote, by copying
our weights into torch's module and demanding the same numbers out.

The rubric forbids *using* torch's attention. It does not forbid proving ours agrees
with it, and a from-scratch implementation that quietly disagrees with the reference is
worth catching before it is trained for twenty minutes on a T4.

torch stores the three input projections stacked in one `in_proj_weight`, so the copy is
`cat([query, key, value])` in that order. Getting that order wrong is itself a good test:
the shapes all match and the numbers all differ.
"""

from __future__ import annotations

import pytest
import torch

from cv_mba.attention import MultiHeadAttention

CASES = [
    pytest.param(12, 3, 5, 2, id="3 heads"),
    pytest.param(16, 1, 7, 1, id="single head"),
    pytest.param(64, 8, 17, 3, id="vit sized"),
    pytest.param(8, 8, 4, 2, id="one dimension per head"),
]


def torch_reference(module: MultiHeadAttention) -> torch.nn.MultiheadAttention:
    """A torch MultiheadAttention carrying exactly our weights."""
    reference = torch.nn.MultiheadAttention(
        embed_dim=module.d_model, num_heads=module.n_heads, batch_first=True
    )
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat(
                [
                    module.query_projection.weight,
                    module.key_projection.weight,
                    module.value_projection.weight,
                ]
            )
        )
        reference.in_proj_bias.copy_(
            torch.cat(
                [
                    module.query_projection.bias,
                    module.key_projection.bias,
                    module.value_projection.bias,
                ]
            )
        )
        reference.out_proj.weight.copy_(module.output_projection.weight)
        reference.out_proj.bias.copy_(module.output_projection.bias)
    return reference.eval()


@pytest.mark.parametrize("d_model,n_heads,seq,batch", CASES)
def test_self_attention_matches_torch(d_model: int, n_heads: int, seq: int, batch: int) -> None:
    torch.manual_seed(0)
    ours = MultiHeadAttention(d_model=d_model, n_heads=n_heads).eval()
    theirs = torch_reference(ours)
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        our_output, our_weights = ours(x, x, x)
        their_output, their_weights = theirs(x, x, x, average_attn_weights=False)

    assert torch.allclose(our_output, their_output, atol=1e-5)
    assert torch.allclose(our_weights, their_weights, atol=1e-5)


@pytest.mark.parametrize("d_model,n_heads,seq,batch", CASES)
def test_cross_attention_matches_torch(d_model: int, n_heads: int, seq: int, batch: int) -> None:
    torch.manual_seed(1)
    ours = MultiHeadAttention(d_model=d_model, n_heads=n_heads).eval()
    theirs = torch_reference(ours)
    query = torch.randn(batch, seq, d_model)
    memory = torch.randn(batch, seq + 3, d_model)

    with torch.no_grad():
        our_output, our_weights = ours(query, memory, memory)
        their_output, their_weights = theirs(query, memory, memory, average_attn_weights=False)

    assert torch.allclose(our_output, their_output, atol=1e-5)
    assert torch.allclose(our_weights, their_weights, atol=1e-5)


def test_masked_attention_matches_torch() -> None:
    torch.manual_seed(2)
    ours = MultiHeadAttention(d_model=12, n_heads=3).eval()
    theirs = torch_reference(ours)
    x = torch.randn(2, 5, 12)

    blocked = torch.zeros(2, 5, dtype=torch.bool)
    blocked[:, 2] = True  # torch marks positions to IGNORE
    keep = ~blocked[:, None, None, :]  # we mark positions to KEEP

    with torch.no_grad():
        our_output, our_weights = ours(x, x, x, mask=keep)
        their_output, their_weights = theirs(
            x, x, x, key_padding_mask=blocked, average_attn_weights=False
        )

    assert torch.all(our_weights[..., 2] == 0.0)
    assert torch.allclose(our_output, their_output, atol=1e-5)
    assert torch.allclose(our_weights, their_weights, atol=1e-5)
