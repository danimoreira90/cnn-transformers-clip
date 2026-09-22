"""Unit tests for the Vision Transformer.

Rubric 3.1: patch embedding, a learnable CLS token and positional encoding, producing a
complete ViT that takes an image and returns classification logits. Rubric 3.2 needs the
attention maps out of it afterwards, so those are tested here too.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cv_mba.vit import PatchEmbedding, VisionTransformer

IMAGE, PATCH, CHANNELS, D_MODEL, HEADS, LAYERS, CLASSES = 32, 8, 1, 24, 4, 3, 2
PATCHES = (IMAGE // PATCH) ** 2


@pytest.fixture
def vit() -> VisionTransformer:
    torch.manual_seed(0)
    return VisionTransformer(
        image_size=IMAGE, patch_size=PATCH, in_channels=CHANNELS, n_classes=CLASSES,
        d_model=D_MODEL, n_heads=HEADS, n_layers=LAYERS,
    ).eval()


# --------------------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------------------

def test_an_image_becomes_one_token_per_patch() -> None:
    embedding = PatchEmbedding(IMAGE, PATCH, CHANNELS, D_MODEL)
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    tokens = embedding(x)

    assert tokens.shape == (2, PATCHES, D_MODEL)
    assert PATCHES == 16


def test_patches_do_not_overlap(vit) -> None:
    # Change one patch of the image and exactly one token changes. If the patches
    # overlapped, or the stride were wrong, neighbouring tokens would move too.
    embedding = PatchEmbedding(IMAGE, PATCH, CHANNELS, D_MODEL).eval()
    x = torch.randn(1, CHANNELS, IMAGE, IMAGE)
    disturbed = x.clone()
    disturbed[0, :, :PATCH, :PATCH] += 10.0

    with torch.no_grad():
        before, after = embedding(x), embedding(disturbed)

    changed = (~torch.isclose(before, after, atol=1e-6)).any(dim=-1)[0]
    assert changed.sum() == 1
    assert changed[0]


def test_an_image_size_not_divisible_by_the_patch_size_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PatchEmbedding(image_size=30, patch_size=8, in_channels=1, d_model=D_MODEL)


def test_patch_embedding_handles_one_channel_and_three() -> None:
    for channels in (1, 3):
        embedding = PatchEmbedding(IMAGE, PATCH, channels, D_MODEL)
        tokens = embedding(torch.randn(2, channels, IMAGE, IMAGE))
        assert tokens.shape == (2, PATCHES, D_MODEL)


def test_an_image_of_the_wrong_size_is_refused() -> None:
    embedding = PatchEmbedding(IMAGE, PATCH, CHANNELS, D_MODEL)

    with pytest.raises(ValueError, match="expected"):
        embedding(torch.randn(1, CHANNELS, IMAGE + 8, IMAGE))


# --------------------------------------------------------------------------------------
# The CLS token
# --------------------------------------------------------------------------------------

def test_the_sequence_is_one_longer_than_the_patch_count(vit) -> None:
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    tokens, _ = vit.forward_features(x)

    assert tokens.shape == (2, PATCHES + 1, D_MODEL)


def test_the_cls_token_sits_at_index_zero_and_is_learnable(vit) -> None:
    named = dict(vit.named_parameters())

    assert "cls_token" in named
    assert named["cls_token"].shape == (1, 1, D_MODEL)
    assert named["cls_token"].requires_grad


def test_the_cls_token_starts_identical_for_every_image_in_a_batch(vit) -> None:
    # One learned vector, broadcast. Before any attention runs it carries no information
    # about the image; everything it ends up holding it gathered from the patches.
    x = torch.randn(4, CHANNELS, IMAGE, IMAGE)

    prepared = vit.prepare_tokens(x)

    assert torch.allclose(prepared[0, 0], prepared[3, 0])


def test_the_classifier_reads_only_the_cls_token(vit) -> None:
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    with torch.no_grad():
        logits = vit(x)
        tokens, _ = vit.forward_features(x)
        by_hand = vit.head(vit.final_norm(tokens[:, 0]))

    assert torch.allclose(logits, by_hand, atol=1e-6)


def test_the_cls_token_receives_gradient(vit) -> None:
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    vit(x).sum().backward()

    assert vit.cls_token.grad is not None
    assert vit.cls_token.grad.abs().sum() > 0


# --------------------------------------------------------------------------------------
# Classification and attention maps
# --------------------------------------------------------------------------------------

def test_an_image_goes_in_and_class_logits_come_out(vit) -> None:
    x = torch.randn(3, CHANNELS, IMAGE, IMAGE)

    logits = vit(x)

    assert logits.shape == (3, CLASSES)
    assert torch.isfinite(logits).all()


def test_attention_comes_back_one_map_per_layer_per_head(vit) -> None:
    # Rubric 3.2 asks for attention maps from at least one head of a trained model.
    # A model that discards them makes that line unanswerable after training.
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    logits, attentions = vit(x, return_attention=True)

    assert logits.shape == (2, CLASSES)
    assert len(attentions) == LAYERS
    for layer in attentions:
        assert layer.shape == (2, HEADS, PATCHES + 1, PATCHES + 1)


def test_the_cls_row_of_attention_is_a_distribution_over_patches(vit) -> None:
    # This row is what gets drawn over the image: how much the classification token
    # looked at each patch. It has to be a proper distribution or the heatmap is a lie.
    x = torch.randn(1, CHANNELS, IMAGE, IMAGE)

    _, attentions = vit(x, return_attention=True)
    cls_row = attentions[-1][0, :, 0, :]

    assert torch.allclose(cls_row.sum(dim=-1), torch.ones(HEADS), atol=1e-6)
    assert (cls_row >= 0).all()


def test_positional_encoding_covers_the_patches_and_the_cls_token(vit) -> None:
    assert vit.positional.max_len >= PATCHES + 1


def test_the_two_positional_flavours_are_both_selectable() -> None:
    learned = VisionTransformer(IMAGE, PATCH, CHANNELS, CLASSES, D_MODEL, HEADS, 1,
                                positional="learned")
    fixed = VisionTransformer(IMAGE, PATCH, CHANNELS, CLASSES, D_MODEL, HEADS, 1,
                              positional="sinusoidal")

    assert any(name == "positional.table" for name, _ in learned.named_parameters())
    assert not any(name.startswith("positional.") for name, _ in fixed.named_parameters())


def test_an_unknown_positional_flavour_is_refused() -> None:
    with pytest.raises(ValueError, match="positional"):
        VisionTransformer(IMAGE, PATCH, CHANNELS, CLASSES, D_MODEL, HEADS, 1,
                          positional="rotary")


def test_a_deeper_model_has_more_blocks(vit) -> None:
    shallow = VisionTransformer(IMAGE, PATCH, CHANNELS, CLASSES, D_MODEL, HEADS, 2)

    assert len(vit.blocks) == LAYERS
    assert len(shallow.blocks) == 2


def test_the_head_can_be_replaced_for_a_new_class_count(vit) -> None:
    # Rubric 3.3 fine-tunes a pretrained ViT by swapping the head. The same move has to
    # work on ours, or the comparison is between two different kinds of object.
    vit.head = nn.Linear(D_MODEL, 7)

    logits = vit(torch.randn(2, CHANNELS, IMAGE, IMAGE))

    assert logits.shape == (2, 7)


def test_in_eval_mode_the_model_is_a_function_not_a_sample() -> None:
    model = VisionTransformer(IMAGE, PATCH, CHANNELS, CLASSES, D_MODEL, HEADS, 2,
                              dropout=0.5).eval()
    x = torch.randn(2, CHANNELS, IMAGE, IMAGE)

    assert torch.allclose(model(x), model(x))
