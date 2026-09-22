"""Unit tests for the conditional GAN.

A GAN can fail in ways no shape check catches, so these test the things that are
checkable: that conditioning actually conditions, that the two losses respond in opposite
directions, and that the diversity measure used to detect mode collapse can tell a
collapsed batch from a varied one.
"""

from __future__ import annotations

import pytest
import torch

from cv_mba.gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    diversity_score,
    discriminator_loss,
    generator_loss,
)

LATENT, CLASSES, SIZE, BATCH = 16, 3, 64, 4


@pytest.fixture
def generator() -> ConditionalGenerator:
    torch.manual_seed(0)
    return ConditionalGenerator(latent_dim=LATENT, n_classes=CLASSES, image_size=SIZE)


@pytest.fixture
def discriminator() -> ConditionalDiscriminator:
    torch.manual_seed(0)
    return ConditionalDiscriminator(n_classes=CLASSES, image_size=SIZE)


def test_the_generator_produces_one_image_per_noise_vector(generator) -> None:
    z = torch.randn(BATCH, LATENT)
    labels = torch.randint(0, CLASSES, (BATCH,))

    images = generator(z, labels)

    assert images.shape == (BATCH, 1, SIZE, SIZE)


def test_generated_pixels_stay_in_the_tanh_range(generator) -> None:
    # Real images are scaled to [-1, 1] to match, so a generator that escaped this range
    # would be solving a different problem than the discriminator is judging.
    images = generator(torch.randn(BATCH, LATENT), torch.zeros(BATCH, dtype=torch.long))

    assert images.min() >= -1.0 and images.max() <= 1.0


def test_the_class_label_changes_what_is_generated(generator) -> None:
    # The whole point of a conditional GAN. If the label is ignored, asking for more
    # COVID images gives back the same distribution as asking for anything else, and the
    # augmentation in R5.4 adds nothing that was scarce.
    z = torch.randn(BATCH, LATENT)

    as_class_0 = generator(z, torch.zeros(BATCH, dtype=torch.long))
    as_class_2 = generator(z, torch.full((BATCH,), 2, dtype=torch.long))

    assert not torch.allclose(as_class_0, as_class_2, atol=1e-4)


def test_the_same_noise_and_label_give_the_same_image(generator) -> None:
    generator.eval()
    z = torch.randn(BATCH, LATENT)
    labels = torch.ones(BATCH, dtype=torch.long)

    with torch.no_grad():
        assert torch.equal(generator(z, labels), generator(z, labels))


def test_the_discriminator_scores_one_number_per_image(discriminator) -> None:
    images = torch.randn(BATCH, 1, SIZE, SIZE)
    labels = torch.randint(0, CLASSES, (BATCH,))

    assert discriminator(images, labels).shape == (BATCH,)


def test_the_discriminator_accepts_what_the_generator_makes(generator, discriminator) -> None:
    labels = torch.randint(0, CLASSES, (BATCH,))

    scores = discriminator(generator(torch.randn(BATCH, LATENT), labels), labels)

    assert scores.shape == (BATCH,)
    assert torch.isfinite(scores).all()


def test_the_discriminator_label_matters(discriminator) -> None:
    images = torch.randn(BATCH, 1, SIZE, SIZE)

    as_class_0 = discriminator(images, torch.zeros(BATCH, dtype=torch.long))
    as_class_2 = discriminator(images, torch.full((BATCH,), 2, dtype=torch.long))

    assert not torch.allclose(as_class_0, as_class_2, atol=1e-4)


def test_the_two_losses_pull_in_opposite_directions() -> None:
    # The adversarial relationship, checked directly. A discriminator score that makes
    # the generator happy must make the discriminator unhappy, and the reverse.
    confident_fake_is_real = torch.tensor([5.0, 5.0, 5.0, 5.0])
    confident_fake_is_fake = torch.tensor([-5.0, -5.0, -5.0, -5.0])
    real_scores = torch.tensor([5.0, 5.0, 5.0, 5.0])

    generator_when_fooling = generator_loss(confident_fake_is_real)
    generator_when_caught = generator_loss(confident_fake_is_fake)
    discriminator_when_fooled = discriminator_loss(real_scores, confident_fake_is_real)
    discriminator_when_right = discriminator_loss(real_scores, confident_fake_is_fake)

    assert generator_when_fooling < generator_when_caught
    assert discriminator_when_right < discriminator_when_fooled


def test_label_smoothing_keeps_the_discriminator_from_certainty() -> None:
    real_scores = torch.tensor([10.0, 10.0])
    fake_scores = torch.tensor([-10.0, -10.0])

    plain = discriminator_loss(real_scores, fake_scores, real_label=1.0)
    smoothed = discriminator_loss(real_scores, fake_scores, real_label=0.9)

    assert smoothed > plain


# --------------------------------------------------------------------------------------
# Detecting mode collapse
# --------------------------------------------------------------------------------------

def test_identical_samples_score_no_diversity() -> None:
    # Mode collapse in its purest form: every sample the same. This is the number the
    # notebook watches across epochs.
    one_image = torch.randn(1, 1, SIZE, SIZE)
    collapsed = one_image.repeat(8, 1, 1, 1)

    assert diversity_score(collapsed) == pytest.approx(0.0, abs=1e-6)


def test_varied_samples_score_above_zero() -> None:
    varied = torch.randn(8, 1, SIZE, SIZE)

    assert diversity_score(varied) > 0.5


def test_partial_collapse_scores_between_the_two() -> None:
    one_image = torch.randn(1, 1, SIZE, SIZE)
    half_collapsed = torch.cat([one_image.repeat(6, 1, 1, 1), torch.randn(2, 1, SIZE, SIZE)])

    score = diversity_score(half_collapsed)

    assert 0.0 < score < diversity_score(torch.randn(8, 1, SIZE, SIZE))


def test_diversity_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        diversity_score(torch.randn(1, 1, SIZE, SIZE))


def test_the_minibatch_standard_deviation_channel_can_be_switched_on() -> None:
    # The mitigation applied in R5.3. With it, the discriminator can see how varied the
    # batch is, so a generator emitting one image is detectable rather than invisible.
    plain = ConditionalDiscriminator(n_classes=CLASSES, image_size=SIZE)
    aware = ConditionalDiscriminator(n_classes=CLASSES, image_size=SIZE, minibatch_stddev=True)
    images = torch.randn(BATCH, 1, SIZE, SIZE)
    labels = torch.zeros(BATCH, dtype=torch.long)

    assert plain(images, labels).shape == aware(images, labels).shape
    assert sum(p.numel() for p in aware.parameters()) > sum(p.numel() for p in plain.parameters())


def test_a_batch_aware_discriminator_separates_collapsed_from_varied() -> None:
    # The mechanism, checked. A plain discriminator judges each image alone and cannot
    # tell eight copies from eight different pictures; a batch-aware one can.
    torch.manual_seed(0)
    aware = ConditionalDiscriminator(n_classes=CLASSES, image_size=SIZE,
                                     minibatch_stddev=True).eval()
    one_image = torch.randn(1, 1, SIZE, SIZE)
    collapsed = one_image.repeat(8, 1, 1, 1)
    varied = torch.randn(8, 1, SIZE, SIZE)
    labels = torch.zeros(8, dtype=torch.long)

    with torch.no_grad():
        assert not torch.allclose(
            aware(collapsed, labels).mean(), aware(varied, labels).mean(), atol=1e-3
        )


def test_the_label_survives_normalisation_in_every_mode(generator) -> None:
    # The regression that motivated conditional batch normalisation. Ordinary BatchNorm
    # subtracts the batch mean, and when every sample carries the same label that mean is
    # the label's whole contribution — so the conditioning vanished in training mode with
    # a uniform batch, which is exactly how a batch of COVID images would be requested.
    z = torch.randn(BATCH, LATENT)
    class_0 = torch.zeros(BATCH, dtype=torch.long)
    class_2 = torch.full((BATCH,), 2, dtype=torch.long)

    generator.train()
    assert not torch.allclose(generator(z, class_0), generator(z, class_2), atol=1e-4)

    generator.eval()
    with torch.no_grad():
        assert not torch.allclose(generator(z, class_0), generator(z, class_2), atol=1e-4)

    generator.train()
    mixed_a = torch.tensor([0, 1, 2, 0])
    mixed_b = torch.tensor([2, 0, 1, 2])
    assert not torch.allclose(generator(z, mixed_a), generator(z, mixed_b), atol=1e-4)
