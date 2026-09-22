"""A conditional GAN for chest X-rays, and the tools to tell when it is failing.

Activity 4.1 asks for synthetic images of the scarce COVID class, so the generator has to
be told which class to produce. That is what "conditional" means here: the class label is
fed to the generator alongside the noise, and to the discriminator alongside the image.
Without it, asking for more COVID images returns the overall distribution — mostly
Normal, since that is most of the training data — and the augmentation adds nothing that
was scarce in the first place.

The failure this module is built to expose is **mode collapse**: the generator discovers
one image the discriminator happens to accept and emits it forever. Losses look
unremarkable while it happens. `diversity_score` is the number that makes it visible, and
`minibatch_stddev` on the discriminator is the mitigation, with a mechanism simple enough
to state in one sentence — it lets the discriminator see how varied the batch is, so a
generator emitting one picture becomes detectable rather than invisible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConditionalBatchNorm2d(nn.Module):
    """Batch normalisation whose scale and shift are chosen by the class label.

    This exists because of a bug worth describing, since the same shape appears in a lot
    of conditional GAN code.

    The obvious design concatenates the label embedding to the noise vector and lets
    ordinary BatchNorm follow. It works during training and fails silently at exactly the
    moment it is needed. BatchNorm subtracts the batch mean feature by feature. When every
    sample in a batch carries the *same* label — which is what happens when you ask for a
    batch of 120 COVID images — the label's entire contribution *is* that batch mean, so
    normalisation removes it and the generator produces unconditional output. No error, no
    warning, and the augmentation in R5.4 quietly adds images of the wrong distribution.

    Making the label supply the normalisation parameters removes the failure by
    construction: the conditioning cannot be normalised away, because it is applied after
    normalisation rather than carried through it.
    """

    def __init__(self, num_features: int, n_classes: int) -> None:
        super().__init__()
        self.normalise = nn.BatchNorm2d(num_features, affine=False)
        self.scale = nn.Embedding(n_classes, num_features)
        self.shift = nn.Embedding(n_classes, num_features)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        normalised = self.normalise(x)
        scale = self.scale(labels).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(labels).unsqueeze(-1).unsqueeze(-1)
        return normalised * scale + shift


class UpsampleBlock(nn.Module):
    """Double the resolution, then normalise conditionally on the class."""

    def __init__(self, in_channels: int, out_channels: int, n_classes: int) -> None:
        super().__init__()
        self.convolution = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.normalise = ConditionalBatchNorm2d(out_channels, n_classes)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        return self.activate(self.normalise(self.convolution(x), labels))


class ConditionalGenerator(nn.Module):
    """Noise plus a class label in, one greyscale image out.

    The label enters twice, deliberately. It is concatenated to the noise so it is present
    from the first layer, and it selects the normalisation parameters at every upsampling
    stage so it cannot be normalised away. See ConditionalBatchNorm2d for why the second
    route is not redundant.

    Output passes through tanh into [-1, 1], matching how the real images are scaled.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_classes: int = 3,
        image_size: int = 64,
        features: int = 64,
        label_dim: int = 32,
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be a multiple of 16, got {image_size}")

        self.latent_dim = latent_dim
        self.start = image_size // 16
        self.features = features
        self.label_embedding = nn.Embedding(n_classes, label_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim + label_dim, features * 8 * self.start * self.start),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.ModuleList([
            UpsampleBlock(features * 8, features * 4, n_classes),
            UpsampleBlock(features * 4, features * 2, n_classes),
            UpsampleBlock(features * 2, features, n_classes),
        ])
        self.to_image = nn.Sequential(
            nn.ConvTranspose2d(features, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        conditioned = torch.cat([noise, self.label_embedding(labels)], dim=1)
        x = self.project(conditioned).view(-1, self.features * 8, self.start, self.start)
        for block in self.upsample:
            x = block(x, labels)
        return self.to_image(x)


class ConditionalDiscriminator(nn.Module):
    """One image plus a class label in, one realism score out.

    The label is broadcast to a full extra image channel rather than appended at the end,
    so every convolution sees it. A label injected only at the final layer arrives after
    all the spatial reasoning is finished and is largely ignored.

    `minibatch_stddev` is the mode-collapse mitigation. When on, the standard deviation
    across the batch is computed for every feature, averaged, and appended as one more
    channel before the final scoring layer. A batch of identical images makes that channel
    zero, which is a signature the discriminator can learn to punish. Without it the
    discriminator judges each image in isolation and literally cannot tell eight copies
    from eight different pictures.
    """

    def __init__(
        self,
        n_classes: int = 3,
        image_size: int = 64,
        features: int = 64,
        minibatch_stddev: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.minibatch_stddev = minibatch_stddev
        self.label_embedding = nn.Embedding(n_classes, image_size * image_size)

        self.trunk = nn.Sequential(
            _down(2, features, normalise=False),
            _down(features, features * 2),
            _down(features * 2, features * 4),
            _down(features * 4, features * 8),
        )
        final_channels = features * 8 + (1 if minibatch_stddev else 0)
        self.score = nn.Conv2d(final_channels, 1, kernel_size=image_size // 16)

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        label_map = self.label_embedding(labels).view(-1, 1, self.image_size, self.image_size)
        features = self.trunk(torch.cat([images, label_map], dim=1))

        if self.minibatch_stddev:
            spread = features.std(dim=0, unbiased=False).mean()
            channel = spread.expand(features.shape[0], 1, *features.shape[2:])
            features = torch.cat([features, channel], dim=1)

        return self.score(features).flatten(1).mean(dim=1)


def generator_loss(fake_scores: Tensor) -> Tensor:
    """How badly the generator failed to fool the discriminator.

    Non-saturating form: the generator maximises the chance its fakes are called real,
    rather than minimising the chance they are called fake. The two are equivalent at the
    optimum but not in the early epochs, where the saturating version gives almost no
    gradient exactly when the generator is worst and needs it most.
    """
    return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))


def discriminator_loss(
    real_scores: Tensor,
    fake_scores: Tensor,
    real_label: float = 1.0,
) -> Tensor:
    """How badly the discriminator told real from generated.

    `real_label` below 1.0 is one-sided label smoothing: the discriminator is asked to be
    90% sure rather than certain. A discriminator that becomes certain stops producing
    useful gradient for the generator, and the training stalls with both losses looking
    calm.
    """
    real_target = torch.full_like(real_scores, real_label)
    on_real = F.binary_cross_entropy_with_logits(real_scores, real_target)
    on_fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    return on_real + on_fake


def diversity_score(samples: Tensor) -> float:
    """Mean pairwise distance between samples, scaled so it is readable across runs.

    This is the mode-collapse detector. A generator emitting one image scores 0; a healthy
    one scores well above it. Watching this across epochs turns "the samples look the
    same" into a curve that can be shown before and after a mitigation, which is what
    rubric 5.3 asks for as evidence of improvement.
    """
    if samples.shape[0] < 2:
        raise ValueError(f"need at least 2 samples to measure spread, got {samples.shape[0]}")

    flat = samples.flatten(1)
    distances = torch.cdist(flat, flat)
    count = flat.shape[0]
    mean_distance = distances.sum() / (count * (count - 1))

    return float(mean_distance / (flat.shape[1] ** 0.5))


def _down(in_channels: int, out_channels: int, normalise: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    ]
    if normalise:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)
