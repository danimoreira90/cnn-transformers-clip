"""The Vision Transformer, assembled from the attention and encoder modules in this package.

The idea is simple and slightly outrageous: cut an image into a grid of square patches,
flatten each one into a vector, and hand the resulting sequence to a transformer as if the
patches were words. No convolution sees the whole picture, no pooling pyramid, no
hand-designed notion of locality. Whatever spatial structure the model uses, it learns.

That is also why it needs far more data than a CNN. A convolution is handed the prior that
nearby pixels belong together; a transformer has to discover it. Activity 1 measures
exactly that cost on roughly 2,100 solar cell images.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cv_mba.encoder import TransformerEncoderBlock
from cv_mba.positional import LearnedPositionalEncoding, SinusoidalPositionalEncoding

POSITIONAL_ENCODINGS = {
    "learned": LearnedPositionalEncoding,
    "sinusoidal": SinusoidalPositionalEncoding,
}


class PatchEmbedding(nn.Module):
    """Cut an image into non-overlapping patches and project each one to a token.

    Implemented as a convolution whose kernel and stride both equal the patch size. That
    is not a shortcut around the linear projection the paper describes — it *is* that
    projection. A kernel the size of its own stride touches each patch exactly once and
    never straddles two, so the convolution computes one independent linear map per patch,
    which is the definition. `test_patches_do_not_overlap` checks that claim by disturbing
    one patch and counting how many tokens move.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image size {image_size} is not divisible by patch size {patch_size}; "
                f"the last patch would be cut short"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """(batch, channels, H, W) -> (batch, n_patches, d_model)."""
        _, _, height, width = x.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, got {height}x{width}"
            )

        return self.projection(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """Patch embedding, a learnable CLS token, positional encoding, then encoder blocks.

    **The CLS token.** A single learned vector pushed onto the front of the patch
    sequence. It starts out carrying nothing about the image — the same vector for every
    picture — and everything it ends up holding it gathered from the patches through
    attention. The classifier reads only that token. This is what lets one fixed-size
    vector summarise a variable amount of visual evidence without pooling, and it is why
    the CLS row of the last attention layer is the natural thing to draw over an image:
    it is literally how much the model's summary looked at each patch.

    **Defaults are deliberately small.** `d_model=192`, six layers, three heads — roughly
    the "ViT-Tiny" shape. Training a ViT-Base from scratch on a few thousand images would
    not fail interestingly; it would simply memorise. A small model makes the comparison
    against a pretrained ViT in rubric 3.3 a statement about data, not about parameter
    count.

    Attention weights are returned per layer when asked, which rubric 3.2 needs after
    training.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        n_classes: int = 2,
        d_model: int = 192,
        n_heads: int = 3,
        n_layers: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.0,
        positional: str = "learned",
    ) -> None:
        super().__init__()
        if positional not in POSITIONAL_ENCODINGS:
            raise ValueError(
                f"unknown positional encoding {positional!r}; "
                f"choose one of {sorted(POSITIONAL_ENCODINGS)}"
            )

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.positional = POSITIONAL_ENCODINGS[positional](
            d_model, max_len=self.patch_embedding.n_patches + 1
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        """Patches to tokens, CLS token prepended, positions added."""
        patches = self.patch_embedding(x)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        return self.embedding_dropout(self.positional(torch.cat([cls, patches], dim=1)))

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the encoder stack, returning the token sequence and each layer's attention."""
        tokens = self.prepare_tokens(x)
        attentions: list[Tensor] = []
        for block in self.blocks:
            tokens, weights = block(tokens)
            attentions.append(weights)
        return tokens, attentions

    def forward(self, x: Tensor, return_attention: bool = False):
        """Classify a batch of images, optionally returning every layer's attention."""
        tokens, attentions = self.forward_features(x)
        logits = self.head(self.final_norm(tokens[:, 0]))
        return (logits, attentions) if return_attention else logits
