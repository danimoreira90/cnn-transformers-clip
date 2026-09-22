"""Semantic retrieval with CLIP, with no training of any kind.

CLIP was trained to pull an image and its caption together in one shared space and push
mismatched pairs apart. That single objective is why it can be asked questions it was
never taught: a text description and an image become vectors in the same space, so
"which of these pictures is a car" is a dot product rather than a classifier.

This module holds the arithmetic around that idea. The parts that touch the model are
thin wrappers; the parts that carry reasoning — how presence is decided, how a ranking is
built — are pure functions with unit tests.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def l2_normalise(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale every row to unit length, leaving a zero row as zeros rather than nan."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between every row of `a` and every row of `b`.

    Once both sides are unit length, cosine similarity is just a dot product — which is
    why a search over thousands of images is a single matrix multiply.
    """
    return l2_normalise(_as_tensor(a)) @ l2_normalise(_as_tensor(b)).T


def top_matches(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The `k` highest scores and where they came from, best first."""
    scores = _as_tensor(scores)
    k = min(k, scores.shape[-1])
    values, indices = torch.topk(scores, k=k)
    return indices, values


def rank_concepts(
    similarity: Tensor | np.ndarray,
    concepts: Sequence[str],
    threshold: float,
    standardise: bool = True,
) -> pd.DataFrame:
    """Rank concepts by how often they are present across a corpus, and how strongly.

    `similarity` is one row per image, one column per concept. Returns `concept`,
    `mean_similarity`, `frequency` and `count`, sorted by frequency then mean similarity.

    **Why standardise by default.** CLIP's raw similarity values are not comparable
    between images. A busy, dark or cluttered photograph scores lower against every
    concept than a clean studio shot does, so a single absolute cut counts concepts in
    the bright pictures and silently misses them in the rest — the ranking then measures
    image quality as much as image content. Standardising each row turns the question
    into "which concepts stand out *within this image*", which is the question actually
    being asked. With `standardise=True` the threshold is read as a number of standard
    deviations above that image's own mean; with it off, as an absolute similarity.

    `test_an_absolute_threshold_misses_the_dim_image` is that argument measured.
    """
    similarity = _as_tensor(similarity)
    if similarity.shape[1] != len(concepts):
        raise ValueError(
            f"similarity has {similarity.shape[1]} columns but {len(concepts)} concepts "
            f"were given"
        )

    if standardise:
        centred = similarity - similarity.mean(dim=1, keepdim=True)
        scored = centred / similarity.std(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        scored = similarity

    present = scored > threshold
    counts = present.sum(dim=0)

    ranking = pd.DataFrame({
        "concept": list(concepts),
        "mean_similarity": similarity.mean(dim=0).tolist(),
        "frequency": (counts / similarity.shape[0]).tolist(),
        "count": counts.tolist(),
    })

    return ranking.sort_values(
        ["frequency", "mean_similarity"], ascending=False, ignore_index=True
    )


def features_of(output) -> Tensor:
    """Pull the embedding tensor out of whatever the model handed back.

    transformers 4.x returned a plain tensor from `get_image_features` and
    `get_text_features`. transformers 5.x returns a `BaseModelOutputWithPooling`, whose
    `pooler_output` holds the projected vector in the shared space. Colab currently ships
    5.16, so a wrapper written against the older shape fails there with an attribute
    error on a tensor method — which is exactly how this was found.

    Accepting both keeps the notebooks working across that boundary instead of pinning
    the project to whichever version happens to be installed today.
    """
    if isinstance(output, Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            f"cannot read embeddings from {type(output).__name__}; "
            f"expected a Tensor or an output carrying pooler_output"
        )
    return pooled


@torch.no_grad()
def embed_texts(texts: Sequence[str], model, processor, device: str = "cpu") -> Tensor:
    """Encode text descriptions into the shared space, unit length."""
    inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(device)
    return l2_normalise(features_of(model.get_text_features(**inputs))).cpu()


@torch.no_grad()
def embed_images(paths: Sequence[str], model, processor, device: str = "cpu",
                 batch_size: int = 64, on_batch=None) -> Tensor:
    """Encode images into the shared space, unit length, in batches.

    Images are opened one batch at a time rather than all at once: 2,697 decoded images
    would not fit comfortably beside the model on a T4.
    """
    from PIL import Image

    embeddings = []
    for start in range(0, len(paths), batch_size):
        chunk = list(paths[start : start + batch_size])
        images = []
        for path in chunk:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        embeddings.append(
            l2_normalise(features_of(model.get_image_features(**inputs))).cpu()
        )
        if on_batch is not None:
            on_batch(min(start + batch_size, len(paths)), len(paths))

    return torch.cat(embeddings)


def _as_tensor(x: Tensor | np.ndarray) -> Tensor:
    return x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float32)
