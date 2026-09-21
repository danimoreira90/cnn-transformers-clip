"""The 26 rubric lines this project is marked against, and a check that each one is backed.

The marking sheet is the contract. Every rubric line must point at a real artefact that
contains a literal anchor string, so a section silently dropped during a refactor fails
loudly instead of quietly costing a mark.

Anchors should be substantive rather than decorative: prefer a table caption or a heading
that only exists while the work exists, over a bare marker that survives deleting the
content it was meant to guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

RUBRIC: dict[str, str] = {
    # 1. Adapting models to new image domains with pretrained CNNs
    "R1.1": (
        "Loaded a pretrained CNN, correctly replaced the classification head with the "
        "dataset's class count, and performed feature extraction with a frozen backbone."
    ),
    "R1.2": "Documented training curves and reported per-class and global accuracy.",
    "R1.3": (
        "Proposed at least three augmentation strategies, each justified for the "
        "dataset's domain."
    ),
    "R1.4": (
        "Discussed in writing when to use feature extraction versus fine-tuning, based "
        "on dataset size and domain distance."
    ),
    "R1.5": (
        "Justified the choice of pretrained model for the domain, considering Colab T4 "
        "capacity and the number of classes."
    ),
    # 2. Building Transformer architectures from attention to BERT fine-tuning
    "R2.1": (
        "Implemented scaled dot-product attention and multi-head attention from scratch "
        "as testable PyTorch modules, with independent projections per attention head "
        "and concatenation of the head outputs."
    ),
    "R2.2": (
        "Visualised attention weights as a heatmap for at least one example from the "
        "domain and interpreted in writing what the model attends to."
    ),
    "R2.3": (
        "Built a complete TransformerEncoderBlock with a two-layer feedforward, "
        "LayerNorm and residual connections, used as the base of the ViT."
    ),
    "R2.4": (
        "Applied positional encoding to the ViT token sequence and explained why "
        "attention without positional encoding does not preserve position."
    ),
    "R2.5": (
        "Analysed in writing the differences between BERT pretraining and ViT "
        "pretraining, identifying what each strategy maximises."
    ),
    # 3. Designing Vision Transformers for image classification
    "R3.1": (
        "Implemented patch embedding, a learnable CLS token and positional encoding, "
        "producing a complete ViT that takes an image and returns classification logits."
    ),
    "R3.2": (
        "Trained the ViT from scratch on the chosen domain, generated attention maps "
        "from at least one head and identified emergent regions in writing."
    ),
    "R3.3": (
        "Fine-tuned a pretrained ViT on the same domain, replaced the classification "
        "head and compared it with the from-scratch ViT in a quantitative table."
    ),
    "R3.4": (
        "Analysed DeiT and Swin Transformer, explaining what each one solves that the "
        "original ViT does not."
    ),
    "R3.5": (
        "Justified in writing when Vision Transformers beat CNNs and when CNNs are "
        "preferable, based on the chosen domain."
    ),
    "R3.6": (
        "Built a comparison table of ViT from scratch versus pretrained ViT and "
        "justified the architecture choice for the domain from the data."
    ),
    # 4. Zero-shot classification and semantic search with CLIP
    "R4.1": (
        "Analysed the alignment of visual and textual representations in CLIP, "
        "explaining why contrastive pretraining enables semantic retrieval with no "
        "supervised training."
    ),
    "R4.2": (
        "Implemented ranking of objects by semantic frequency over the ADS-16 corpus "
        "with at least 20 descriptions, and visualised the 5 most frequent with example "
        "images."
    ),
    "R4.3": (
        "Implemented semantic search by text with at least 8 queries varying in "
        "specificity, documenting and analysing the results."
    ),
    "R4.4": (
        "Compared CLIP's textual query mechanism with BERT's sequence tokenisation, "
        "explaining the role of padding and the attention mask."
    ),
    # 5. Building GANs for conditional synthesis and domain translation
    "R5.1": (
        "Diagnosed at least five technical problems in the X-ray project, each with its "
        "expected clinical impact."
    ),
    "R5.2": (
        "Implemented a GAN, conditional or CycleGAN, generating medical-domain images "
        "with a correct adversarial training loop."
    ),
    "R5.3": (
        "Diagnosed training instability, mode collapse or divergence, and applied at "
        "least one mitigation with evidence of improvement."
    ),
    "R5.4": (
        "Evaluated the impact of synthetic augmentation on COVID-19 class recall by "
        "comparing training with and without the generated images."
    ),
    "R5.5": (
        "Identified at least four problems in the traffic analysis project and explained "
        "the risks of ImageNet transfer learning for flow classification."
    ),
    "R5.6": (
        "Delivered an integrated improvement plan for the X-ray project covering model, "
        "metric, synthetic augmentation and a clinical adoption criterion."
    ),
}


class RubricError(Exception):
    """Raised when the evidence file itself is malformed."""


@dataclass(frozen=True)
class Evidence:
    """A claim that `artifact` contains `anchor`, and so satisfies one rubric line."""

    artifact: str
    anchor: str


@dataclass(frozen=True)
class Gap:
    """One rubric line with nothing standing behind it."""

    rubric_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.rubric_id}  {self.reason}"


def load_evidence(path: Path) -> dict[str, Evidence]:
    """Read the evidence file, returning only the lines that have been recorded.

    A key with no value is a rubric line not yet earned, which is expected for most of
    the project and is not an error. An unknown rubric id is an error, because it is
    almost always a typo that would otherwise hide a real gap.
    """
    if not path.is_file():
        raise RubricError(f"evidence file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RubricError(f"evidence file must be a mapping, got {type(raw).__name__}")

    recorded: dict[str, Evidence] = {}
    for rubric_id, entry in raw.items():
        if rubric_id not in RUBRIC:
            raise RubricError(f"unknown rubric id in evidence file: {rubric_id}")
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise RubricError(f"{rubric_id}: entry must be a mapping")
        artifact = entry.get("artifact")
        anchor = entry.get("anchor")
        if not artifact:
            raise RubricError(f"{rubric_id}: entry has no artifact")
        if not anchor:
            raise RubricError(f"{rubric_id}: entry has no anchor")
        recorded[rubric_id] = Evidence(artifact=str(artifact), anchor=str(anchor))

    return recorded


def missing_evidence(
    evidence: Mapping[str, Evidence],
    repo_root: Path,
    rubric: Mapping[str, str] = RUBRIC,
) -> list[Gap]:
    """Return every rubric line that is unrecorded, unbacked, or points at the wrong file.

    Gaps come back in rubric order so the failure output reads like the marking sheet.
    """
    gaps: list[Gap] = []

    for rubric_id in rubric:
        claim = evidence.get(rubric_id)
        if claim is None:
            gaps.append(Gap(rubric_id, "no evidence recorded"))
            continue

        artifact_path = repo_root / claim.artifact
        if not artifact_path.is_file():
            gaps.append(Gap(rubric_id, f"artifact not found: {claim.artifact}"))
            continue

        try:
            text = artifact_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            gaps.append(Gap(rubric_id, f"artifact is not utf-8 text: {claim.artifact}"))
            continue

        if claim.anchor not in text:
            gaps.append(
                Gap(rubric_id, f"anchor {claim.anchor!r} absent from {claim.artifact}")
            )

    return gaps
