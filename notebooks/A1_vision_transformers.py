# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Activity 1 — Vision Transformers on solar cell defect detection
#
# **Covers rubric lines R2.2, R3.2, R3.3 and R3.6.**
# R2.2 — visualise attention weights as a heatmap for at least one example and interpret
# in writing what the model attends to.
# R3.2 — train the ViT from scratch on the chosen domain, produce attention maps from at
# least one head, identify the emergent regions in writing.
# R3.3 / R3.6 — fine-tune a pretrained ViT on the same domain with the head replaced, and
# compare it against the from-scratch model in a quantitative table.
#
# **Requirements.** Colab with the T4 GPU runtime. Measured runtime and peak GPU memory
# are printed by the final cell.
#
# ## The domain, and why this one
#
# Electroluminescence images of photovoltaic solar cells. A current is passed through the
# cell and the silicon emits infrared light; healthy material glows evenly, and cracks,
# broken contact fingers and inactive regions show up as dark lines and patches.
#
# Three reasons this domain and not another:
#
# 1. It is a real inspection problem from the energy sector, which the brief asks for.
# 2. **Defects are localised.** A crack is a thin dark line in one specific place. That
#    makes the attention maps in R2.2 and R3.2 interpretable — attention either lands on
#    the defect or it visibly does not, and either outcome is something to write about. On
#    a texture-classification domain the maps come out as an even smear and say nothing.
# 3. **It is small.** 2,624 images. A Vision Transformer has no built-in notion that
#    neighbouring pixels belong together; a convolution is handed that prior for free.
#    Training one from scratch on a few thousand images is therefore expected to
#    underperform, and measuring *how much* is the point of the comparison in R3.6.
#
# ## The labels, and where the line is drawn
#
# Each cell carries a defect probability that records annotator agreement, not a verdict:
# 0.0, 0.333, 0.667 or 1.0. A cell at 0.333 is one where roughly two experts in three
# said it was fine.
#
# The binary cut is at **0.5**, giving 1,803 functional against 821 defective. Cutting at
# "anything above zero" instead would fold all 295 disputed cells into the defective
# class, putting about a quarter of that class in dispute. A weak from-scratch result
# would then be unreadable: impossible to tell data scarcity from contradictory labels,
# and data scarcity is the thing being measured.
#
# Splits are stratified on **label and wafer type together**. Defect rates differ by wafer
# type — 34.4% of monocrystalline cells against 29.2% of polycrystalline — so a split
# balanced only on the label lets wafer type drift between the halves, and part of any
# measured difference between the two models becomes which wafers happened to land where.

# %%
# !pip install -q git+https://github.com/danimoreira90/cnn-transformers-clip.git

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from cv_mba.data import elpv_index, stratified_split
from cv_mba.determinism import enable_determinism
from cv_mba.imagesets import ImagePathDataset, evaluation_transform, training_transform
from cv_mba.metrics import accuracy, confusion_matrix, macro_f1, per_class_recall
from cv_mba.training import EpochRecord, predict, train_classifier
from cv_mba.vit import VisionTransformer

# %% [markdown]
# ## Configuration

# %%
SEEDS = [0, 1, 2]
IMAGE_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 8
SCRATCH_LR = 3e-4
FINETUNE_LR = 3e-5
TEST_FRACTION = 0.2
CLASS_NAMES = ("functional", "defective")
DATA_ROOT = Path("data/raw/elpv")

enable_determinism(SEEDS[0])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
STARTED_AT = time.time()

# %%
if not (DATA_ROOT / "src" / "elpv_dataset" / "data" / "labels.csv").is_file():
    # !git clone -q https://github.com/zae-bayern/elpv-dataset.git {DATA_ROOT}
    print("cloned")
else:
    print("already present, skipping clone")

# %% [markdown]
# ## The data

# %%
frame = elpv_index(DATA_ROOT)

print(f"cells: {len(frame)}")
print()
print("defect probability, which is annotator agreement:")
print(frame["defect_probability"].round(3).value_counts().sort_index().to_string())
print()
print("binary label at the 0.5 cut:")
print(frame["label"].map({0: "functional", 1: "defective"}).value_counts().to_string())
print()
print("defect rate by wafer type:")
print(frame.groupby("wafer_type")["label"].mean().round(4).to_string())

# %%
train_frame, val_frame = stratified_split(
    frame, by=["label", "wafer_type"], test_size=TEST_FRACTION, seed=SEEDS[0]
)
print(f"train {len(train_frame)}   validation {len(val_frame)}")
print()
print(
    pd.crosstab(train_frame["label"], train_frame["wafer_type"]).to_string(),
    "\n\nvalidation\n",
    pd.crosstab(val_frame["label"], val_frame["wafer_type"]).to_string(),
)

# %% [markdown]
# The cells are single-channel. Both models here take three channels, with the grey
# channel repeated, because the pretrained model was trained on three-channel photographs
# and the comparison has to be between two models that saw identical inputs.
#
# Only a horizontal flip is used. A solar cell has no intrinsic left or right, so
# mirroring it produces an image that could have come off the line. Rotation and colour
# jitter are left off: a rotated cell is not a cell the inspection system will ever see,
# and these are single-channel infrared images where a colour shift has no physical
# meaning.

# %%
def loaders_for(seed: int) -> tuple[DataLoader, DataLoader]:
    train_part, val_part = stratified_split(
        frame, by=["label", "wafer_type"], test_size=TEST_FRACTION, seed=seed
    )
    train = DataLoader(
        ImagePathDataset(train_part, training_transform(IMAGE_SIZE), channels=3),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    validation = DataLoader(
        ImagePathDataset(val_part, evaluation_transform(IMAGE_SIZE), channels=3),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    return train, validation


# %% [markdown]
# ## The two models
#
# **From scratch.** The Vision Transformer built in this package, at roughly the ViT-Tiny
# shape: 192 wide, six layers, three heads. A ViT-Base from scratch on 2,100 images would
# not fail interestingly, it would simply memorise; a small model keeps the comparison a
# statement about data rather than about parameter count.
#
# **Pretrained.** torchvision's ViT-B/16 with ImageNet weights and the classification head
# replaced by a two-class layer. The whole model is fine-tuned, at a learning rate ten
# times smaller than the from-scratch run, because the pretrained weights are already
# close to something useful and a large step would destroy what transfer learning is for.

# %%
def build_from_scratch() -> nn.Module:
    return VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        n_classes=len(CLASS_NAMES), d_model=192, n_heads=3, n_layers=6, dropout=0.1,
    )


def build_pretrained() -> nn.Module:
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
    return model


scratch_parameters = sum(p.numel() for p in build_from_scratch().parameters())
pretrained_parameters = sum(p.numel() for p in build_pretrained().parameters())
print(f"from scratch : {scratch_parameters:,} parameters")
print(f"pretrained   : {pretrained_parameters:,} parameters")

# %% [markdown]
# ## Training both, on three seeds each — R3.3
#
# Three seeds rather than one. A single run's difference between two models is partly the
# difference between two random initialisations, and the whole argument of R3.6 rests on
# that difference being real. The eval gate is deliberately strict: the pretrained model
# must win on **every** seed, not on average.

# %%
def run(build, learning_rate: float, seed: int, name: str) -> dict:
    enable_determinism(seed)
    train_loader, val_loader = loaders_for(seed)
    model = build().to(DEVICE)

    history = train_classifier(
        model=model, train_loader=train_loader, val_loader=val_loader,
        epochs=EPOCHS,
        optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05),
        device=DEVICE,
        on_epoch=lambda r: print(
            f"  {name} seed {seed} epoch {r.epoch}/{EPOCHS}  "
            f"train {r.train_loss:.4f}/{r.train_accuracy:.4f}  "
            f"val {r.val_loss:.4f}/{r.val_accuracy:.4f}"
        ),
    )

    y_true, y_pred = predict(model, val_loader, device=DEVICE)
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), n_classes=len(CLASS_NAMES))
    recall = per_class_recall(cm)

    return {
        "model": name, "seed": seed,
        "accuracy": accuracy(cm), "macro_f1": macro_f1(cm),
        "recall_functional": recall[0], "recall_defective": recall[1],
        "history": EpochRecord.to_frame(history), "confusion": cm,
        "trained_model": model,
    }


results = []
for seed in SEEDS:
    results.append(run(build_from_scratch, SCRATCH_LR, seed, "from scratch"))
for seed in SEEDS:
    results.append(run(build_pretrained, FINETUNE_LR, seed, "pretrained"))

# %% [markdown]
# ## The comparison table — R3.6

# %%
table = pd.DataFrame([
    {k: v for k, v in record.items() if k not in ("history", "confusion", "trained_model")}
    for record in results
])
print(table.round(4).to_string(index=False))
print()

summary = table.groupby("model")[["accuracy", "macro_f1", "recall_defective"]].agg(["mean", "std", "min"])
print(summary.round(4).to_string())
table.to_csv("a1_seed_results.csv", index=False)

# %%
scratch_f1 = table.query("model == 'from scratch'").sort_values("seed")["macro_f1"].to_numpy()
pretrained_f1 = table.query("model == 'pretrained'").sort_values("seed")["macro_f1"].to_numpy()
deltas = pretrained_f1 - scratch_f1

print("macro F1 by seed")
for seed, scratch, pretrained, delta in zip(SEEDS, scratch_f1, pretrained_f1, deltas):
    print(f"  seed {seed}:  from scratch {scratch:.4f}   pretrained {pretrained:.4f}   delta {delta:+.4f}")
print()
print(f"worst seed delta : {deltas.min():+.4f}   (eval C3 requires > 0)")
print(f"mean delta       : {deltas.mean():+.4f}   (eval C3 requires > 0.05)")
print()
print("EVAL C3:", "PASS" if deltas.min() > 0 and deltas.mean() > 0.05 else "FAIL")

# %% [markdown]
# ## Training curves for both models

# %%
figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for record in results:
    style = "-" if record["model"] == "pretrained" else "--"
    colour = "tab:blue" if record["model"] == "pretrained" else "tab:orange"
    label = f"{record['model']} seed {record['seed']}"
    axes[0].plot(record["history"]["epoch"], record["history"]["val_loss"],
                 style, color=colour, alpha=0.75, label=label)
    axes[1].plot(record["history"]["epoch"], record["history"]["val_accuracy"],
                 style, color=colour, alpha=0.75, label=label)

axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation loss"); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy"); axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=7, loc="lower right")
figure.suptitle("Validation curves, three seeds per model")
figure.tight_layout(); plt.show()

# %% [markdown]
# ## Attention maps — R2.2 and R3.2
#
# The classification token's row of the last attention layer is what gets drawn. That row
# is literally how much the model's summary vector looked at each patch, so overlaying it
# on the image shows where the decision came from.
#
# Two cases are shown deliberately: a cell the model got right, and a cell it got wrong.
# Showing only successes would make the interpretation unfalsifiable.

# %%
best_scratch = max(
    (r for r in results if r["model"] == "from scratch"), key=lambda r: r["macro_f1"]
)
model = best_scratch["trained_model"].eval()
_, attention_val_loader = loaders_for(best_scratch["seed"])
_, val_part = stratified_split(
    frame, by=["label", "wafer_type"], test_size=TEST_FRACTION, seed=best_scratch["seed"]
)
y_true, y_pred = predict(model, attention_val_loader, device=DEVICE)

correct_defective = np.flatnonzero((y_true.numpy() == 1) & (y_pred.numpy() == 1))
missed_defective = np.flatnonzero((y_true.numpy() == 1) & (y_pred.numpy() == 0))
print(f"defective cells found: {len(correct_defective)}   missed: {len(missed_defective)}")


# %%
def attention_over_image(position: int, caption: str, axis_pair) -> None:
    path = val_part["path"].iloc[position]
    with Image.open(path) as handle:
        picture = handle.convert("L")

    tensor, _ = ImagePathDataset(
        val_part.iloc[[position]], evaluation_transform(IMAGE_SIZE), channels=3
    )[0]

    with torch.no_grad():
        _, attentions = model(tensor.unsqueeze(0).to(DEVICE), return_attention=True)

    grid = IMAGE_SIZE // PATCH_SIZE
    cls_row = attentions[-1][0, :, 0, 1:].mean(dim=0)          # mean over heads
    head_row = attentions[-1][0, 0, 0, 1:]                     # head 0 alone

    for axis, row, title in (
        (axis_pair[0], None, f"{caption}\ncell image"),
        (axis_pair[1], cls_row, "attention, mean of 3 heads"),
        (axis_pair[2], head_row, "attention, head 0 alone"),
    ):
        axis.imshow(picture, cmap="gray")
        if row is not None:
            heat = row.reshape(1, 1, grid, grid)
            heat = F.interpolate(heat, size=picture.size[::-1], mode="bilinear")
            axis.imshow(heat[0, 0].cpu(), cmap="inferno", alpha=0.55)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([]); axis.set_yticks([])


def draw_row(candidates, caption: str, axis_row) -> None:
    """Draw one example if there is one, and say so plainly if there is not.

    A model that predicts a single class for everything leaves one of these lists empty.
    That is a result worth seeing stated on the figure, not an IndexError that stops the
    notebook halfway through.
    """
    if len(candidates):
        attention_over_image(int(candidates[0]), caption, axis_row)
        return
    for axis in axis_row:
        axis.axis("off")
    axis_row[0].set_title(f"no cells in this category: {caption}", fontsize=9)


figure, axes = plt.subplots(2, 3, figsize=(11, 7.5))
draw_row(correct_defective, "defective, correctly found", axes[0])
draw_row(missed_defective, "defective, missed", axes[1])
figure.tight_layout(); plt.show()

# %% [markdown]
# ## Per-class results for the best model of each kind

# %%
for name in ("from scratch", "pretrained"):
    best = max((r for r in results if r["model"] == name), key=lambda r: r["macro_f1"])
    cm = best["confusion"]
    print(f"===== {name}, seed {best['seed']} =====")
    print(pd.DataFrame({
        "class": CLASS_NAMES,
        "validation images": cm.sum(axis=1),
        "recall": per_class_recall(cm).round(4),
    }).to_string(index=False))
    print(f"  global accuracy {accuracy(cm):.4f}   macro F1 {macro_f1(cm):.4f}")
    print()

# %% [markdown]
# ## Measured cost

# %%
elapsed_minutes = (time.time() - STARTED_AT) / 60
peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0.0

print(f"RUNTIME   : {elapsed_minutes:.1f} minutes")
print(f"PEAK GPU  : {peak_mib:,.0f} MiB of 15,360 MiB available on a T4")
