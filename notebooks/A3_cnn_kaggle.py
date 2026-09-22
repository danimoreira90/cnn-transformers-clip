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
# # Activity 3 — Transfer learning with a pretrained CNN
#
# Seven object classes, 1,803 images, one training run with the backbone frozen.
#
# **Covers rubric lines R1.1 and R1.2.**
# R1.1 — load a pretrained CNN, replace the classification head with the dataset's class
# count, train with the backbone frozen.
# R1.2 — document training curves, report per-class and global accuracy.
#
# **Requirements.** Colab with the T4 GPU runtime selected. Measured runtime and peak GPU
# memory are printed by the final cell of this notebook and recorded in the report; they
# are not estimated here.
#
# **Data.** `pavansanagapati/images-dataset` from Kaggle, downloaded by this notebook.
# Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab's Secrets panel (the key icon in the
# left sidebar) and grant this notebook access to both.

# %%
# The package carries every piece of real logic, so this notebook stays readable and the
# logic stays unit tested. Installing from the repository pulls the version on `main`.
# !pip install -q git+https://github.com/danimoreira90/cnn-transformers-clip.git

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from cv_mba.data import A3_CLASSES, a3_index, stratified_split
from cv_mba.determinism import enable_determinism
from cv_mba.imagesets import ImagePathDataset, evaluation_transform, training_transform
from cv_mba.metrics import accuracy, confusion_matrix, macro_f1, per_class_recall
from cv_mba.training import EpochRecord, predict, train_classifier

# %% [markdown]
# ## Configuration
#
# Every number the run depends on, in one place, so the report can state what was used
# without anyone reading the code to find out.

# %%
SEED = 0
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
TEST_FRACTION = 0.2
DATA_ROOT = Path("data/raw/a3")

enable_determinism(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
STARTED_AT = time.time()

# %% [markdown]
# ## Fetching the data
#
# Credentials come from Colab's Secrets panel rather than a pasted key, so nothing
# sensitive is ever written into the notebook or into version control.

# %%
if "google.colab" in sys.modules:
    from google.colab import userdata

    try:
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    except Exception as error:
        raise RuntimeError(
            "This notebook cannot read your Kaggle credentials.\n"
            "\n"
            "  1. Click the key icon in the left sidebar (Secrets).\n"
            "  2. Add two secrets, named exactly KAGGLE_USERNAME and KAGGLE_KEY.\n"
            "     Both values are in your kaggle.json, or from kaggle.com ->\n"
            "     Settings -> API -> Create New Token.\n"
            "  3. Turn the Notebook access toggle ON for BOTH secrets. A secret that\n"
            "     exists but is not shared with this notebook fails the same way.\n"
            "  4. Re-run this cell.\n"
            "\n"
            f"Colab reported: {type(error).__name__}: {error}"
        ) from error

    print("using Kaggle credentials from Colab Secrets")
else:
    print("not on Colab; using the local ~/.kaggle/kaggle.json")

# %%
if not (DATA_ROOT / "data" / "cats").is_dir():
    # !pip install -q kaggle
    # !kaggle datasets download -d pavansanagapati/images-dataset -p {DATA_ROOT} --unzip
    print("downloaded")
else:
    print("already present, skipping download")

# %% [markdown]
# ## What the archive actually contains
#
# The archive ships `data/` and, inside it, `data/data/` — a byte-for-byte copy of all
# seven class folders. A loader that recurses therefore sees every image twice and
# invents an eighth class named `data`. Split that 80/20 and the same picture lands in
# training and in validation, so validation accuracy stops measuring generalisation and
# starts measuring memorisation.
#
# The cell below counts both so the difference is on the record rather than in a comment.

# %%
recursive_count = sum(
    1 for path in DATA_ROOT.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)
frame = a3_index(DATA_ROOT / "data")

print(f"images a recursive loader would find : {recursive_count}")
print(f"images this notebook uses            : {len(frame)}")
print(f"classes                              : {len(A3_CLASSES)}  {A3_CLASSES}")
print()
print(frame["class_name"].value_counts().sort_index().to_string())

# %% [markdown]
# ## Splitting
#
# Stratified on the class, so every class keeps its share in both halves. The largest
# class (cars, 420) is twice the size of the smallest (cats, dogs, horses and human at
# 202), and an unstratified draw at this ratio can shift a small class's validation share
# by several percentage points from run to run.

# %%
train_frame, val_frame = stratified_split(
    frame, by=["label"], test_size=TEST_FRACTION, seed=SEED
)
print(f"train {len(train_frame)}   validation {len(val_frame)}")
print()
print(
    pd.DataFrame({
        "train": train_frame["class_name"].value_counts(),
        "validation": val_frame["class_name"].value_counts(),
    }).sort_index().to_string()
)

# %% [markdown]
# ## Loaders
#
# Training images are flipped horizontally at random; validation images are not touched.
# Augmenting the validation set would measure a slightly different problem every epoch
# and make its curve incomparable with itself.
#
# Both pipelines normalise with the ImageNet channel statistics. The backbone learned its
# filters on inputs centred that way, and feeding it raw 0-to-1 pixels shifts every
# activation away from what those filters expect.

# %%
train_loader = DataLoader(
    ImagePathDataset(train_frame, training_transform(IMAGE_SIZE)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
)
val_loader = DataLoader(
    ImagePathDataset(val_frame, evaluation_transform(IMAGE_SIZE)),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
)
print(f"{len(train_loader)} training batches, {len(val_loader)} validation batches")

# %% [markdown]
# ## The model — R1.1
#
# ResNet-50 with ImageNet weights, every backbone parameter frozen, and the 1000-class
# ImageNet head replaced by a fresh 7-class layer.
#
# The choice is argued in the report against the T4's 15,360 MiB and this dataset's seven
# classes. In short: with the backbone frozen there are no activations to keep for the
# backward pass through it, so memory is dominated by the forward pass at batch 32 and
# neither ResNet-50 nor a smaller backbone comes close to the limit. That makes feature
# quality the deciding factor rather than memory, and ResNet-50's 2,048-dimensional
# pooled features carry more than EfficientNet-B0's 1,280 for a linear probe to separate.

# %%
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(A3_CLASSES))
model = model.to(DEVICE)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"total parameters     : {sum(p.numel() for p in model.parameters()):,}")
print(f"trainable parameters : {sum(p.numel() for p in trainable):,}")
print(f"frozen               : {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

# %% [markdown]
# Only the new head is trainable. That is what makes this feature extraction rather than
# fine-tuning: the pretrained features are used exactly as they are, and the only thing
# learned is how to combine them into seven decisions.

# %% [markdown]
# ## Training — one run

# %%
history = train_classifier(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    optimizer=torch.optim.Adam(trainable, lr=LEARNING_RATE),
    device=DEVICE,
    on_epoch=lambda r: print(
        f"epoch {r.epoch:2d}  "
        f"train loss {r.train_loss:.4f} acc {r.train_accuracy:.4f}   "
        f"val loss {r.val_loss:.4f} acc {r.val_accuracy:.4f}"
    ),
)
curves = EpochRecord.to_frame(history)

# %% [markdown]
# ## Curves — R1.2

# %%
figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))

loss_axis.plot(curves["epoch"], curves["train_loss"], marker="o", label="train")
loss_axis.plot(curves["epoch"], curves["val_loss"], marker="o", label="validation")
loss_axis.set_xlabel("epoch"); loss_axis.set_ylabel("cross-entropy loss")
loss_axis.set_title("Loss per epoch"); loss_axis.legend(); loss_axis.grid(alpha=0.3)

accuracy_axis.plot(curves["epoch"], curves["train_accuracy"], marker="o", label="train")
accuracy_axis.plot(curves["epoch"], curves["val_accuracy"], marker="o", label="validation")
accuracy_axis.set_xlabel("epoch"); accuracy_axis.set_ylabel("accuracy")
accuracy_axis.set_title("Accuracy per epoch"); accuracy_axis.legend(); accuracy_axis.grid(alpha=0.3)

figure.tight_layout()
plt.show()

curves.to_csv("a3_training_curves.csv", index=False)
print(curves.to_string(index=False))

# %% [markdown]
# ## Per-class and global accuracy — R1.2
#
# Global accuracy alone would hide a class the model never predicts, which is the exact
# failure Activity 4.1 asks us to diagnose in someone else's project. Per-class recall is
# reported beside it, and macro-F1 gives a small class the same weight as a large one.

# %%
y_true, y_pred = predict(model, val_loader, device=DEVICE)
cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), n_classes=len(A3_CLASSES))

per_class = pd.DataFrame({
    "class": A3_CLASSES,
    "validation images": cm.sum(axis=1),
    "accuracy (recall)": per_class_recall(cm).round(4),
})
print(per_class.to_string(index=False))
print()
print(f"global accuracy : {accuracy(cm):.4f}")
print(f"macro F1        : {macro_f1(cm):.4f}")

per_class.to_csv("a3_per_class_accuracy.csv", index=False)

# %%
figure, axis = plt.subplots(figsize=(6.5, 5.5))
image = axis.imshow(cm, cmap="Blues")
axis.set_xticks(range(len(A3_CLASSES)), A3_CLASSES, rotation=45, ha="right")
axis.set_yticks(range(len(A3_CLASSES)), A3_CLASSES)
axis.set_xlabel("predicted"); axis.set_ylabel("true")
axis.set_title("Confusion matrix, validation split")
for row in range(len(A3_CLASSES)):
    for column in range(len(A3_CLASSES)):
        axis.text(column, row, int(cm[row, column]), ha="center", va="center",
                  color="white" if cm[row, column] > cm.max() / 2 else "black")
figure.colorbar(image, ax=axis)
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Measured cost
#
# These two numbers go into the notebook header and into the report. They are measured
# here rather than estimated, because an estimated figure in a requirements header is an
# unverified claim.

# %%
elapsed_minutes = (time.time() - STARTED_AT) / 60
peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0.0

print(f"RUNTIME   : {elapsed_minutes:.1f} minutes")
print(f"PEAK GPU  : {peak_mib:,.0f} MiB of 15,360 MiB available on a T4")
