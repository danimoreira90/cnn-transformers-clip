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
# # Activity 4.1 — COVID-19 screening on chest X-rays
#
# A previous group closed this project with results marked "promising". The clinical team
# said the model "ignores positive cases". Both statements were true at the same time,
# and this notebook reproduces the setup that made that possible before trying to fix it.
#
# **Covers rubric lines R5.1, R5.2, R5.3 and R5.4.**
# R5.1 — diagnose at least five technical problems, each with its clinical impact.
# R5.2 — implement a GAN for the medical domain with a correct adversarial training loop.
# R5.3 — diagnose training instability, apply a mitigation, show evidence it improved.
# R5.4 — measure the effect of synthetic augmentation on COVID-class recall, comparing
# training with and without the generated images.
#
# **Requirements.** Colab with the T4 GPU runtime. Measured runtime and peak GPU memory
# are printed by the final cell.
#
# **Data.** `tawsifurrahman/covid19-radiography-database` from Kaggle, 780 MB. Set
# `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab's Secrets panel.
#
# ## Why reproduce the failure first
#
# It would be easy to build a good classifier and declare the problem solved. That would
# answer nothing. Every number later in this notebook is a comparison, and a comparison
# needs something measured to compare against — so the original 1,200-image split, the
# original architecture and the original training settings are rebuilt exactly as the
# brief describes them, and the failure is measured rather than described.

# %%
# !pip install -q git+https://github.com/danimoreira90/cnn-transformers-clip.git

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from cv_mba.data import COVID_CLASSES, FAILED_PROJECT_COUNTS, covid_subsample, stratified_split
from cv_mba.determinism import enable_determinism
from cv_mba.gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    discriminator_loss,
    diversity_score,
    generator_loss,
)
from cv_mba.imagesets import ImagePathDataset, evaluation_transform, training_transform
from cv_mba.metrics import accuracy, confusion_matrix, macro_f1, per_class_recall
from cv_mba.training import EpochRecord, predict, train_classifier

# %% [markdown]
# ## Configuration
#
# The first block reproduces the failed project. The second is this notebook's own work.

# %%
# --- exactly as the previous group had it ---
ORIGINAL_EPOCHS = 15
ORIGINAL_LR = 0.01
ORIGINAL_TEST_FRACTION = 0.2

# --- this notebook ---
SEEDS = [0, 1, 2]
IMAGE_SIZE = 64
BATCH_SIZE = 64
GAN_EPOCHS = 60
GAN_LATENT = 100
GAN_LR = 2e-4
SYNTHETIC_PER_CLASS = 480      # brings COVID from 120 up to Normal's 600 in training
CLASSIFIER_EPOCHS = 15
DATA_ROOT = Path("data/raw/covid/COVID-19_Radiography_Dataset")
SYNTHETIC_ROOT = Path("synthetic_covid")

enable_determinism(SEEDS[0])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
STARTED_AT = time.time()

# %%
if "google.colab" in sys.modules:
    from google.colab import userdata

    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    print("using Kaggle credentials from Colab Secrets")
else:
    print("not on Colab; using the local ~/.kaggle/kaggle.json")

# %%
if not (DATA_ROOT / "COVID").is_dir():
    # !pip install -q kaggle
    # !kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p data/raw/covid --unzip
    print("downloaded")
else:
    print("already present, skipping download")

# %% [markdown]
# ## Rebuilding the original 1,200 images
#
# 840 Normal, 240 Pneumonia, 120 COVID. The archive holds far more of each — 10,192
# Normal, 1,345 Pneumonia and 3,616 COVID — so the shortage the previous group worked
# around was a choice, not a constraint. That is the first thing worth noticing.

# %%
subsample = covid_subsample(DATA_ROOT, seed=SEEDS[0])
print(subsample["class_name"].value_counts().reindex(COVID_CLASSES).to_string())
print(f"\ntotal: {len(subsample)}   requested: {FAILED_PROJECT_COUNTS}")
print(f"imbalance, Normal to COVID: {840 / 120:.0f} to 1")

# %% [markdown]
# ### The split, unstratified, exactly as described
#
# An 80/20 draw with no stratification. With only 120 COVID images, the share that lands
# in validation is left to chance, and the cell below measures how far it drifts across
# ten different draws.

# %%
def unstratified_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - ORIGINAL_TEST_FRACTION))
    return shuffled.iloc[:cut], shuffled.iloc[cut:]


drift = []
for seed in range(10):
    _, validation = unstratified_split(subsample, seed)
    counts = validation["class_name"].value_counts()
    drift.append({name: int(counts.get(name, 0)) for name in COVID_CLASSES})

drift_frame = pd.DataFrame(drift)
print(drift_frame.to_string())
print()
print(f"COVID images in validation ranged from {drift_frame['COVID'].min()} to "
      f"{drift_frame['COVID'].max()} across ten unstratified draws")
print(f"every single validation COVID case is worth "
      f"{100 / drift_frame['COVID'].mean():.1f} percentage points of recall")

# %% [markdown]
# ## Reproducing the failed model
#
# ResNet-18 with no pretrained weights, SGD, a fixed learning rate of 0.01, fifteen
# epochs, no augmentation. Evaluated on global accuracy, as reported.

# %%
original_train, original_val = unstratified_split(subsample, SEEDS[0])
name_to_label = {name: index for index, name in enumerate(COVID_CLASSES)}


def loader_for(frame: pd.DataFrame, training: bool, batch: int = BATCH_SIZE) -> DataLoader:
    transform = (training_transform(IMAGE_SIZE, horizontal_flip=False)
                 if training else evaluation_transform(IMAGE_SIZE))
    return DataLoader(
        ImagePathDataset(frame, transform, channels=3),
        batch_size=batch, shuffle=training, num_workers=2,
    )


def build_original_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COVID_CLASSES))
    return model


enable_determinism(SEEDS[0])
original_model = build_original_model().to(DEVICE)
original_history = train_classifier(
    model=original_model,
    train_loader=loader_for(original_train, training=True),
    val_loader=loader_for(original_val, training=False),
    epochs=ORIGINAL_EPOCHS,
    optimizer=torch.optim.SGD(original_model.parameters(), lr=ORIGINAL_LR),
    device=DEVICE,
    on_epoch=lambda r: print(
        f"  epoch {r.epoch:2d}/{ORIGINAL_EPOCHS}  train acc {r.train_accuracy:.4f}  "
        f"val acc {r.val_accuracy:.4f}"
    ),
)

# %% [markdown]
# ### What they reported

# %%
final = original_history[-1]
print(f"training accuracy   : {final.train_accuracy:.1%}")
print(f"validation accuracy : {final.val_accuracy:.1%}")
print()
print("Reported as promising. On these two numbers alone, it is not obviously wrong.")

# %% [markdown]
# ### What global accuracy was hiding

# %%
y_true, y_pred = predict(original_model, loader_for(original_val, training=False), device=DEVICE)
original_cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), n_classes=len(COVID_CLASSES))
original_recall = per_class_recall(original_cm)

print(pd.DataFrame({
    "class": COVID_CLASSES,
    "validation cases": original_cm.sum(axis=1),
    "recall": original_recall.round(4),
    "caught": original_cm.diagonal(),
}).to_string(index=False))
print()
print(f"global accuracy : {accuracy(original_cm):.4f}")
print(f"macro F1        : {macro_f1(original_cm):.4f}")
print()
print(f"COVID recall    : {original_recall[2]:.1%}  <- what the clinical team was reacting to")

# %%
figure, axis = plt.subplots(figsize=(5.5, 4.5))
image = axis.imshow(original_cm, cmap="Reds")
axis.set_xticks(range(3), COVID_CLASSES, rotation=20, ha="right")
axis.set_yticks(range(3), COVID_CLASSES)
axis.set_xlabel("predicted"); axis.set_ylabel("true")
axis.set_title("The reproduced baseline")
for row in range(3):
    for column in range(3):
        axis.text(column, row, int(original_cm[row, column]), ha="center", va="center",
                  color="white" if original_cm[row, column] > original_cm.max() / 2 else "black")
figure.colorbar(image, ax=axis); figure.tight_layout(); plt.show()

# %% [markdown]
# ## R5.1 — five technical problems, and what each one costs a patient
#
# **1. Untreated class imbalance, 7 to 1 against the class that matters.** 840 Normal
# against 120 COVID, with no class weighting, no resampling and no weighted loss.
# Answering "Normal" to everything scores 70% accuracy, so gradient descent is rewarded
# for learning the prior instead of the disease.
# *Clinically:* the model's failures concentrate entirely in the class where a miss sends
# an infectious patient home untreated and uncontained.
#
# **2. Global accuracy as the only metric.** The measured figures above show a global
# accuracy that looks acceptable beside a COVID recall that is not. One number cannot
# report a three-class problem where the classes matter unequally.
# *Clinically:* the project passed its own acceptance test while failing at its purpose,
# and nobody found out until clinicians noticed by hand.
#
# **3. An unstratified 80/20 split.** The table above shows the COVID count in validation
# drifting across ten draws of the same data. With so few cases, each one is worth several
# percentage points of recall.
# *Clinically:* the reported figure is partly an accident of the draw, so a rerun can
# appear to improve or degrade the model when nothing has changed — and a model can be
# approved or rejected on that noise.
#
# **4. No pretrained weights.** ResNet-18 trained from scratch on 960 images. There is not
# remotely enough data to learn general visual features, so capacity goes into memorising
# the training set. The gap between training and validation accuracy above is that
# memorisation, measured.
# *Clinically:* performance will not survive a different X-ray machine, a different
# exposure setting or a different hospital, because nothing general was learned.
#
# **5. A fixed learning rate and no augmentation.** 0.01 held constant for fifteen epochs,
# with no early stopping, no scheduler and no augmentation. The model cannot settle at the
# end of training, and it sees each image exactly as captured.
# *Clinically:* the model becomes sensitive to acquisition artefacts — patient rotation,
# contrast, cropping — rather than to lung pathology, and a routine change in radiography
# protocol silently degrades it.
#
# **A sixth, on top of the five asked for: the scarcity was self-imposed.** The archive
# holds 3,616 COVID images. The project used 120. Before any modelling fix is worth
# discussing, that is the cheapest available improvement by an enormous margin, and
# synthetic augmentation should be judged against it rather than instead of it.

# %% [markdown]
# ## R5.2 — a conditional GAN on the scarce class
#
# The generator is told which class to produce, so it can be asked specifically for COVID
# images. Real images are scaled to [-1, 1] to match the generator's tanh output.

# %%
def gan_batches(frame: pd.DataFrame) -> DataLoader:
    from torchvision import transforms

    to_signed = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    return DataLoader(
        ImagePathDataset(frame, to_signed, channels=1),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2,
    )


def train_gan(frame: pd.DataFrame, minibatch_stddev: bool, seed: int, label: str):
    """One adversarial training run, tracking sample diversity every epoch."""
    enable_determinism(seed)
    loader = gan_batches(frame)
    generator = ConditionalGenerator(GAN_LATENT, len(COVID_CLASSES), IMAGE_SIZE).to(DEVICE)
    discriminator = ConditionalDiscriminator(
        len(COVID_CLASSES), IMAGE_SIZE, minibatch_stddev=minibatch_stddev
    ).to(DEVICE)

    optimise_g = torch.optim.Adam(generator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    optimise_d = torch.optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    watch_noise = torch.randn(32, GAN_LATENT, device=DEVICE)
    watch_labels = torch.full((32,), 2, dtype=torch.long, device=DEVICE)   # COVID
    trace = []

    for epoch in range(1, GAN_EPOCHS + 1):
        g_total, d_total, steps = 0.0, 0.0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            noise = torch.randn(images.shape[0], GAN_LATENT, device=DEVICE)
            fakes = generator(noise, labels)

            optimise_d.zero_grad()
            d_loss = discriminator_loss(
                discriminator(images, labels),
                discriminator(fakes.detach(), labels),
                real_label=0.9,
            )
            d_loss.backward(); optimise_d.step()

            optimise_g.zero_grad()
            g_loss = generator_loss(discriminator(fakes, labels))
            g_loss.backward(); optimise_g.step()

            g_total += g_loss.item(); d_total += d_loss.item(); steps += 1

        generator.eval()
        with torch.no_grad():
            watched = generator(watch_noise, watch_labels)
        generator.train()

        trace.append({
            "epoch": epoch,
            "generator_loss": g_total / steps,
            "discriminator_loss": d_total / steps,
            "diversity": diversity_score(watched.cpu()),
        })
        if epoch % 10 == 0 or epoch == 1:
            row = trace[-1]
            print(f"  {label} epoch {epoch:3d}  G {row['generator_loss']:.3f}  "
                  f"D {row['discriminator_loss']:.3f}  diversity {row['diversity']:.4f}")

    return generator.eval(), pd.DataFrame(trace)


# %%
print("=== run 1: plain discriminator ===")
plain_generator, plain_trace = train_gan(
    subsample, minibatch_stddev=False, seed=SEEDS[0], label="plain"
)

# %% [markdown]
# ## R5.3 — diagnosing the instability
#
# The losses alone do not say whether training is healthy. Mode collapse — the generator
# finding one image the discriminator accepts and emitting it forever — leaves the loss
# curves looking unremarkable. The diversity trace is what makes it visible: it is the
# mean pairwise distance between 32 COVID samples drawn from fixed noise, measured every
# epoch, so a fall towards zero means the generator has stopped producing variety.

# %%
print(f"diversity at epoch 1   : {plain_trace['diversity'].iloc[0]:.4f}")
print(f"diversity at the end   : {plain_trace['diversity'].iloc[-1]:.4f}")
print(f"lowest diversity seen  : {plain_trace['diversity'].min():.4f} "
      f"at epoch {int(plain_trace.loc[plain_trace['diversity'].idxmin(), 'epoch'])}")

# %% [markdown]
# ### The mitigation
#
# The discriminator is given a minibatch standard deviation channel: the spread of
# features across the batch, appended before the final scoring layer. The mechanism is
# simple enough to state in one sentence — a plain discriminator judges each image alone
# and cannot tell thirty-two copies from thirty-two different pictures, while a
# batch-aware one can, so collapse becomes a signature it can learn to punish.

# %%
print("=== run 2: discriminator with minibatch standard deviation ===")
aware_generator, aware_trace = train_gan(
    subsample, minibatch_stddev=True, seed=SEEDS[0], label="batch-aware"
)

# %% [markdown]
# ### Evidence of improvement

# %%
figure, (diversity_axis, loss_axis) = plt.subplots(1, 2, figsize=(13, 4.5))

diversity_axis.plot(plain_trace["epoch"], plain_trace["diversity"],
                    label="plain discriminator", color="tab:orange")
diversity_axis.plot(aware_trace["epoch"], aware_trace["diversity"],
                    label="with minibatch std", color="tab:blue")
diversity_axis.set_xlabel("epoch"); diversity_axis.set_ylabel("sample diversity")
diversity_axis.set_title("Diversity of 32 COVID samples from fixed noise")
diversity_axis.legend(); diversity_axis.grid(alpha=0.3)

for trace, name, colour in ((plain_trace, "plain", "tab:orange"),
                            (aware_trace, "batch-aware", "tab:blue")):
    loss_axis.plot(trace["epoch"], trace["generator_loss"], color=colour, label=f"G, {name}")
    loss_axis.plot(trace["epoch"], trace["discriminator_loss"], "--", color=colour,
                   label=f"D, {name}")
loss_axis.set_xlabel("epoch"); loss_axis.set_ylabel("loss")
loss_axis.set_title("Losses — note how little they say about collapse")
loss_axis.legend(fontsize=8); loss_axis.grid(alpha=0.3)

figure.tight_layout(); plt.show()

comparison = pd.DataFrame({
    "run": ["plain", "with minibatch std"],
    "final diversity": [plain_trace["diversity"].iloc[-1], aware_trace["diversity"].iloc[-1]],
    "lowest diversity": [plain_trace["diversity"].min(), aware_trace["diversity"].min()],
    "mean over last 10 epochs": [plain_trace["diversity"].tail(10).mean(),
                                 aware_trace["diversity"].tail(10).mean()],
})
print(comparison.round(4).to_string(index=False))
comparison.to_csv("a4_gan_diversity.csv", index=False)

improved = comparison.loc[1, "mean over last 10 epochs"] > comparison.loc[0, "mean over last 10 epochs"]
print(f"\nmitigation improved sustained diversity: {improved}")

# %% [markdown]
# ## Generated COVID images
#
# The generator used from here is whichever run held up better, chosen on the diversity
# measure rather than by eye.

# %%
best_generator = aware_generator if improved else plain_generator
print(f"using the {'batch-aware' if improved else 'plain'} generator")

with torch.no_grad():
    sample_noise = torch.randn(16, GAN_LATENT, device=DEVICE)
    sample_covid = best_generator(sample_noise, torch.full((16,), 2, dtype=torch.long, device=DEVICE))
    sample_normal = best_generator(sample_noise, torch.zeros(16, dtype=torch.long, device=DEVICE))

figure, axes = plt.subplots(2, 8, figsize=(14, 4))
for column in range(8):
    axes[0, column].imshow(sample_covid[column, 0].cpu(), cmap="gray"); axes[0, column].axis("off")
    axes[1, column].imshow(sample_normal[column, 0].cpu(), cmap="gray"); axes[1, column].axis("off")
axes[0, 0].set_title("conditioned on COVID", loc="left", fontsize=9)
axes[1, 0].set_title("conditioned on Normal, same noise", loc="left", fontsize=9)
figure.suptitle("The same noise vectors under two different class labels")
figure.tight_layout(); plt.show()

print(f"the two rows differ: {not torch.allclose(sample_covid, sample_normal, atol=1e-3)}")
print("if they did not, the label would be being ignored and the augmentation would add nothing")

# %% [markdown]
# ## R5.4 — does synthetic augmentation raise COVID recall
#
# The experiment is kept narrow on purpose. One generator, its output frozen, and three
# seeded classifier pairs that differ **only** in whether the synthetic COVID images are
# present in training. Retraining the GAN per seed would mix two sources of variation and
# answer neither question.
#
# The split is stratified here, because problem 3 above has to be fixed before problem 1
# can be measured — otherwise the recall difference is partly split noise.

# %%
SYNTHETIC_ROOT.mkdir(exist_ok=True)
with torch.no_grad():
    noise = torch.randn(SYNTHETIC_PER_CLASS, GAN_LATENT, device=DEVICE)
    labels = torch.full((SYNTHETIC_PER_CLASS,), 2, dtype=torch.long, device=DEVICE)
    generated = best_generator(noise, labels).cpu()

synthetic_rows = []
for index, image in enumerate(generated):
    pixels = ((image[0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    path = SYNTHETIC_ROOT / f"synthetic_covid_{index:04d}.png"
    Image.fromarray(pixels, mode="L").save(path)
    synthetic_rows.append({"path": str(path.resolve()), "class_name": "COVID", "label": 2})

synthetic_frame = pd.DataFrame(synthetic_rows)
print(f"{len(synthetic_frame)} synthetic COVID images written to {SYNTHETIC_ROOT}/")

# %%
def train_and_score(train_frame: pd.DataFrame, val_frame: pd.DataFrame, seed: int) -> dict:
    enable_determinism(seed)
    model = build_original_model().to(DEVICE)
    train_classifier(
        model=model,
        train_loader=loader_for(train_frame, training=True),
        val_loader=loader_for(val_frame, training=False),
        epochs=CLASSIFIER_EPOCHS,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=DEVICE,
    )
    y_true, y_pred = predict(model, loader_for(val_frame, training=False), device=DEVICE)
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), n_classes=len(COVID_CLASSES))
    recall = per_class_recall(cm)
    return {
        "seed": seed, "accuracy": accuracy(cm), "macro_f1": macro_f1(cm),
        "recall_normal": recall[0], "recall_pneumonia": recall[1], "recall_covid": recall[2],
    }


pairs = []
for seed in SEEDS:
    real_train, validation = stratified_split(
        subsample, by=["label"], test_size=ORIGINAL_TEST_FRACTION, seed=seed
    )
    augmented_train = pd.concat([real_train, synthetic_frame], ignore_index=True)

    print(f"seed {seed}: {len(real_train)} real, {len(augmented_train)} with synthetic")
    without = train_and_score(real_train, validation, seed)
    with_synthetic = train_and_score(augmented_train, validation, seed)
    pairs.append({"seed": seed, "without": without, "with": with_synthetic})
    print(f"  COVID recall  without {without['recall_covid']:.4f}   "
          f"with {with_synthetic['recall_covid']:.4f}   "
          f"delta {with_synthetic['recall_covid'] - without['recall_covid']:+.4f}")

# %%
rows = []
for pair in pairs:
    for condition in ("without", "with"):
        rows.append({"seed": pair["seed"], "synthetic": condition, **pair[condition]})
recall_table = pd.DataFrame(rows).drop(columns=["seed"]).assign(
    seed=[row["seed"] for row in rows]
)
print(pd.DataFrame(rows).round(4).to_string(index=False))
pd.DataFrame(rows).to_csv("a4_recall_comparison.csv", index=False)

deltas = np.array([p["with"]["recall_covid"] - p["without"]["recall_covid"] for p in pairs])
print()
print("COVID recall delta by seed:", np.round(deltas, 4).tolist())
print(f"worst seed : {deltas.min():+.4f}   (eval C5 requires > 0)")
print(f"mean       : {deltas.mean():+.4f}")
print()
print("EVAL C5:", "PASS" if deltas.min() > 0 else "FAIL")
print()
print("A negative result here is a finding, not a failure. Synthetic augmentation does not")
print("reliably help minority-class recall, and reporting six numbers honestly is worth more")
print("to the improvement plan than a positive delta obtained by re-rolling seeds.")

# %%
figure, axis = plt.subplots(figsize=(7, 4.5))
width = 0.35
positions = np.arange(len(SEEDS))
axis.bar(positions - width / 2, [p["without"]["recall_covid"] for p in pairs],
         width, label="real images only", color="tab:orange")
axis.bar(positions + width / 2, [p["with"]["recall_covid"] for p in pairs],
         width, label="with synthetic COVID", color="tab:blue")
axis.axhline(original_recall[2], color="crimson", linestyle="--",
             label=f"reproduced baseline ({original_recall[2]:.2f})")
axis.set_xticks(positions, [f"seed {s}" for s in SEEDS])
axis.set_ylabel("COVID-class recall"); axis.set_ylim(0, 1)
axis.set_title("COVID recall, three seeded pairs"); axis.legend(); axis.grid(alpha=0.3, axis="y")
figure.tight_layout(); plt.show()

# %% [markdown]
# ## Measured cost

# %%
elapsed_minutes = (time.time() - STARTED_AT) / 60
peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0.0

print(f"RUNTIME   : {elapsed_minutes:.1f} minutes")
print(f"PEAK GPU  : {peak_mib:,.0f} MiB of 15,360 MiB available on a T4")
