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
# # Activity 2 — Semantic recognition in visual advertising with CLIP
#
# No model is trained here. CLIP's pretrained embeddings and natural-language queries do
# all of the work.
#
# **Covers rubric lines R4.2 and R4.3**, and produces the printed evidence behind R4.4.
# R4.2 — rank objects by semantic frequency across the corpus using at least 20
# descriptions, with a justified threshold, and show the five most frequent with examples.
# R4.3 — text-to-image search with at least 8 queries spanning the specificity range,
# each result analysed.
# R4.4 — compare CLIP's text query mechanism with BERT's tokenisation, and the role of
# padding and the attention mask.
#
# **Requirements.** Colab with the T4 GPU runtime. Measured runtime and peak GPU memory
# are printed by the final cell.
#
# **Data.** `groffo/ads16-dataset` from Kaggle, 1.5 GB, downloaded by this notebook. Set
# `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab's Secrets panel.
#
# ## What this corpus actually is, and why it is split in two
#
# The archive holds 2,697 images. Only **301 of them are advertisements**. The other
# **2,396 belong to the 120 survey participants** — pictures they supplied, sorted into
# folders they rated positively and negatively.
#
# The brief offers "the corpus, or a subset of at least 500 representative images". The
# advertising set is 301, short of that number. Padding it with participant photographs
# to reach 500 would put personal snapshots into a ranking of advertising content and
# make the result meaningless.
#
# So this notebook runs over all 2,697 and reports the two partitions separately: the
# advertisements as the subject, the participant pictures as a control. The contrast is
# the finding. Concepts like text and logos should dominate the advertising partition and
# collapse in the control; people should do the reverse. That is a real check on whether
# CLIP's zero-shot reading of this corpus means anything, and it satisfies both the whole
# corpus reading and the 500-image floor without pretending.

# %%
# !pip install -q git+https://github.com/danimoreira90/cnn-transformers-clip.git

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from cv_mba.clip_search import cosine_similarity, embed_images, embed_texts, rank_concepts, top_matches
from cv_mba.data import ads16_index
from cv_mba.determinism import enable_determinism

# %% [markdown]
# ## Configuration

# %%
SEED = 0
CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
STANDARD_DEVIATIONS = 1.0   # how far above an image's own mean a concept must score
TOP_K = 5
DATA_ROOT = Path("data/raw/ads16")

enable_determinism(SEED)
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
if not any(DATA_ROOT.glob("ADS16_Benchmark_part*")):
    # !pip install -q kaggle
    # !kaggle datasets download -d groffo/ads16-dataset -p {DATA_ROOT} --unzip
    print("downloaded")
else:
    print("already present, skipping download")

# %% [markdown]
# ## The corpus

# %%
corpus = ads16_index(DATA_ROOT)
advertisements = corpus[corpus["partition"] == "ads"].reset_index(drop=True)
participants = corpus[corpus["partition"] == "corpus"].reset_index(drop=True)

print(f"total images              : {len(corpus)}")
print(f"advertisements            : {len(advertisements)}")
print(f"participant pictures      : {len(participants)}")
print(f"advertisement folders     : {advertisements['group'].nunique()}")
print(f"survey participants       : {participants['group'].nunique()}")

# %% [markdown]
# A note for the report. The brief describes the advertisements as sitting in **16 product
# categories**. The archive ships them in **20 numbered folders** and contains no file
# naming any of them — the only non-image files are a licence and the source archives. The
# folder is therefore reported as found rather than relabelled as a product category, and
# where the sixteen comes from cannot be established from what was shipped.

# %% [markdown]
# ## Loading CLIP
#
# Frozen, in evaluation mode, under `no_grad`. Nothing here is trained or adapted; the
# only thing that changes between one question and the next is the text.

# %%
model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters, all frozen")

# %% [markdown]
# ## Embedding the corpus
#
# Every image becomes one unit-length vector in the space CLIP shares with text. After
# this cell, answering a new question costs a single matrix multiply.

# %%
image_embeddings = embed_images(
    list(corpus["path"]), model, processor, device=DEVICE, batch_size=BATCH_SIZE,
    on_batch=lambda done, total: print(f"\rembedded {done}/{total}", end=""),
)
print(f"\nembeddings: {tuple(image_embeddings.shape)}")

ads_mask = (corpus["partition"] == "ads").to_numpy()
ads_embeddings = image_embeddings[ads_mask]
participant_embeddings = image_embeddings[~ads_mask]

# %% [markdown]
# ## 2.1 — Ranking objects by semantic frequency
#
# Twenty-five descriptions, chosen to cover what advertising imagery is made of — people,
# products, settings, and the text and branding that distinguishes an advertisement from
# an ordinary photograph.

# %%
CONCEPTS = [
    "a car", "a person", "a human face", "a group of people", "a child",
    "food or a drink", "a bottle", "a piece of clothing", "a cosmetic product",
    "an electronic device", "a mobile phone", "a computer screen",
    "a piece of furniture", "a household product", "a toy",
    "text and a logo", "a printed price or discount offer",
    "outdoor scenery", "a building", "a plant or a flower",
    "an animal", "a sports scene", "a travel destination",
    "a medical or health product", "a chart or a diagram",
]
print(f"{len(CONCEPTS)} concept descriptions")

concept_embeddings = embed_texts(CONCEPTS, model, processor, device=DEVICE)
similarity = cosine_similarity(image_embeddings, concept_embeddings)
print(f"similarity matrix: {tuple(similarity.shape)}")

# %% [markdown]
# ### Justifying the threshold
#
# CLIP's raw similarity values are not comparable between images. A cluttered or dark
# picture scores lower against *every* concept than a clean studio shot does. A single
# absolute cut would therefore count concepts in the bright images and miss the same
# concepts in the dim ones, and the ranking would partly measure image quality rather
# than image content.
#
# So presence is decided **within each image**: a concept counts as present when it scores
# more than one standard deviation above that image's own mean across all 25 concepts.
# The plot below is the reason — the spread of raw similarities per image is wide enough
# that no single absolute line sits in a sensible place for all of them.

# %%
figure, (raw_axis, standardised_axis) = plt.subplots(1, 2, figsize=(12, 4))

raw_axis.hist(similarity.flatten().numpy(), bins=60, color="steelblue")
raw_axis.set_title("Raw cosine similarity, every image against every concept")
raw_axis.set_xlabel("cosine similarity"); raw_axis.set_ylabel("count")

per_image_mean = similarity.mean(dim=1)
raw_axis.axvline(per_image_mean.min().item(), color="crimson", linestyle="--")
raw_axis.axvline(per_image_mean.max().item(), color="crimson", linestyle="--",
                 label="range of per-image means")
raw_axis.legend()

standardised = (similarity - similarity.mean(dim=1, keepdim=True)) / similarity.std(dim=1, keepdim=True)
standardised_axis.hist(standardised.flatten().numpy(), bins=60, color="seagreen")
standardised_axis.axvline(STANDARD_DEVIATIONS, color="crimson", linestyle="--",
                          label=f"threshold = {STANDARD_DEVIATIONS} sd")
standardised_axis.set_title("After standardising within each image")
standardised_axis.set_xlabel("standard deviations above that image's mean")
standardised_axis.legend()

figure.tight_layout(); plt.show()

print(f"per-image mean similarity spans {per_image_mean.min():.3f} to {per_image_mean.max():.3f}")
print("a single absolute cut would sit above some images' best concept and below others' worst")

# %% [markdown]
# ### The ranking, on each partition

# %%
ads_ranking = rank_concepts(similarity[ads_mask], CONCEPTS, threshold=STANDARD_DEVIATIONS)
participant_ranking = rank_concepts(similarity[~ads_mask], CONCEPTS, threshold=STANDARD_DEVIATIONS)

comparison = (
    ads_ranking.set_index("concept")[["frequency", "mean_similarity"]]
    .join(
        participant_ranking.set_index("concept")[["frequency", "mean_similarity"]],
        lsuffix="_ads", rsuffix="_participants",
    )
    .sort_values("frequency_ads", ascending=False)
)
print(comparison.round(4).to_string())
comparison.to_csv("a2_concept_ranking.csv")

# %%
top_ads = list(ads_ranking["concept"][:TOP_K])
top_participants = list(participant_ranking["concept"][:TOP_K])

print(f"top {TOP_K} in the advertisements      : {top_ads}")
print(f"top {TOP_K} in the participant control : {top_participants}")
differing = [concept for concept in top_ads if concept not in top_participants]
print(f"in the advertisement top {TOP_K} but not the control : {differing}")
print(f"entries that differ                    : {len(differing)} of {TOP_K}")

# %% [markdown]
# ### The five most frequent concepts, with the images that confirm them
#
# Each row is one concept; each image is among the advertisements that scored highest
# against it. This is the visual check that the ranking is reading the pictures rather
# than the wording of the descriptions.

# %%
figure, axes = plt.subplots(TOP_K, TOP_K, figsize=(13, 13))
ads_similarity = similarity[ads_mask]

for row, concept in enumerate(top_ads):
    column_index = CONCEPTS.index(concept)
    best, scores = top_matches(ads_similarity[:, column_index], k=TOP_K)
    for column, (position, score) in enumerate(zip(best.tolist(), scores.tolist())):
        axis = axes[row, column]
        with Image.open(advertisements["path"].iloc[position]) as handle:
            axis.imshow(handle.convert("RGB"))
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(f"{score:.3f}", fontsize=8)
        if column == 0:
            axis.set_ylabel(concept, fontsize=8, rotation=0, ha="right", va="center")

figure.suptitle(f"Top {TOP_K} concepts in the advertising partition, with their strongest matches")
figure.tight_layout(); plt.show()

# %% [markdown]
# ## 2.2 — Search by natural-language query
#
# Ten queries running from a bare noun to an abstract feeling. The interesting cases are
# the ones where CLIP returns something the query did not literally describe.

# %%
QUERIES = [
    "a car",                                        # generic, concrete
    "a red sports car on an empty road",            # specific, concrete
    "a person smiling at the camera",               # generic, concrete
    "a family sharing a meal together",             # specific, concrete
    "a smartphone held in someone's hand",          # very specific, concrete
    "a large discount printed in bold letters",     # specific, and about text itself
    "luxury",                                       # abstract, single word
    "freedom",                                      # abstract, single word
    "something that feels safe and reassuring",     # abstract, phrased as a feeling
    "the moment before something exciting happens", # abstract, narrative
]
print(f"{len(QUERIES)} queries")

query_embeddings = embed_texts(QUERIES, model, processor, device=DEVICE)
query_similarity = cosine_similarity(query_embeddings, image_embeddings)

# %%
figure, axes = plt.subplots(len(QUERIES), TOP_K, figsize=(13, 2.6 * len(QUERIES)))

records = []
for row, query in enumerate(QUERIES):
    best, scores = top_matches(query_similarity[row], k=TOP_K)
    for column, (position, score) in enumerate(zip(best.tolist(), scores.tolist())):
        axis = axes[row, column]
        with Image.open(corpus["path"].iloc[position]) as handle:
            axis.imshow(handle.convert("RGB"))
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(f"{corpus['partition'].iloc[position]}  {score:.3f}", fontsize=7)
        if column == 0:
            axis.set_ylabel(query, fontsize=7, rotation=0, ha="right", va="center")
        records.append({
            "query": query, "rank": column + 1, "score": round(score, 4),
            "partition": corpus["partition"].iloc[position],
            "path": corpus["path"].iloc[position],
        })

figure.suptitle("Top 5 results per query")
figure.tight_layout(); plt.show()

results = pd.DataFrame(records)
results.to_csv("a2_search_results.csv", index=False)

# %% [markdown]
# ### Where the results came from
#
# The share of advertisements against participant pictures in each query's top five is
# itself informative. Concrete product queries should lean towards the advertisements;
# queries about people and feelings should lean towards the participants' own pictures.

# %%
print(
    results.groupby("query")["partition"]
    .value_counts()
    .unstack(fill_value=0)
    .reindex(QUERIES)
    .to_string()
)

# %% [markdown]
# ## R4.4 — CLIP's text queries against BERT's tokenisation
#
# Both models turn a sentence into integers and both pad a batch to one length, but they
# are answering different questions, and the padding and the attention mask are where the
# difference shows.

# %%
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

SENTENCES = ["a car", "a red sports car on an empty road at night"]

clip_tokens = processor(text=SENTENCES, return_tensors="pt", padding=True)
bert_tokens = bert_tokenizer(SENTENCES, return_tensors="pt", padding=True)

for name, tokens, tokenizer in (
    ("CLIP", clip_tokens, processor.tokenizer),
    ("BERT", bert_tokens, bert_tokenizer),
):
    print(f"===== {name} =====")
    for row, sentence in enumerate(SENTENCES):
        ids = tokens["input_ids"][row]
        mask = tokens["attention_mask"][row]
        print(f"  {sentence!r}")
        print(f"    tokens        : {tokenizer.convert_ids_to_tokens(ids.tolist())}")
        print(f"    attention mask: {mask.tolist()}")
        print(f"    real tokens   : {int(mask.sum())} of {len(mask)}")
    print()

# %% [markdown]
# What the output above shows, in order.
#
# **Both pad, and both mark the padding.** The short sentence is stretched to the length
# of the long one so the batch is one rectangular tensor. The attention mask is zero at
# exactly those added positions, and inside the model that zero becomes a minus-infinity
# entry in the attention scores, so softmax gives padding exactly zero weight. Without the
# mask, padding would be attended to like any other token and the short sentence's
# meaning would drift with the length of whatever else happened to be in the batch.
#
# **The special tokens differ, and so does what is read out.** BERT wraps the sentence in
# `[CLS]` and `[SEP]`, and fine-tuning reads the `[CLS]` position as the sentence summary.
# CLIP wraps it in start and end markers and reads the **end** token, because its text
# encoder is causal — each token may only look backwards — so the final position is the
# only one that has seen the whole sentence.
#
# **CLIP has no separate padding token, and the printout above shows it.** BERT pads with
# a dedicated `[PAD]`. CLIP repeats `<|endoftext|>`, so the very token it reads out is the
# same token it pads with. The attention mask is what keeps those apart: the first
# `<|endoftext|>` is masked in and is the sentence summary, every repeat after it is
# masked out. Drop the mask and a short caption in a batch of long ones is summarised
# from a padding position — silently, with no error, and with a vector that drifts
# depending on what else happened to be in the batch.
#
# **The objectives differ, which is the real difference.** BERT is pretrained to fill in
# masked words using both sides of the gap, so its representation is tuned for questions
# about language. CLIP's text encoder is pretrained only to land near the matching image,
# so its representation is tuned for agreeing with a picture. Neither is a general
# sentence embedding, and this is why CLIP can answer "which image is this" with no
# supervised training while BERT cannot.

# %% [markdown]
# ## Measured cost

# %%
elapsed_minutes = (time.time() - STARTED_AT) / 60
peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0.0

print(f"RUNTIME   : {elapsed_minutes:.1f} minutes")
print(f"PEAK GPU  : {peak_mib:,.0f} MiB of 15,360 MiB available on a T4")
