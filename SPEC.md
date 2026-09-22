# Spec: MBA Computer Vision — CNN, Transformers and CLIP

**Status**: Approved — decisions 1-4 resolved 2026-09-20
**Date**: 2026-09-20
**Author**: Daniel Moreira
**Harness**: `~/.claude/rules/{tdd,edd,verification,anti-cheat}-discipline.md`, `skills/spec-driven-dev`

---

## Problem Statement

A graded MBA deliverable made of four activities and one technical report. The grade is
decided by a 26-line rubric, not by the assignment prose. Thirteen rubric lines demand
work the prose never mentions. Building to the prose alone loses half the marks.

The deliverable is four Colab notebooks plus one PDF. Everything is written in English.
The submitted artefacts must run top to bottom on a Colab T4 with no edits.

---

## Scope

### In scope
- Four notebooks: `A1_vision_transformers`, `A2_clip_ads16`, `A3_cnn_kaggle`, `A4_estudo_caso_raio_x`
- One installable package `cv_mba` holding every piece of real logic, unit tested
- One technical report covering activities 1, 2, 3, 4.1 and 4.2
- A rubric traceability file that fails a test when any rubric line has no evidence

### Out of scope
- A notebook for activity 4.2 (traffic). Report only, by instruction.
- Training any model for activity 2. CLIP is used frozen, zero-shot.
- Beating published benchmarks. The rubric grades justification, not leaderboard position.
- Any deployment, API or serving layer.

---

## Verified Dataset Facts

Every number below was produced by a command run on 2026-09-20, not recalled.

### A1 — ELPV solar cell electroluminescence
`git clone https://github.com/zae-bayern/elpv-dataset.git`

| Fact | Value |
|---|---|
| Labels file | `src/elpv_dataset/data/labels.csv` (not repo root — the repo was restructured) |
| Rows | 2,624 |
| Defect probability values | 0.0 → 1,508 · 0.333 → 295 · 0.667 → 106 · 1.0 → 715 |
| Wafer type | mono 1,074 · poly 1,550 |
| Binary split used here | functional (p < 0.5) **1,803** · defective (p >= 0.5) **821** |
| Defect rate by wafer type | mono 369/1,074 = 34.4% · poly 452/1,550 = 29.2% |
| Duplicate images | none — 0 md5 collisions across 2,624 files |
| Image format | 300x300, 8-bit, single channel greyscale |

The 0.333 bucket is annotator disagreement: roughly one expert in three called those
cells defective. Folding them into the positive class injects about 26% label noise into
the class the whole activity depends on, which makes the from-scratch ViT result
unreadable — you could not tell data scarcity from contradictory labels. Threshold at 0.5.
Splits are stratified jointly on label and wafer type, because defect rate differs by type
by 5 percentage points and an unstratified split lets that drift into the comparison.

### A2 — ADS-16 computational advertising
`kaggle datasets download -d groffo/ads16-dataset`

| Branch | Images | What it actually is |
|---|---|---|
| `part1/Ads` + `part2/Ads` | 151 + 150 = **301** | The real advertisements, in 16 product category folders |
| `part1/Corpus` + `part2/Corpus` | 1,158 + 1,238 = **2,396** | Personal photographs of the 120 survey participants |
| `Documents` | 0 | Empty of images |
| Total | 2,697 | |

The assignment asks for the corpus "or a subset of at least 500 representative images".
The advertising set is 301, not 500. This is handled in F2.1 below rather than papered over.

### A3 — Kaggle images-dataset (professor-specified)
`kaggle datasets download -d pavansanagapati/images-dataset`

| Class | Files | Format |
|---|---|---|
| bike | 365 | bmp |
| cars | 420 | bmp |
| cats | 202 | jpg |
| dogs | 202 | jpg |
| flowers | 210 | png |
| horses | 202 | jpg |
| human | 202 | jpg |
| **Unique total** | **1,803** | matches the "1,800 images" in the brief |

**Trap, and it is the important finding of this section.** The archive contains
`data/` and, nested inside it, `data/data/` — a byte-for-byte duplicate of all seven
class folders. On disk there are 3,606 files, exactly twice 1,803. Pointing
`ImageFolder` at `data/` recursively yields 3,606 images and a phantom eighth class
named `data`, with every image present twice. An 80/20 split over that puts the same
image in train and validation, and validation accuracy becomes meaningless.

This is the same category of defect the assignment's own case studies ask the student
to diagnose. The loader must enumerate the seven class folders explicitly and assert
the count is 1,803. That assertion is a test, listed in F3.1.

### A4.1 — COVID-19 Radiography Database
`kaggle datasets download -d tawsifurrahman/covid19-radiography-database`

| Class | Available | Used, to reproduce the failed project |
|---|---|---|
| Normal | 10,192 | 840 |
| Viral Pneumonia | 1,345 | 240 |
| COVID | 3,616 | 120 |
| Lung_Opacity | 6,012 | unused |

The brief describes a prior group's project using 1,200 images at 840/240/120. We
reproduce that split exactly, from a fixed seed, so the failure being diagnosed is
reproduced rather than described. Fixes are then measured against it.

---

## Architecture

Real logic lives in an installable package. Notebooks narrate and orchestrate; they hold
no logic that could be unit tested. This is what makes rubric 2.1's word "testable" true
rather than claimed, and it is what lets the notebooks run on Colab unmodified.

```
src/cv_mba/
  attention.py     ScaledDotProductAttention, MultiHeadAttention      R2.1
  encoder.py       TransformerEncoderBlock                            R2.3
  positional.py    SinusoidalPositionalEncoding, LearnedPositional    R2.4
  vit.py           PatchEmbedding, VisionTransformer                  R3.1
  metrics.py       per_class_accuracy, per_class_recall, confusion    R1.2 R5.4
  data.py          elpv_split, a3_split, covid_subsample, ads16_index
  clip_search.py   embed_images, embed_texts, rank_concepts, search   R4.2 R4.3
  gan.py           ConditionalGenerator, ConditionalDiscriminator     R5.2
  rubric.py        RUBRIC registry, evidence loader, gap detector

tests/             one file per module, mirrors the layout
notebooks/         jupytext percent-format .py is the source of truth; .ipynb is generated
report/            report.md, figures/, → PDF
evidence.yaml      rubric id → artefact path + anchor string
```

Colab cell one of every notebook:

```python
!pip install -q git+https://github.com/danimoreira90/cnn-transformers-clip.git
```

The repo is public, so no token is needed. `pyproject.toml` declares `numpy>=2.1.3`,
`pandas>=2.2.3`, `matplotlib>=3.10.0`, `scikit-learn>=1.6.1`, `pillow>=11.3.0`,
`transformers>=5.16.1` — all already satisfied on Colab, so pip installs nothing and
touches nothing. `torch` and `torchvision` are deliberately absent from the main
dependency list and sit in the `local` extra, so pip can never replace Colab's
`2.11.0+cu128` build with a CPU wheel. Locally, `resolution = "lowest-direct"` pins the
dev machine to Colab's exact versions. Verified locally on 2026-09-20:
`2.1.3 2.2.3 3.10.0 1.6.1 11.3.0 5.16.1 2.11.0+cpu`.

Kaggle credentials on Colab come from `google.colab.userdata`, never from a file
committed or pasted into a cell. This follows `rules/coding-standards.md`, no secrets in
code.

---

## Functional Requirements

Priority is Must for every line, because every line is a rubric line. The rubric is
pass/fail per item; there is no partial credit to trade away.

| ID | Requirement | Rubric | Artefact |
|---|---|---|---|
| F1.1 | Load pretrained CNN, replace head with 7 outputs, freeze backbone, train head only, one run | R1.1 | A3 |
| F1.2 | Loss and accuracy curves per epoch; per-class and global accuracy table | R1.2 | A3 |
| F1.3 | At least three augmentation strategies, each justified per class, each with the classes it could harm | R1.3 | Report |
| F1.4 | Written: feature extraction vs fine-tuning as a function of dataset size and domain distance | R1.4 | Report |
| F1.5 | Written: why this backbone, against T4 memory and seven classes | R1.5 | Report |
| F2.1 | `ScaledDotProductAttention` and `MultiHeadAttention` from scratch, independent per-head projections, concatenated output, unit tested | R2.1 | Package + A1 |
| F2.2 | Attention weight heatmap for at least one head on a real image, with written reading of what it weights | R2.2 | A1 |
| F2.3 | `TransformerEncoderBlock`: two-layer feedforward, LayerNorm, residual connections, used as the ViT's block | R2.3 | Package + A1 |
| F2.4 | Positional encoding applied to the token sequence; written explanation of permutation equivariance | R2.4 | A1 + Report |
| F2.5 | Written: BERT masked-token pretraining vs ViT pretraining, what each objective maximises | R2.5 | Report |
| F3.1 | Patch embedding, learnable CLS token, positional encoding, complete ViT image → logits | R3.1 | Package + A1 |
| F3.2 | Train ViT from scratch on ELPV; attention maps from at least one head; written account of emergent regions | R3.2 | A1 |
| F3.3 | Fine-tune a pretrained ViT on ELPV, head replaced, compared to from-scratch in one table | R3.3 R3.6 | A1 |
| F3.4 | Written: what DeiT fixes and what Swin fixes in the original ViT | R3.4 | Report |
| F3.5 | Written: when ViT beats CNN and when CNN wins, argued from the ELPV numbers | R3.5 | Report |
| F4.1 | Written: why contrastive pretraining enables retrieval with no supervised training | R4.1 | Report + A2 |
| F4.2 | Concept ranking over ADS-16, at least 20 descriptions, threshold justified, top 5 visualised with example images | R4.2 | A2 |
| F4.3 | Text-to-image search, at least 8 queries across the specificity range, each result analysed | R4.3 | A2 |
| F4.4 | Written: CLIP text encoding vs BERT tokenisation, role of padding and attention mask, backed by a printed comparison | R4.4 | Report + A2 |
| F5.1 | At least five technical defects in the X-ray project, each with its clinical consequence | R5.1 | Report |
| F5.2 | Conditional GAN on chest X-rays with a correct adversarial training loop | R5.2 | A4.1 |
| F5.3 | Diagnose mode collapse or divergence, apply at least one mitigation, show measured improvement | R5.3 | A4.1 |
| F5.4 | COVID-class recall with and without synthetic images, same seed, same split | R5.4 | A4.1 |
| F5.5 | At least four defects in the traffic project plus the ImageNet transfer risks | R5.5 | Report |
| F5.6 | Integrated improvement plan covering model, metric, synthetic augmentation and a clinical adoption criterion | R5.6 | Report |

### F2.1 — ADS-16 corpus decision

The advertising set is 301 images and the brief's fallback threshold is 500. Rather than
padding the ad set with participant selfies to clear a number, the pipeline runs over all
2,697 images and reports two partitions:

- **Primary ranking** over the 301 advertisements. This is the advertising corpus and the
  activity is about advertising.
- **Control ranking** over the 2,396 participant photographs.

The contrast is the point. "Text and logo" should dominate the ad partition and collapse
in the control; "a person" should do the reverse. That is a real validity check on CLIP's
zero-shot behaviour, it reads as analysis rather than compliance, and it satisfies both
the "whole corpus" reading and the 500-image floor. Search in F4.3 indexes all 2,697,
because retrieval over a mixed corpus is a harder and more informative test than
retrieval over ads alone.

---

## Non-Functional Requirements

| Constraint | Target |
|---|---|
| GPU | Tesla T4, 15,360 MiB, driver 580.82.07 — verified on Colab 2026-09-20 |
| Per-notebook wall clock | under 45 minutes end to end |
| A1 total, both ViT trainings | under 25 minutes |
| A4.1 GAN training | under 20 minutes |
| Notebook header | actual peak memory and actual wall clock, read from a completed run — never estimated |
| Reproducibility | every split and every training seeded; seed recorded in the notebook |

The assignment requires each notebook to state memory and runtime. Those numbers are
written after the first complete Colab run, from `torch.cuda.max_memory_allocated()` and
measured wall clock. An estimated number in that header would be an unverified claim,
which `rules/verification-discipline.md` forbids.

---

## Testing Strategy

Unit tests cover shape, algebra and invariants. They run on CPU in under thirty seconds
and never touch the network, per `rules/testing-requirements.md`.

| Test | What it proves |
|---|---|
| `test_attention.py` | Output shape equals input shape; softmax rows sum to 1; a masked position receives zero weight; scaling by `sqrt(d_k)` is applied; heads use independent projections (perturbing one head's weights changes only that head's output); concatenated output width equals `n_heads × d_head` |
| `test_encoder.py` | Residual path is present (zeroing the sublayer returns the input); LayerNorm produces zero mean and unit variance per token; feedforward has exactly two linear layers |
| `test_positional.py` | Encoding is added, not concatenated; two different token orders give different outputs with PE and identical outputs without it — this is the executable proof behind R2.4 |
| `test_vit.py` | Patch count equals `(H/P)×(W/P)`; sequence length equals patches + 1; CLS token sits at index 0 and is learnable; forward returns `(batch, n_classes)` |
| `test_metrics.py` | Per-class recall on a hand-built confusion matrix matches hand-computed values; a class with zero support raises rather than returning silent NaN |
| `test_data.py` | ELPV loader finds 2,624 rows; A3 loader returns exactly 1,803 unique images across 7 classes and raises if the `data/data` duplicate is reachable; COVID subsample is exactly 840/240/120 and stratified; splits contain no shared file path |
| `test_clip_search.py` | Cosine similarity is symmetric and bounded in [-1, 1]; embeddings are L2-normalised; ranking is stable under row permutation of the corpus |
| `test_gan.py` | Generator output shape matches the discriminator's expected input; the conditioning label changes the output; discriminator and generator losses move in opposite directions on a crafted batch |
| `test_determinism.py` | Two forward passes over the same batch produce bitwise-identical tensors under the deterministic flags. GPU-marked; skipped on CPU with a reason, never silently. |
| `test_rubric_coverage.py` | Every one of the 26 rubric ids appears in `evidence.yaml`; every referenced file exists; every anchor string is found in that file |

`test_rubric_coverage.py` is the single highest-value test in the project. It is the one
that fails when a rubric line has quietly lost its artefact during a refactor, which is
the failure mode that actually costs marks.

No test asserts on training accuracy. Those are slow, stochastic and would produce
flaky red, which under `rules/anti-cheat-discipline.md` leads straight to the temptation
to skip or weaken them. Model quality is gated by evals, not unit tests.

---

## EDD Requirements

Eval file: `.claude/evals/cv-mba.md`, written before any model code.

| ID | Capability | Grader | Threshold |
|---|---|---|---|
| C1 | Concept ranking over the ad partition puts advertising-typical concepts above person-typical ones | code | top-5 of ads partition differs from top-5 of control partition by at least 2 entries |
| C2 | Text search returns semantically correct top-1 for the 4 concrete queries of the 8 | human-labelled, code-checked | 4 of 4 |
| C3 | Pretrained ViT beats from-scratch ViT on ELPV macro-F1 | code | min delta across 3 seeds > 0 AND mean delta > 0.05 |
| C4 | Conditional GAN generates COVID-conditioned images distinguishable from Normal-conditioned by the frozen baseline classifier | code | better than chance by a stated margin |
| C5 | Synthetic augmentation raises COVID recall against the reproduced 840/240/120 baseline | code | one frozen synthetic set, 3 seeded classifier pairs; min recall delta > 0, pass@3 >= 0.90 |

| ID | Regression | Grader | Threshold |
|---|---|---|---|
| R1 | Package imports and every unit test passes on CPU | code | pass^3 = 1.00 |
| R2 | Every notebook executes end to end on T4 with no edits | code (`jupyter nbconvert --execute`) | pass^3 = 1.00 |
| R3 | Rubric coverage is complete | code | pass^3 = 1.00 |
| R4 | CLIP embeddings are bitwise reproducible within a session | code | pass^3 = 1.00 |

### Deviation from the EDD Iron Law — decided 2026-09-20

`rules/edd-discipline.md` sets `capability pass@3 >= 0.90` to absorb LLM
non-determinism. Three adjustments, each argued on the system rather than on
convenience. The rule against bending a gate to fit a weak implementation stands; none
of these is that.

**C1 and C2 run once, but only after determinism is proven.** GPU floating point is not
deterministic by default: cuDNN selects algorithms by benchmarking and some reductions
accumulate in thread-completion order, so results can differ in the last bits. A concept
ranking sorts similarity scores, and two concepts within 1e-6 of each other will swap
places between runs. So determinism is established, not assumed:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

and eval **R4** embeds the same batch twice in one session and asserts bitwise equality.
R4 passing is what earns C1 and C2 a single run against a committed golden snapshot. If
R4 fails, C1 and C2 revert to three runs. The snapshot is a stronger guarantee than a
triple run, because it also fails when Colab moves past `transformers` 5.16.1.

**C3 gates on the worst seed, not the mean.** Three seeds, and the gate is `min delta > 0`
with `mean delta > 0.05`. The pretrained model must win on every seed, which is strictly
harder to pass than 90% of runs. The three per-seed numbers become the comparison table
rubric 3.6 asks for, with seed spread visible.

**C5 freezes the treatment.** Three full GAN trainings would measure GAN reproducibility
and augmentation benefit at the same time and answer neither. The GAN trains once, the
generated set is frozen, and three seeded classifier pairs run with and without it on the
same split. That isolates the one variable rubric 5.4 asks about, and costs roughly 40
fewer T4 minutes.

**C4 keeps `pass@3 >= 0.90` unchanged.** GAN sampling is genuinely unstable and the
triple run does real work there.

---

## Edge Cases

| Case | Expected behaviour |
|---|---|
| A3 loader reaches `data/data` | Raise with the duplicate path named. Never silently train on 3,606 images. |
| ELPV `labels.csv` not at repo root | Loader resolves `src/elpv_dataset/data/labels.csv`; raises with the searched paths if absent. |
| Colab session loses the downloaded ADS-16 archive | Notebook re-downloads; optional Drive cache is a documented opt-in, never required for a clean run. |
| Kaggle credentials absent on Colab | Fail immediately with the exact `userdata` secret names to set. Do not fall back to an interactive prompt that would hang an unattended run. |
| GAN collapses to one mode | Detected by a diversity metric inside the loop, reported, mitigated, and the before-and-after shown. This is R5.3, so the failure is an expected path, not an error. |
| Class with zero support in a metric | Raise. A silent NaN in per-class recall is precisely the reporting defect A4.1 is about. |
| `transformers` upgraded on Colab past 5.16.1 | Golden snapshot eval fails, which is the intended warning, not a crash. |

---

## Decisions Made

1. **A1 domain: ELPV solar cell defect detection.** EuroSAT was the only serious
   alternative and loses twice: more data narrows the from-scratch versus pretrained gap
   that rubrics 3.3 and 3.6 exist to explain, and land-cover tiles give attention maps with
   nothing localised to look at. Solar cell cracks are thin bright lines in one place, so
   attention either lands on them or visibly does not, and either result is writable.
   Medical was excluded to avoid doing chest X-rays twice in one submission.
2. **EDD: adapted as set out above**, with determinism proven by R4 rather than asserted.
3. **`_quarantine` is superseded material; the live `rules/` files govern.** Evidence:
   `_quarantine/anti-cheat-laws.md` is 10,000 bytes and older, the live
   `anti-cheat-discipline.md` is 4,271 bytes and newer; same shape for
   `clean-code-principles.md` against `coding-standards.md`; and `CLAUDE.md` imports the
   eight live files and none of the quarantined ones. That is a consolidation.
4. **ELPV labels: threshold at 0.5**, splits stratified jointly on label and wafer type.

### Harness defect found, out of scope for this project

Two live Iron Laws close with pointers into quarantine:
`rules/tdd-discipline.md` -> `skills/tdd-workflow.md` and
`rules/anti-cheat-discipline.md` -> `skills/anti-cheat/SKILL.md`. Both targets sit in
`skills/_quarantine/`. An agent following either finds nothing, or finds the superseded
version and follows it. Fix by deleting the two Skill Reference sections or promoting
those two skills back out.

5. **Submission filenames: the discipline is `deep-learning-and-vision`.** Decided by
   inference on 2026-09-22 rather than by asking.

   The ZIP template `nomedoaluno_nomedadisciplina_pd.ZIP` has one slot, labelled name of
   the discipline. The PDF name carries two strings, so one of them is the discipline and
   one is not. The rubric decides which: lines 2.5 and 4.4 grade BERT pretraining
   objectives and BERT tokenisation, and competency heading 2 reads "from the attention
   mechanism to fine-tuning BERT". A discipline named Computer Vision does not examine a
   language model. So `deep-learning-and-vision` is the discipline, and `computer-vision`
   is the project descriptor, matching the project's own title, Visao Computacional com
   Transformers.

       daniel_moreira_deep-learning-and-vision_computer-vision.pdf
       daniel_moreira_deep-learning-and-vision_pd.ZIP

   Residual risk is low and hedged. The PDF name is fully specified by the brief; only
   the archive name is inferred, and the grader opens the archive to find a correctly
   named PDF inside. Task 7.5 also places a README.txt at the archive root naming the
   student, discipline, project and contents, so the archive identifies itself if the
   name is ever questioned.

## Still Open

1. **AI usage citation.** The brief states that using AI without citing it is academic
   misconduct. The report needs an AI usage section. Separate from git trailers, which
   stay clean.

## Changelog

- 2026-09-22: Submission filenames resolved by inference from the rubric's BERT lines
  rather than by asking the professor. Only the AI usage citation remains open, and that
  is a task rather than a question.
- 2026-09-20 (2): Open questions 1-4 resolved. ELPV binary threshold corrected from p > 0
  to p >= 0.5 after the annotator-agreement cross-tab showed 26% label noise in the
  positive class at the original threshold. EDD deviation tightened: determinism proven by
  eval R4 rather than assumed, C3 gates on worst seed, C5 freezes the synthetic set.
  Joint label-and-wafer-type stratification added.
- 2026-09-20: Initial draft. All four datasets downloaded and verified. A3 duplicate-folder
  trap found and made a test. ADS-16 ad count established at 301 against a 500 threshold,
  resolved by partitioned analysis rather than padding.
