# Plan: MBA Computer Vision — CNN, Transformers and CLIP

**Spec**: `SPEC.md` (Approved, 2026-09-20)
**Status**: Ready to implement
**Evals**: `.claude/evals/cv-mba.md`

> **For agentic workers:** REQUIRED: Use subagent-driven-development for every task.
> Each task = one fresh subagent. Spec compliance review → code quality review → done.
> No task is "done" without: passing verification command, reviewer sign-off.

---

## EDD Preamble

This project contains model-backed behaviour. Per `rules/edd-discipline.md`, evals are
defined before any model code. `.claude/evals/cv-mba.md` is written in Task 0.3, before
Phase 2 begins, and its BASELINE run happens there.

Gate criteria, as adjusted and argued in SPEC.md:

```
C1, C2  pass@1 against a committed golden snapshot, conditional on R4 passing
C3      min delta across 3 seeds > 0  AND  mean delta > 0.05
C4      pass@3 >= 0.90
C5      one frozen synthetic set, 3 seeded classifier pairs, min recall delta > 0, pass@3 >= 0.90
R1..R4  pass^3 = 1.00
```

No model code is written before the eval file exists and its baseline failure is recorded.

---

## Build order rationale

Ordered by risk and dependency, not by activity number.

The rubric scaffold comes first because it is the only artefact that detects a rubric line
silently losing its evidence during a later refactor, which is the failure that actually
costs marks. Data and metrics come next because everything depends on them. The
transformer core is third: it is pure CPU, fully testable, and carries five rubric lines
on its own. A3 is the first notebook not because it is the first activity but because it
is the cheapest complete pipeline, so it proves the Colab install path and a training loop
before any expensive session is spent. The GAN is last because it is the only component
that can fail for reasons nobody can fix by reading code.

---

## Phase 0 — Scaffold and rubric registry

Implements: the coverage guarantee behind every rubric line.

### Task 0.1 — Repo configuration
- **File**: `pyproject.toml`, `.gitignore`, `.claude/settings.json`
- **Action**: Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and
  `markers = ["gpu: requires CUDA"]`. Add `[tool.jupytext]` with
  `formats = "py:percent,ipynb"`. Confirm `data/`, `_harness/`, `.venv/` are ignored.
  Create `.claude/settings.json` containing `{"includeCoAuthoredBy": false}` so no commit
  carries a generated-by trailer.
- **TDD steps**: Configuration only, no production code, so no RED. Commit as `chore`.
- **Verification**: `uv run pytest --collect-only` prints `no tests ran` without a
  configuration error, and `git log -1 --format=%B` shows no trailer.
- **Exit criteria**: pytest resolves `tests/`; jupytext pairing configured.
- **Dependencies**: none
- **Risk**: Low

### Task 0.2 — Rubric registry
- **File**: `src/cv_mba/rubric.py`, `tests/test_rubric_coverage.py`, `evidence.yaml`
- **Action**: Define `RUBRIC: dict[str, str]` holding all 26 ids R1.1–R1.5, R2.1–R2.5,
  R3.1–R3.6, R4.1–R4.4, R5.1–R5.6, each mapped to the rubric line in English. Define
  `load_evidence(path) -> dict[str, Evidence]` where `Evidence` carries `artifact` (a repo
  path) and `anchor` (a literal string that must appear in that file). Define
  `missing_evidence(rubric, evidence, repo_root) -> list[str]` returning every id that is
  absent, points at a file that does not exist, or points at a file not containing its
  anchor.
- **TDD steps**:
  1. RED: `tests/test_rubric_coverage.py` asserts `missing_evidence(...) == []` against
     the real `evidence.yaml`. Run it. It must fail listing all 26 ids. Paste the output.
  2. GREEN: implement `rubric.py` so the failure is the honest 26-gap list rather than an
     import error.
  3. Commit: `test: add failing rubric coverage test` then `feat: implement rubric registry`
- **Verification**: `uv run pytest tests/test_rubric_coverage.py -v` → fails with exactly
  26 named ids.
- **Exit criteria**: The failure list is the full rubric. This red stays red until Phase 7.
- **Dependencies**: 0.1
- **Risk**: Low

### Task 0.3 — Eval file and baseline
- **File**: `.claude/evals/cv-mba.md`
- **Action**: Write the eval file per `rules/edd-discipline.md` format with C1–C5 and
  R1–R4 as specified above. Record the BASELINE section: every capability eval fails
  because nothing is implemented.
- **TDD steps**: Documentation artefact. No code.
- **Verification**: File exists, every eval has a grader and a threshold, baseline section
  is filled with the date and the pre-implementation failure state.
- **Exit criteria**: EDD DEFINE and BASELINE phases complete before any model code.
- **Dependencies**: none
- **Risk**: Low

---

## Phase 1 — Data and metrics

Implements: F1.2 groundwork, and the A3 duplicate defence.

### Task 1.1 — Per-class metrics
- **File**: `src/cv_mba/metrics.py`, `tests/test_metrics.py`
- **Action**: `per_class_accuracy`, `per_class_recall`, `macro_f1`, `confusion_matrix`,
  all operating on integer label arrays, all typed, no torch dependency.
- **TDD steps**:
  1. RED: hand-build a 3×3 confusion matrix with known values; assert recall per class
     matches hand-computed numbers; assert a class with zero support raises
     `ZeroSupportError` rather than returning NaN.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_metrics.py -v` → all pass.
- **Exit criteria**: Zero-support raises. A silent NaN in per-class recall is the exact
  reporting defect A4.1 exists to diagnose, so it must be impossible here.
- **Dependencies**: 0.1
- **Risk**: Low

### Task 1.2 — ELPV loader and stratified split
- **File**: `src/cv_mba/data.py`, `tests/test_data.py`
- **Action**: `elpv_index(root) -> DataFrame` resolving `src/elpv_dataset/data/labels.csv`
  and raising with every searched path if absent. Binary label at `p >= 0.5`.
  `stratified_split(df, by, test_size, seed)` stratifying jointly on label and wafer type.
- **TDD steps**:
  1. RED: assert index length 2,624; assert binary counts 1,803 functional and 821
     defective; assert the split preserves the label×type proportions within one
     percentage point; assert no file path appears in both halves.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_data.py -k elpv -v`
- **Exit criteria**: Counts match the verified facts in SPEC.md exactly.
- **Dependencies**: 1.1
- **Risk**: Low

### Task 1.3 — A3 loader with the duplicate trap as an assertion
- **File**: `src/cv_mba/data.py`, `tests/test_data.py`
- **Action**: `a3_index(root) -> DataFrame` enumerating exactly the seven class
  directories `bike, cars, cats, dogs, flowers, horses, human` at the top level, never
  recursing. Raise `DuplicateCorpusError` naming the path if a nested `data/data`
  directory is reachable from the root given.
- **TDD steps**:
  1. RED: assert 1,803 rows across 7 classes; assert per-class counts 365, 420, 202, 202,
     210, 202, 202; assert pointing the loader at a root containing `data/data` raises.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_data.py -k a3 -v`
- **Exit criteria**: 3,606 can never be loaded. An 80/20 split over the duplicated archive
  would place the same image in train and validation and make validation accuracy
  fiction; the loader makes that unreachable rather than documented.
- **Dependencies**: 1.2
- **Risk**: Low

### Task 1.4 — COVID subsample and ADS-16 index
- **File**: `src/cv_mba/data.py`, `tests/test_data.py`
- **Action**: `covid_subsample(root, seed)` returning exactly 840 Normal, 240 Viral
  Pneumonia, 120 COVID, stratified, seeded, reproducing the failed project's split.
  `ads16_index(root)` returning 2,697 rows tagged `partition` as `ads` (301) or
  `corpus` (2,396), with the 16 product category folder for ads rows.
- **TDD steps**:
  1. RED: assert the three COVID counts; assert the same seed gives the same file list
     twice; assert ADS-16 partition counts 301 and 2,396.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_data.py -v`
- **Exit criteria**: The reproduced baseline is byte-identical across runs, so the
  with-and-without comparison in F5.4 is valid.
- **Dependencies**: 1.3
- **Risk**: Low

---

## Phase 2 — Transformer core

Implements: F2.1, F2.3, F2.4, F3.1. Rubric R2.1, R2.3, R2.4, R3.1.
No GPU. Every line of this phase is CPU-testable, which is what makes "testable PyTorch
modules" in rubric 2.1 a fact rather than a claim.

### Task 2.1 — Scaled dot-product attention
- **File**: `src/cv_mba/attention.py`, `tests/test_attention.py`
- **Action**: `ScaledDotProductAttention(nn.Module)` returning `(output, weights)`.
- **TDD steps**:
  1. RED: output shape equals value shape; weight rows sum to 1 within 1e-6; a masked
     position receives exactly zero weight; removing the `1/sqrt(d_k)` scaling changes
     the output, proving the scaling is applied rather than decorative.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_attention.py -k scaled -v`
- **Exit criteria**: Four assertions pass. The scaling test is the one that distinguishes a
  correct implementation from a plausible one.
- **Dependencies**: 0.2
- **Risk**: Low

### Task 2.2 — Multi-head attention
- **File**: `src/cv_mba/attention.py`, `tests/test_attention.py`
- **Action**: `MultiHeadAttention(nn.Module)` with independent per-head query, key and
  value projections and a concatenated output projection.
- **TDD steps**:
  1. RED: concatenated width equals `n_heads * d_head`; perturbing one head's projection
     weights changes that head's output slice and leaves the others bitwise identical,
     which is the executable proof of "independent projections per head" in rubric 2.1;
     `n_heads=1` reproduces single-head attention exactly.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_attention.py -v`
- **Exit criteria**: Head independence is proven by test, not by reading the code.
- **Dependencies**: 2.1
- **Risk**: Low

### Task 2.3 — Transformer encoder block
- **File**: `src/cv_mba/encoder.py`, `tests/test_encoder.py`
- **Action**: `TransformerEncoderBlock` with two-layer feedforward, LayerNorm and
  residual connections.
- **TDD steps**:
  1. RED: zeroing the sublayer output returns the input unchanged, proving the residual
     path exists; LayerNorm output has zero mean and unit variance per token within 1e-5;
     the feedforward contains exactly two `nn.Linear` layers.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_encoder.py -v`
- **Dependencies**: 2.2
- **Risk**: Low

### Task 2.4 — Positional encoding
- **File**: `src/cv_mba/positional.py`, `tests/test_positional.py`
- **Action**: `SinusoidalPositionalEncoding` and `LearnedPositionalEncoding`, both added
  to the token sequence.
- **TDD steps**:
  1. RED: encoding is added not concatenated, so sequence width is unchanged; and the
     central test — permute the token order, then assert the encoder output **changes**
     with positional encoding and is **identical** without it. That single test is the
     written claim of rubric 2.4 turned into evidence.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_positional.py -v`
- **Exit criteria**: The permutation test passes both directions.
- **Dependencies**: 2.3
- **Risk**: Low

### Task 2.5 — Vision Transformer
- **File**: `src/cv_mba/vit.py`, `tests/test_vit.py`
- **Action**: `PatchEmbedding` and `VisionTransformer(image) -> logits`, built on the
  encoder block from 2.3.
- **TDD steps**:
  1. RED: patch count equals `(H/P) * (W/P)`; sequence length equals patches + 1; the CLS
     token sits at index 0 and appears in `model.parameters()`, proving it is learnable;
     forward returns `(batch, n_classes)`; `return_attention=True` yields per-layer,
     per-head weights with the right shape.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_vit.py -v`
- **Exit criteria**: Attention weights are retrievable, which F2.2 and F3.2 both need.
- **Dependencies**: 2.4
- **Risk**: Medium — the attention-return path is where shape bugs hide.

### Task 2.6 — Determinism harness
- **File**: `src/cv_mba/determinism.py`, `tests/test_determinism.py`
- **Action**: `enable_determinism(seed)` setting torch, numpy and python seeds,
  `torch.use_deterministic_algorithms(True)` and `cudnn.benchmark = False`.
- **TDD steps**:
  1. RED: two forward passes over the same batch produce bitwise-identical tensors. Mark
     `@pytest.mark.gpu`. When CUDA is absent, skip with an explicit reason string — never
     a bare skip, which `rules/anti-cheat-discipline.md` treats as hiding a failure.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_determinism.py -v` on CPU shows a skip with
  its reason; the Colab run shows a pass.
- **Exit criteria**: This is eval R4. C1 and C2 only run once if it passes on the T4.
- **Dependencies**: 2.5
- **Risk**: Medium — deterministic algorithms are unavailable for a few operations and
  raise when selected. If that happens, the fix is changing the operation, not disabling
  the flag.

---

## Phase 3 — A3 notebook, transfer learning

Implements: F1.1, F1.2. Rubric R1.1, R1.2. First notebook, chosen as the pipeline
shakedown.

### Task 3.1 — Colab install path proof
- **File**: `notebooks/A3_cnn_kaggle.py` (jupytext percent format)
- **Action**: Cell 1 installs the package from GitHub. Cell 2 asserts torch is still
  `2.11.0+cu128` and reports `nvidia-smi`. Kaggle credentials read from
  `google.colab.userdata`, never pasted.
- **Verification**: Run on Colab T4. Paste the version line. If torch shows `+cpu`, stop
  everything: pip has replaced the GPU build and the dependency strategy needs rework
  before any further work.
- **Exit criteria**: Package imports on Colab and torch is unchanged.
- **Dependencies**: Phase 2 pushed to `main`
- **Risk**: **High, and it is the cheapest high risk to retire.** This is the single
  assumption the whole delivery rests on and it costs five minutes to test.

### Task 3.2 — Feature extraction training
- **File**: `notebooks/A3_cnn_kaggle.py`
- **Action**: Load pretrained backbone, replace head with 7 outputs, freeze backbone,
  train the head only, one run. Loss and accuracy curves per epoch. Per-class and global
  accuracy via `cv_mba.metrics`.
- **Verification**: Notebook executes end to end; per-class table has 7 rows; curves
  rendered.
- **Exit criteria**: R1.1 and R1.2 evidence exists and is registered in `evidence.yaml`.
- **Dependencies**: 3.1, 1.3
- **Risk**: Low

---

## Phase 4 — A2 notebook, CLIP

Implements: F4.2, F4.3, and the printed comparison behind F4.4. Rubric R4.2, R4.3.
No training.

### Task 4.1 — CLIP embedding and concept ranking
- **File**: `src/cv_mba/clip_search.py`, `tests/test_clip_search.py`,
  `notebooks/A2_clip_ads16.py`
- **Action**: `embed_images`, `embed_texts`, `rank_concepts(embeddings, concepts,
  threshold)`. At least 20 concept descriptions. Ranking reported separately for the
  `ads` partition and the `corpus` partition.
- **TDD steps**:
  1. RED: cosine similarity symmetric and within [-1, 1]; embeddings L2-normalised to
     within 1e-5; ranking unchanged under row permutation of the corpus.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_clip_search.py -v`, then the Colab run
  produces both rankings plus the top-5 visualisation with example images.
- **Exit criteria**: Eval C1 passes — the two partitions' top-5 lists differ by at least
  two entries. If they do not differ, that is a finding worth reporting, not a bug to hide.
- **Dependencies**: 2.6, 1.4
- **Risk**: Medium — the threshold must be justified in writing, and justifying a number
  chosen after seeing the results is the kind of reasoning the rubric is testing.

### Task 4.2 — Semantic search and the BERT tokenisation comparison
- **File**: `notebooks/A2_clip_ads16.py`
- **Action**: At least 8 queries spanning generic to specific and concrete to abstract,
  top-5 images each, indexed over all 2,697 images. Plus a cell printing CLIP tokens
  beside BERT tokens for the same sentence, showing padding tokens and the attention mask
  explicitly.
- **Verification**: Eval C2 — top-1 correct for the 4 concrete queries.
- **Exit criteria**: R4.3 evidence, and R4.4's written claim backed by printed output
  rather than assertion.
- **Dependencies**: 4.1
- **Risk**: Low

---

## Phase 5 — A1 notebook, Vision Transformers

Implements: F2.2, F3.2, F3.3. Rubric R2.2, R3.2, R3.3, R3.6.

### Task 5.1 — ViT from scratch on ELPV
- **File**: `notebooks/A1_vision_transformers.py`
- **Action**: Train `cv_mba.vit.VisionTransformer` on the ELPV binary task, three seeds.
  Greyscale replicated to three channels. Attention maps from at least one head, overlaid
  on a p=1.0 cell with a visible crack and on a misclassified cell, so the write-up covers
  a success and a failure.
- **Verification**: Three seeds complete; attention maps rendered; written reading of
  what the head weights.
- **Exit criteria**: R2.2 and R3.2 evidence.
- **Dependencies**: 3.1, 1.2, 2.5
- **Risk**: Medium — a from-scratch ViT on 2,100 training images is expected to perform
  poorly. That is the result, not a failure. Do not tune it into looking good; the
  comparison in 5.2 is the deliverable.

### Task 5.2 — Pretrained ViT fine-tune and comparison
- **File**: `notebooks/A1_vision_transformers.py`
- **Action**: Fine-tune a pretrained ViT on the same split, head replaced, same three
  seeds. One comparison table: per-seed macro-F1 for both models, the delta, and the
  spread.
- **Verification**: Eval C3 — `min delta > 0` across all three seeds and `mean delta >
  0.05`. Paste the per-seed table.
- **Exit criteria**: R3.3 and R3.6 evidence, with seed variance visible.
- **Dependencies**: 5.1
- **Risk**: Medium — ELPV is far from ImageNet, so the transfer benefit may be smaller
  than the literature suggests. If `min delta` lands at or below zero, that is a genuine
  result about domain distance and it belongs in the report under R1.4 and R3.5, with the
  eval recorded as failed rather than quietly re-run until it passes.

---

## Phase 6 — A4.1 notebook, GAN and the case study

Implements: F5.2, F5.3, F5.4. Rubric R5.2, R5.3, R5.4. Highest risk, scheduled last.

### Task 6.1 — Reproduce the failed baseline
- **File**: `notebooks/A4_estudo_caso_raio_x.py`
- **Action**: ResNet-18 without pretraining, SGD, fixed learning rate 0.01, 15 epochs, no
  augmentation, unstratified 80/20, global accuracy only — exactly as the brief describes
  the failed project. Then report what that reporting hides: per-class recall.
- **Verification**: Training and validation accuracy are in the region the brief reports,
  and COVID recall is far below global accuracy.
- **Exit criteria**: The failure is reproduced and measured, which makes every later
  number a comparison rather than an assertion.
- **Dependencies**: 1.4, 1.1
- **Risk**: Medium — exact numbers will not match the brief and are not expected to. The
  claim being reproduced is the pattern, not the digits.

### Task 6.2 — Conditional GAN
- **File**: `src/cv_mba/gan.py`, `tests/test_gan.py`, `notebooks/A4_estudo_caso_raio_x.py`
- **Action**: `ConditionalGenerator`, `ConditionalDiscriminator`, adversarial training
  loop conditioned on class label.
- **TDD steps**:
  1. RED: generator output shape matches discriminator input; changing the conditioning
     label changes the output; on a crafted batch the two losses move in opposite
     directions.
  2. GREEN: implement.
  3. Commit pair.
- **Verification**: `uv run pytest tests/test_gan.py -v`, then eval C4 at `pass@3 >= 0.90`.
- **Dependencies**: 6.1
- **Risk**: **High.** This is the only component that can fail for reasons no code review
  catches.

### Task 6.3 — Instability diagnosis and mitigation
- **File**: `notebooks/A4_estudo_caso_raio_x.py`
- **Action**: A diversity metric computed inside the training loop across epochs, so mode
  collapse is detected rather than guessed at. Apply one mitigation. Show the metric
  before and after.
- **Verification**: Two curves, one mitigation, measured improvement.
- **Exit criteria**: R5.3 evidence. Collapse is an expected path here, not an error state.
- **Dependencies**: 6.2
- **Risk**: High

### Task 6.4 — Recall impact of synthetic augmentation
- **File**: `notebooks/A4_estudo_caso_raio_x.py`
- **Action**: Freeze one synthetic set from the trained generator. Run three seeded
  classifier pairs, with and without, on the same split.
- **Verification**: Eval C5 — `min recall delta > 0` across the three pairs. Paste all six
  recall numbers, not the mean.
- **Exit criteria**: R5.4 evidence. A single-run recall delta is exactly the sloppiness
  this activity asks the student to diagnose, so six numbers are the point.
- **Dependencies**: 6.3
- **Risk**: Medium — synthetic augmentation may fail to help. If it does, that is a
  publishable result and the report says so. Do not re-roll seeds until a positive delta
  appears.

---

## Phase 7 — Report and submission

### Task 7.1 — Written rubric lines
- **File**: `report/report.md`
- **Action**: Write the thirteen rubric lines that exist only as prose: R1.3, R1.4, R1.5,
  R2.4, R2.5, R3.4, R3.5, R4.1, R4.4, R5.1, R5.5, R5.6, plus the per-activity problem
  definition, decisions, results and critical analysis the brief requires.
- **Verification**: Each section contains its anchor string from `evidence.yaml`.
- **Dependencies**: Phases 3–6
- **Risk**: Low

### Task 7.2 — AI usage citation
- **File**: `report/report.md`
- **Action**: A section naming the AI tools used and how. The brief states that omitting
  this is academic misconduct. Independent of git trailers, which stay clean.
- **Dependencies**: 7.1
- **Risk**: Low — and the cheapest possible way to lose the whole submission.

### Task 7.3 — Notebook headers from measured runs
- **File**: all four notebooks
- **Action**: Write actual peak memory from `torch.cuda.max_memory_allocated()` and actual
  wall clock from a completed run into each notebook header.
- **Verification**: Numbers trace to a run in this session.
- **Exit criteria**: No estimated number anywhere in a header. An estimate is an
  unverified claim, which `rules/verification-discipline.md` forbids.
- **Dependencies**: Phases 3–6
- **Risk**: Low

### Task 7.4 — Close the rubric red
- **File**: `evidence.yaml`
- **Action**: Fill all 26 entries with real artefact paths and anchors.
- **Verification**: `uv run pytest tests/test_rubric_coverage.py -v` → passes for the
  first time since Task 0.2. Paste the output.
- **Exit criteria**: The test that has been red since the first commit goes green on
  earned evidence.
- **Dependencies**: 7.1, 7.2, 7.3
- **Risk**: Low

### Task 7.5 — Submission package
- **File**: `daniel_moreira_deep-learning-and-vision_computer-vision.pdf`, submission ZIP
- **Action**: Generate `.ipynb` from the paired `.py` sources. Execute each one clean on
  Colab T4 top to bottom. Build the PDF. Assemble the ZIP.
- **Verification**: Eval R2 — `jupyter nbconvert --execute` completes for all four
  notebooks with no edits beyond mounting Drive.
- **Exit criteria**: Every notebook runs clean from a fresh runtime.
- **Dependencies**: 7.4, and the professor's answer on the discipline string
- **Risk**: Medium — a notebook that runs in a warm session and fails from cold is the
  classic submission failure. The verification must start from a fresh runtime.

---

## Task summary

| Phase | Tasks | GPU needed | Risk |
|---|---|---|---|
| 0 Scaffold | 3 | no | Low |
| 1 Data and metrics | 4 | no | Low |
| 2 Transformer core | 6 | only 2.6 | Low–Medium |
| 3 A3 notebook | 2 | yes | **High at 3.1** |
| 4 A2 CLIP | 2 | yes | Medium |
| 5 A1 ViT | 2 | yes | Medium |
| 6 A4.1 GAN | 4 | yes | **High** |
| 7 Report | 5 | no | Low–Medium |

19 of 28 tasks need no GPU. Phases 0 through 2 are thirteen tasks that run entirely on
your Windows machine with the CPU torch build, which is why the environment was pinned
that way.
