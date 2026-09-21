# Eval: MBA Computer Vision — CNN, Transformers and CLIP

**Date:** 2026-09-20
**Feature:** Four model-backed activities — CLIP zero-shot retrieval, Vision Transformer
classification, CNN transfer learning, conditional GAN augmentation.
**Spec:** `SPEC.md`
**Plan:** `PLAN.md`

**Ship criteria:** as adjusted in SPEC.md and argued there, not defaulted:

```
C1, C2   pass@1 against a committed golden snapshot, valid only while R4 passes
C3       min delta across 3 seeds > 0  AND  mean delta > 0.05
C4       pass@3 >= 0.90
C5       one frozen synthetic set, 3 seeded classifier pairs, min delta > 0, pass@3 >= 0.90
R1..R4   pass^3 = 1.00
```

The deviation from the standard `capability pass@3 >= 0.90` is documented in SPEC.md
under "Deviation from the EDD Iron Law". It is argued from system determinism and
experiment design, not from implementation difficulty. C3's gate is strictly harder to
pass than the default, because it requires a win on every seed rather than on 90% of runs.

---

## Capability Evals

| ID | Description | Grader | Threshold |
|----|-------------|--------|-----------|
| C1 | Concept ranking separates advertising content from participant photographs. Top-5 concepts over the 301-image `ads` partition differ from top-5 over the 2,396-image `corpus` partition | code | at least 2 of 5 entries differ |
| C2 | Text-to-image search retrieves what the query describes for unambiguous queries | code, against a hand-labelled key committed before the run | top-1 correct on 4 of the 4 concrete queries |
| C3 | Pretrained ViT outperforms from-scratch ViT on ELPV binary defect detection | code | min macro-F1 delta across seeds 0, 1, 2 > 0 AND mean > 0.05 |
| C4 | Conditional GAN output responds to the conditioning label. COVID-conditioned and Normal-conditioned samples are separable by the frozen Task 6.1 baseline classifier | code | separation above chance by a margin fixed before the run |
| C5 | Synthetic augmentation raises COVID-class recall against the reproduced 840/240/120 baseline | code | min recall delta > 0 across 3 seeded classifier pairs sharing one frozen synthetic set and one split |

### Notes on graders

**C2's answer key is committed before the search runs.** Writing the key afterwards would
make the eval unfalsifiable, which `rules/edd-discipline.md` lists as an EDD violation.
Four of the eight queries are concrete enough to have a defensible ground truth; the other
four are abstract by design, because rubric 4.3 asks for the specificity range and the
interesting result there is where CLIP interprets unexpectedly. Those four are analysed
in the report and are deliberately not graded.

**C4's margin is fixed before the run** for the same reason. A margin chosen after seeing
the separation is not a threshold, it is a description.

**C5 holds the synthetic set fixed.** Retraining the GAN per seed would measure GAN
reproducibility and augmentation benefit simultaneously and answer neither. The generated
set is the treatment; only classifier seeds vary.

---

## Regression Evals

| ID | Description | Grader | Threshold |
|----|-------------|--------|-----------|
| R1 | Package imports and the full unit suite passes on CPU | code — `uv run pytest -q` | pass^3 = 1.00 |
| R2 | All four notebooks execute top to bottom on a cold Colab T4 with no edits | code — `jupyter nbconvert --execute` from a fresh runtime | pass^3 = 1.00 |
| R3 | Every rubric id has a real artefact containing its anchor | code — `uv run pytest tests/test_rubric_coverage.py` | pass^3 = 1.00 |
| R4 | CLIP embeddings are bitwise reproducible within a session under the determinism flags | code — `uv run pytest tests/test_determinism.py` on GPU | pass^3 = 1.00 |

**R4 is load-bearing.** C1 and C2 are permitted a single run only while R4 passes. GPU
floating point is not deterministic by default: cuDNN selects algorithms by benchmarking
and some reductions accumulate in thread-completion order, so results can differ in the
final bits. A concept ranking sorts similarity scores, and two concepts within 1e-6 of
each other swap places when those bits move. If R4 fails, C1 and C2 revert to three runs
and the golden snapshot is void.

---

## Baseline

**Date:** 2026-09-20
**State:** Pre-implementation. No model code exists.

| Eval | Result | Failure signature |
|---|---|---|
| C1 | FAIL | `cv_mba.clip_search` does not exist |
| C2 | FAIL | `cv_mba.clip_search` does not exist |
| C3 | FAIL | `cv_mba.vit` does not exist |
| C4 | FAIL | `cv_mba.gan` does not exist |
| C5 | FAIL | no baseline classifier, no generator |
| R1 | FAIL | no test suite yet |
| R2 | FAIL | no notebooks yet |
| R3 | FAIL | 26 of 26 rubric ids have no evidence |
| R4 | FAIL | `cv_mba.determinism` does not exist |

capability score: 0 of 5 — FAIL, pre-implementation
regression score: 0 of 4 — FAIL, pre-implementation

This baseline is honest rather than ceremonial. Every line names the missing module that
causes the failure, so a later pass is traceable to something built rather than to an eval
quietly loosened.

---

## Release Snapshot

Filled after the gate is passed. Left empty until then.

- Date:
- capability score:
- regression score:
- CLIP checkpoint and `transformers` version:
- torch version and GPU:
- Seeds used:
- Per-seed C3 deltas:
- Per-pair C5 recall values, all six:

---

## Recorded failures

Any eval that fails and stays failed is recorded here with its number, not deleted and not
re-run until it passes. Two are live possibilities and both are results rather than bugs:

- **C3 may fail on `min delta`.** ELPV electroluminescence images are far from ImageNet
  photographs. If transfer benefit is smaller than the literature suggests, that is a
  finding about domain distance and it belongs in the report under rubrics 1.4 and 3.5.
- **C5 may fail.** Synthetic augmentation does not always help minority-class recall. A
  negative delta, reported with all six numbers, is a stronger contribution to rubric 5.6's
  improvement plan than a positive delta obtained by re-rolling seeds.
