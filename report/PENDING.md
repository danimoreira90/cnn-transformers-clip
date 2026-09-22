# What the report still needs, and which notebook produces it

Everything here requires a Colab T4 run. Nothing here is blocked on a decision.

Run order is cheapest first, so a problem surfaces before an expensive session is spent.

| # | Notebook | Produces | Rubric lines it closes |
|---|---|---|---|
| 1 | `A3_cnn_kaggle.ipynb` | training curves, per-class and global accuracy, confusion matrix, the 3,606-against-1,803 duplicate count | R1.1, R1.2 |
| 2 | `A2_clip_ads16.ipynb` | concept ranking on both partitions, top-5 concepts with example images, 10 search queries with results, the similarity distribution plot | R4.2, R4.3 |
| 3 | `A1_vision_transformers.ipynb` | per-seed macro-F1 for both models, the comparison table, validation curves, attention maps on a found and a missed defect | R2.2, R3.2, R3.3, R3.6 |
| 4 | `A4_estudo_caso_raio_x.ipynb` | the reproduced baseline's per-class recall, split-drift table, GAN diversity curves before and after mitigation, generated samples, six COVID recall values | R5.3, R5.4, and the measured figures inside R5.1 |

Each notebook writes its numbers to CSV files beside itself, so the tables can be built from
data rather than retyped:

- `a3_training_curves.csv`, `a3_per_class_accuracy.csv`
- `a2_concept_ranking.csv`, `a2_search_results.csv`
- `a1_seed_results.csv`
- `a4_gan_diversity.csv`, `a4_recall_comparison.csv`

Each notebook's final cell prints its measured runtime and peak GPU memory. Those two
numbers go into that notebook's header, replacing nothing — the headers currently say the
figures are printed by the final cell, which is true until the run happens and the measured
values are written in.

## Also outstanding

- **Fill `evidence.yaml`** for every rubric line each run closes. The coverage check is red
  by design until all 26 are backed; it currently reports 10 gaps, and those 10 are exactly the lines the four runs above close.
- **Assemble the submission**: generate `.ipynb` from the paired `.py` sources, execute each
  from a cold runtime, build the PDF as
  `daniel_moreira_deep-learning-and-vision_computer-vision.pdf`, and package
  `daniel_moreira_deep-learning-and-vision_pd.ZIP` with a README.txt at its root.
