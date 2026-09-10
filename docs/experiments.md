# Experiments

All numbers referenced below are placeholders for structure only until the pipeline is actually
run — see `outputs/model_comparison.csv`, `outputs/ablation_results.csv`, and
`outputs/hybrid_imbalance_comparison.csv` for the real, generated values after training. Nothing
in this repository hard-codes a result.

## Baselines (requirement #9)

| # | Model | Purpose |
|---|-------|---------|
| 1 | ResNet18 | Lightweight CNN-only baseline |
| 2 | ResNet50 | Stronger CNN-only baseline; also the local branch used in the proposed model |
| 3 | ViT (`vit_base_patch16_224`) | Transformer-only baseline; also the global branch used in the proposed model |
| 4 | ResNet50 + ViT (simple concatenation) | Shows what naive multi-branch fusion gets you, without the learned gating mechanism |
| — | **Proposed**: ResNet50 + ViT + attention fusion + imbalance-aware loss | The full contribution |

Trained via `scripts/train_baselines.py` (baselines 1-4) and `scripts/train_hybrid.py` (proposed).

## Ablation Study (requirement #20)

| Experiment | Isolates |
|---|---|
| 1. ResNet50 only | Local-feature-only performance ceiling |
| 2. ViT only | Global-feature-only performance ceiling |
| 3. ResNet50 + ViT, simple concatenation | Effect of *combining* branches without gating |
| 4. ResNet50 + ViT, attention fusion | Effect of the *learned gating mechanism* specifically |
| 5. Attention fusion + class-weighted CE | Effect of basic imbalance handling on top of the architecture |
| 6. Attention fusion + Focal Loss | Effect of a stronger imbalance-handling loss |

Run via `scripts/generate_report.py`, which reuses already-trained checkpoints from steps 1-2
where possible and trains any missing variant explicitly. Results and a bar chart are written to
`outputs/ablation_results.csv` and `outputs/ablation_comparison.png`.

**Important:** the ablation is designed to let the *data* answer whether attention fusion beats
concatenation, and whether imbalance-aware losses beat plain CE — not to assume it. If a component
does not help, `outputs/ablation_results.csv` will show that plainly, and the final report should
say so rather than only reporting favorable comparisons.

## Metrics Protocol (requirement #33)

For every trained model, on the held-out test split:

- Accuracy
- Precision / Recall / F1 — both **macro** (unweighted mean across classes, most informative
  under imbalance) and **weighted** (weighted by class support)
- Sensitivity (= macro recall), Specificity, FPR, FNR, NPV — computed via one-vs-rest
  decomposition of the multiclass confusion matrix
- Matthews Correlation Coefficient (MCC) — a single balanced summary statistic robust to class
  imbalance
- ROC-AUC — one-vs-rest per class, plus macro/weighted average; only computed when every class is
  present in the evaluated split's ground truth (otherwise reported as `None` with an explanatory
  note, never fabricated)

All of the above are implemented in `src/utils/metrics.py::compute_all_metrics`.

## Statistical Validation (requirement #34)

`configs/config.yaml -> statistical_validation` supports multi-seed runs (mean ± std across
seeds). This is **disabled by default** because repeating full training 3+ times is expensive on
free Colab; if you enable it and have the compute budget, results should be reported as
`mean ± std`, and if only a single run is feasible, that limitation should be stated explicitly
rather than presenting a single run as if it were a robust average.

## Cross-Dataset / External Validation (requirement #22)

Not run by default. HAM10000's 7-class taxonomy (akiec, bcc, bkl, df, mel, nv, vasc) does not
line up cleanly with every other free dermatoscopic dataset's class definitions (e.g. PH2 uses a
3-class melanoma/atypical-nevus/common-nevus scheme). If you want to attempt external validation,
document the class-mapping decisions explicitly in this file before trusting any resulting
numbers — do not silently remap or drop classes to force a comparison.

## Robustness Test (requirement #23)

Run via `scripts/robustness_test.py`. Samples real test images and re-runs real inference under
mild brightness change, contrast change, small rotation, and mild Gaussian noise, then reports
what fraction of predictions stay the same as the unperturbed prediction. Results saved to
`outputs/robustness_test_results.csv` and `outputs/figures/robustness_flip_rates.png`.
