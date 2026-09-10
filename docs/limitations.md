# Limitations

This project is designed to be honest about what it can and cannot guarantee. Please read this
before presenting results to faculty, reviewers, or any clinical audience.

## 1. Not a medical diagnosis tool

This is an academic research prototype. It has not been validated for clinical use, has not gone
through regulatory review, and must never be used to make or influence real medical decisions.
Every script and interface in this project that shows a prediction also shows this disclaimer.

## 2. Patient/leakage-prevention key

HAM10000's public metadata does not include a distinct `patient_id` field. This project uses
`lesion_id` as the leakage-prevention grouping key, since multiple images of the *same lesion*
(often from the same patient/visit) share a `lesion_id`. This prevents the most common and severe
form of leakage in this dataset (the same lesion appearing in both train and test), but it is not
a guarantee against a *different* lesion from the *same patient* appearing across splits, since
patient identity is not directly recorded in the public release. This is a known, documented
limitation of HAM10000 itself, not something this codebase can work around.

## 3. Dataset size and class imbalance are real constraints

HAM10000 has ~10,000 images across 7 classes, heavily skewed toward `nv` (melanocytic nevi). Even
with class-weighted loss and Focal Loss, minority classes (`df`, `vasc`, and to a lesser extent
`akiec`) will likely have wider confidence intervals and less reliable per-class metrics than the
majority class, purely due to sample size — not a flaw in the fusion architecture.

## 4. Compute constraints on free Colab

- Full training of all baselines + the proposed model + ablation study, end to end, will typically
  take multiple hours even on a free Colab T4 GPU, and considerably longer on CPU. Reduce
  `epochs_stage1` / `epochs_stage2` / `batch_size` / `image_size` in `configs/config.yaml` if you
  need a faster run, understanding that this may reduce final accuracy.
- Free Colab sessions can disconnect after a period of inactivity or after a fixed time limit.
  Checkpointing (`models/checkpoints/*_best.pth`) means you don't lose a fully-completed model,
  but a run interrupted mid-epoch will need to be restarted from the last saved checkpoint if you
  add resume logic, which is not implemented by default in this initial build.

## 5. Statistical validation

Multi-seed repeated runs (`configs/config.yaml -> statistical_validation`) are supported but
**disabled by default** due to the compute cost on free Colab. Unless you explicitly enable and
run them, all reported metrics come from a **single training run per configuration**, not a
mean ± std over multiple seeds. This should be stated explicitly in any write-up of results.

## 6. Cross-dataset / external validation

Not run by default (see `docs/experiments.md`). Different public dermatoscopic datasets use
different class taxonomies; attempting external validation without carefully reconciling class
definitions would produce misleading numbers.

## 7. Segmentation stage

The optional lesion-segmentation preprocessing stage described in the original project brief
(requirement #21) is not implemented in this initial build, to keep the primary pipeline runnable
on free Colab without additional heavy dependencies. This is a documented, honest omission rather
than a stub that pretends to do something it doesn't.

## 8. Explainability caveats

- Grad-CAM highlights regions that influenced the ResNet branch's prediction, computed from real
  gradients at the last convolutional block. It is a widely used but imperfect proxy for "what the
  model looked at" and should not be over-interpreted as ground truth reasoning.
- The ViT attention map uses the CLS token's attention to patch tokens in the *last* transformer
  block, averaged over heads — a standard, common visualization, but attention weights are not a
  fully faithful explanation of transformer decision-making (a well-known caveat in the
  interpretability literature).

## 9. Grad-CAM / attention are computed independently

The current explainability code visualizes each branch's own attention/activations. It does not
implement a rigorous joint explanation that accounts for the *learned fusion weights* (α_r, α_v)
when combining the two heatmaps into one image. The learned scalar/channel fusion weights are
visualized separately (`outputs/figures/*_fusion_weights.png`) rather than blended pixel-wise into
the Grad-CAM/attention overlays.

## 10. No claim of state-of-the-art performance

Whether the proposed attention-fusion hybrid actually outperforms the baselines and ablation
variants is an empirical question answered by `outputs/model_comparison.csv` and
`outputs/ablation_results.csv` after you run the pipeline — not something claimed in advance in
this documentation.
