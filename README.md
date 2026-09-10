# Hybrid ResNet50 + Vision Transformer for Skin Disease Classification

**An Attention-Based Hybrid Deep Learning Framework Combining ResNet50 and Vision Transformer for
Skin Disease Classification**

A free, open-source, reproducible research project. No paid APIs, no API keys, no paid cloud
services, no proprietary software. Runs on free Google Colab (GPU) and on CPU-only machines
(slower).

> ⚠️ **This is an academic research prototype, not a medical diagnosis tool.** Do not use it to
> make or influence real medical decisions.

---

## Problem Statement

Skin cancer and other dermatological conditions are traditionally diagnosed via visual inspection
and dermatoscopy, which is subjective and dependent on clinician experience. Automated
classification from dermatoscopic images can act as a decision-support aid, but most existing deep
learning approaches rely on CNN-only architectures with a limited receptive field for modeling
long-range spatial relationships within a lesion image.

## Motivation

Combine the complementary strengths of convolutional networks (strong local/texture features) and
Vision Transformers (global, long-range context via self-attention), fused through a mechanism
that *learns* how much to trust each branch per image, rather than a fixed or naively concatenated
combination.

## Objectives

1. Implement a hybrid ResNet50 + ViT architecture with a genuine attention-based fusion module.
2. Handle class imbalance explicitly and compare imbalance-handling strategies empirically.
3. Prevent data leakage via lesion/patient-wise splitting.
4. Provide real explainability (Grad-CAM + ViT attention) for every prediction.
5. Establish honest baselines and an ablation study so the architecture's actual contribution is
   demonstrated with evidence, not assumed.
6. Keep the entire pipeline free, reproducible, and runnable on free Google Colab.

## Existing System (typical prior work)

Single-backbone CNN classifiers (often ResNet or DenseNet variants) fine-tuned on HAM10000 or
similar datasets, usually evaluated with plain accuracy, frequently without explicit leakage
prevention or class-imbalance handling.

## Drawbacks of Existing System

- CNN-only backbones struggle to relate distant regions of a lesion image without very deep
  stacks.
- Class imbalance is often ignored or handled only via oversampling, without a controlled
  comparison of alternatives.
- Random (non-leakage-resistant) train/test splits can silently inflate reported accuracy when a
  dataset contains multiple images of the same lesion/patient.
- Predictions are frequently reported without any explainability, making them hard to trust or
  audit.

## Proposed System

A dual-branch ResNet50 + ViT architecture with a learned, channel-wise attention-gated fusion
module (see `docs/methodology.md` for the full mathematical formulation), trained with
class-imbalance-aware losses, evaluated on a lesion-wise leakage-resistant split, and explained via
Grad-CAM and ViT attention visualization for every prediction.

## Research Gap

Prior hybrid CNN+Transformer skin-lesion papers often (a) use simple concatenation rather than a
learned fusion mechanism, (b) do not rigorously separate the contribution of the fusion mechanism
from the contribution of simply having two backbones (i.e., no concatenation ablation), and/or (c)
do not report leakage-resistant splitting. This project directly ablates each of these.

## Proposed Contributions

1. Hybrid local-global feature extraction using ResNet50 and ViT.
2. Attention-based adaptive feature fusion (channel-wise gating, mathematically documented).
3. Explicit, compared handling of class imbalance (CE vs. weighted CE vs. Focal Loss).
4. Leakage-resistant (lesion-wise) dataset splitting.
5. Explainable predictions via Grad-CAM and Transformer attention, for every demo/prediction.
6. Comprehensive baseline (4 models) and ablation (6 experiments) comparison.
7. Evaluation beyond accuracy: macro/weighted F1, sensitivity, specificity, MCC, ROC-AUC.
8. Optional external validation and robustness testing hooks.
9. Computationally efficient, two-stage transfer learning implementation, feasible on free Colab.

These are framed as an engineering/research contribution of this particular framework, not a claim
that hybrid CNN-Transformer fusion is a wholly new idea in the literature.

## Architecture

See [`docs/methodology.md`](docs/methodology.md) for the full diagram and the mathematical
formulation of the attention fusion module.

## Dataset

**HAM10000** ("Human Against Machine with 10000 training images"), Tschandl, Rosendahl & Kittler
(2018) — a free, publicly available dermatoscopic image dataset of 7 diagnostic categories:
`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

- Official free source (no account required): Harvard Dataverse,
  DOI [10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Free-account mirror: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

`scripts/download_dataset.py` attempts automatic download from both sources (no payment, no
hard-coded credentials) and, if neither is reachable in your environment, prints exact manual
placement instructions instead of failing silently.

## Installation

```bash
git clone <this-repo-url>
cd skin_disease_hybrid
pip install -r requirements.txt
```

No API keys or `.env` files are required anywhere in this project.

## Google Colab Instructions

1. Open `notebooks/skin_disease_hybrid_colab.ipynb` in Google Colab.
2. `Runtime -> Change runtime type -> GPU` (optional but recommended; a free T4 works).
3. Upload/clone this project folder into the Colab environment so `src/` and `scripts/` are
   importable (Cell 1b in the notebook checks for this).
4. Run all cells in order, top to bottom.

## Local Instructions

```bash
python scripts/download_dataset.py --output_dir data/raw
python scripts/prepare_dataset.py --config configs/config.yaml
python scripts/train_baselines.py --config configs/config.yaml
python scripts/train_hybrid.py --config configs/config.yaml
python scripts/evaluate.py --config configs/config.yaml
python scripts/generate_demo.py --config configs/config.yaml
python scripts/generate_report.py --config configs/config.yaml
python scripts/robustness_test.py --config configs/config.yaml
```

All hyperparameters live in `configs/config.yaml` — no source code edits should be needed for a
normal run.

## Training

Two-stage transfer learning (see `docs/methodology.md` section 5): Stage 1 trains only the fusion
module + classifier head with both backbones frozen; Stage 2 unfreezes the deepest layers of each
backbone and fine-tunes at a lower learning rate. Early stopping, cosine LR scheduling, gradient
clipping, and mixed precision (when CUDA is available) are all implemented in
`src/training/train.py`.

## Evaluation

`scripts/evaluate.py` combines baseline and hybrid results into `outputs/model_comparison.csv`,
and generates confusion matrices (raw + normalized), ROC curves, and precision-recall curves for
the best model on the real, held-out test split. See `docs/experiments.md` for the full metrics
protocol.

## Camera Input

The Streamlit app (`app.py`) supports two image input methods, side by side:

1. **📁 Upload Image** — the original file-uploader, unchanged.
2. **📷 Take Photo Using Camera** — browser-native capture via Streamlit's built-in
   `st.camera_input()`. No external camera API, no paid service, no API key.

How to use it:

1. Start the app: `streamlit run app.py`.
2. Under "Choose Image Source", select **📷 Take Photo Using Camera**.
3. Allow the browser's camera permission prompt when asked.
4. Capture the photo (the camera widget shows a live preview and a capture button).
5. Review the "Image Preview" section, then click **Analyze Image**.
6. View the prediction and confidence.
7. View the ResNet50 Grad-CAM heatmap.
8. View the Vision Transformer attention map.

Both input methods feed the **exact same** trained model, preprocessing pipeline, Grad-CAM
implementation, and ViT attention implementation (see `src/utils/image_input.py` and
`app.py::run_prediction_and_explain`) — there is only one prediction code path, regardless of
which input method produced the image.

**Camera predictions are experimental.** The model is trained on dermoscopic images (e.g.
HAM10000), which are captured with a specialized dermatoscope under controlled lighting and
magnification. A normal camera/webcam photograph looks meaningfully different — different scale,
lighting, focus, and lack of the polarized-light artifact removal a dermatoscope provides. The app
displays an in-context note about this whenever a camera-sourced image is analyzed, and camera
predictions should be treated as a research/demo convenience only, not a realistic stand-in for
dermoscopic evaluation.

If the browser denies camera access or the camera widget fails to initialize, the app shows:
*"Camera input is not available in this browser/session. Please use image upload instead."*
and the Upload Image path remains fully usable.

Every prediction can be paired with:
- **ResNet Grad-CAM** (`src/explainability/gradcam.py`) — real gradients w.r.t. the last conv block.
- **ViT attention map** (`src/explainability/attention_visualization.py`) — the real CLS-token
  attention from the last transformer block.
- The model's own **learned branch contribution weights** (α_r, α_v), visualized in
  `outputs/figures/*_fusion_weights.png`.

## Demonstration

`scripts/generate_demo.py` selects real test images (a confident correct prediction, a hard
example, a minority-class example, and a misclassified example where available) and saves full
explanation figures to `outputs/demonstrations/demo_01.png` … `demo_05.png`.

## Expected Output Files

```
outputs/
├── dataset_report.json
├── figures/
│   ├── class_distribution.png
│   ├── sample_images_per_class.png
│   ├── model_comparison_macro_f1.png
│   ├── model_comparison_accuracy.png
│   ├── *_fusion_weights.png
│   └── robustness_flip_rates.png
├── demonstrations/
│   └── demo_01.png … demo_05.png
├── error_analysis/
│   ├── error_analysis_summary.csv
│   ├── error_patterns_report.txt
│   └── error_01.png …
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── roc_curve.png
├── precision_recall_curve.png
├── training_loss.png / training_accuracy.png / validation_loss.png / validation_accuracy.png
├── baseline_results.csv
├── hybrid_imbalance_comparison.csv
├── model_comparison.csv
├── ablation_results.csv
├── ablation_comparison.png
├── robustness_test_results.csv
└── config.json

models/checkpoints/
├── <model_name>_best.pth   (one per trained model/variant)
└── best_model.pth          (the best-performing proposed-model variant)
```

## Limitations

See [`docs/limitations.md`](docs/limitations.md) for the full, honest list — including the
`lesion_id`-based (rather than true patient-ID-based) leakage prevention caveat, single-run
(non-multi-seed) reporting by default, and the segmentation stage not being implemented in this
initial build.

## Ethical Considerations

- Not a diagnostic tool; every user-facing script/interface carries a disclaimer.
- Dataset demographic composition (skin tone representation, geographic origin of cases in
  HAM10000) may not generalize equally to all populations — a known limitation of most public
  dermatoscopic datasets. Any real-world deployment would require careful fairness auditing across
  skin tones, which this project does not perform.
- Explainability outputs (Grad-CAM, attention maps) are aids for understanding model behavior, not
  proof of clinically valid reasoning.

## Future Work

- Implement the optional lesion-segmentation preprocessing stage.
- Add multi-seed statistical validation runs by default (compute budget permitting).
- Extend the joint explanation to blend Grad-CAM and ViT attention *weighted by the model's own
  learned fusion weights*, rather than visualizing each branch independently.
- Fairness/bias auditing across demographic subgroups, if such metadata becomes available.
- Formal external validation against a second, class-taxonomy-compatible free dataset.

---

## Final Summary

**What was implemented:** A complete, runnable, free, reproducible research pipeline — dataset
download/analysis/leakage-resistant splitting, 4 baselines, the proposed ResNet50+ViT
attention-fusion hybrid trained under 3 imbalance-handling strategies, a 6-experiment ablation
study, full metrics suite, confusion/ROC/PR curves, Grad-CAM + ViT attention explainability,
demonstration figure generation, error analysis, a robustness test, a CLI inference script
(`predict.py`), and an optional local Streamlit app (`app.py`).

**Dataset used:** HAM10000 (free, public, Harvard Dataverse / Kaggle mirror).

**Models implemented:** ResNet18, ResNet50, ViT (`vit_base_patch16_224`), ResNet50+ViT
(concatenation), and the proposed ResNet50+ViT+attention-fusion hybrid.

**Proposed architecture:** Dual-branch ResNet50 (local) + ViT (global) with a channel-wise
attention-gated fusion module — see `docs/methodology.md` for the full math.

**Research gaps addressed:** learned (not naive) fusion; leakage-resistant splitting; explicit,
compared imbalance handling; per-prediction explainability; honest baseline + ablation comparison.

**Leakage prevention:** lesion-wise (via `lesion_id`) splitting, with an assertion that no lesion
ID appears in more than one split — see `docs/limitations.md` for the caveat vs. true patient IDs.

**Class imbalance handling:** three loss functions compared empirically (CE / class-weighted CE /
Focal Loss), with weights derived from real training-split class counts.

**Explainability techniques:** Grad-CAM (ResNet branch) and CLS-token self-attention visualization
(ViT branch), plus the model's own learned per-branch contribution weights.

**Baseline models:** ResNet18, ResNet50, ViT, ResNet50+ViT (concatenation) — see
`scripts/train_baselines.py`.

**Ablation experiments:** 6 experiments isolating backbone choice, fusion mechanism, and
imbalance-handling strategy — see `scripts/generate_report.py`.

**Generated demonstration images:** `outputs/demonstrations/demo_01.png` … `demo_05.png`.

**Exact command to run the full project (after dataset download):**
```bash
python scripts/prepare_dataset.py && \
python scripts/train_baselines.py && \
python scripts/train_hybrid.py && \
python scripts/evaluate.py && \
python scripts/generate_demo.py && \
python scripts/generate_report.py
```

**Exact command to run inference:**
```bash
python predict.py --image path/to/image.jpg
```

**Remaining limitations:** see `docs/limitations.md` — most notably, no true patient-ID leakage
key is available in HAM10000's public release (lesion-wise splitting is used instead), single-run
(not multi-seed) reporting by default, and the optional segmentation stage is not implemented.
