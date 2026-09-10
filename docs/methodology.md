# Methodology

## 1. Problem Framing

Automated skin-lesion classification from dermatoscopic images is typically approached with a
single CNN backbone (e.g. ResNet). CNNs excel at capturing **local** texture and edge patterns
via convolution, but their limited receptive field per layer makes it harder to relate distant
regions of an image (e.g. an irregular border on one side of a lesion vs. color variation on the
other) without very deep stacks. Vision Transformers (ViT) use self-attention across all patches
from the first layer, giving every patch direct access to **global** context — at the cost of
weaker inductive bias for local texture, especially with limited training data.

This project's central hypothesis: **combining both, with a learned, adaptive fusion mechanism,
should outperform either backbone alone** on a task where both fine local texture (asymmetry,
border irregularity) and global structure (overall lesion shape, surrounding skin context)
matter. This hypothesis is tested, not assumed — see `docs/experiments.md` and
`outputs/model_comparison.csv` for the actual result.

## 2. Architecture

```
                    Input Image (224x224x3)
                             |
                    IMAGE PREPROCESSING
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
              RESNET50                  ViT (vit_base_patch16_224)
          (ImageNet pretrained)     (ImageNet pretrained)
                 |                       |
          2048-d pooled vector    768-d CLS token vector
          (local/spatial)         (global/contextual)
                 |                       |
                 +-----------+-----------+
                             |
                    FEATURE PROJECTION
              (separate Linear+LayerNorm+GELU
               per branch -> shared dim d=512)
                             |
                  ATTENTION-BASED FUSION
                             |
                     FUSED FEATURES (512-d)
                             |
                    DROPOUT / MLP HEAD
                             |
                    SOFTMAX (7 classes)
```

## 3. Attention-Based Feature Fusion — Mathematical Formulation

Let `r ∈ R^2048` be the ResNet50 pooled feature and `v ∈ R^768` the ViT CLS-token feature for one
image.

**Step 1 — Projection.** Each branch is projected into a shared dimensionality `d = 512`:

```
R = GELU(LayerNorm(W_r r + b_r))    R ∈ R^d
V = GELU(LayerNorm(W_v v + b_v))    V ∈ R^d
```

**Step 2 — Joint context.** Concatenate the projected vectors:

```
H = concat(R, V) ∈ R^{2d}
```

**Step 3 — Gating scores.** A small MLP produces unnormalized per-channel gating scores for each
branch:

```
[S_r, S_v] = W_2 · tanh(W_1 H + b_1) + b_2      S_r, S_v ∈ R^d
```

**Step 4 — Normalization.** Softmax is applied *per channel, across the two branches* (not across
channels), yielding channel-wise gates that sum to 1 for every feature dimension:

```
[α_r, α_v] = softmax_over_branches([S_r, S_v])   α_r, α_v ∈ R^d,  α_r + α_v = 1 (elementwise)
```

**Step 5 — Fusion.**

```
F = α_r ⊙ R + α_v ⊙ V         (elementwise product + sum)
```

`F ∈ R^512` is passed to a two-layer MLP classification head with dropout.

This is a **channel-wise gated attention fusion**: rather than a single scalar weight per image,
every one of the 512 fused feature channels gets its own learned mixture of "how much ResNet vs.
ViT information to use here." The scalar "ResNet contribution 0.62 / ViT contribution 0.38"
figures reported in the demo notebook are the **mean** of `α_r` / `α_v` over channels and/or batch
— a summary of the underlying per-channel gates, not a separately hand-set number.

A simpler `ConcatFusion` module (plain concatenation + projection, no gating) is also implemented
and trained as an ablation/baseline specifically to demonstrate whether the attention mechanism is
actually contributing value.

## 4. Class Imbalance Handling

HAM10000 is heavily imbalanced (the `nv` — melanocytic nevi — class dominates; `df` and `vasc` are
rare). Three loss functions are trained and compared under identical architecture/hyperparameters:

- **Model A** — standard Cross-Entropy (no imbalance handling; establishes the baseline).
- **Model B** — class-weighted Cross-Entropy, with weights `w_c = N / (num_classes * n_c)` computed
  from actual training-split class counts.
- **Model C** — Focal Loss (`γ=2`), with the same class-frequency-derived `α` weights, additionally
  down-weighting easy/well-classified examples.

Per-class recall and macro F1 are the primary metrics for judging whether imbalance-aware training
actually helps minority classes — accuracy alone would be misleading on this dataset.

## 5. Transfer Learning Strategy

Both backbones start from ImageNet-pretrained weights (free, standard `torchvision` /
`timm` weights — no proprietary checkpoints).

- **Stage 1 (frozen backbone):** Both `ResNetBranch` and `ViTBranch` backbones are frozen; only
  the fusion module and classification head are trained. This lets the fusion mechanism adapt
  quickly and cheaply.
- **Stage 2 (partial fine-tuning):** The last convolutional block of ResNet50 (`layer4`) and the
  last transformer block + final norm of ViT are unfrozen and fine-tuned at a much lower learning
  rate, while everything before that stays frozen.

This two-stage approach keeps peak GPU memory and per-epoch compute low enough to be practical on
a free Colab T4 GPU, while still allowing the deepest, most task-specific layers to adapt.

## 6. Leakage-Resistant Evaluation

See `docs/limitations.md` for the exact leakage-prevention key used (HAM10000's `lesion_id`) and
its caveats. Splitting is done at the lesion/patient level, never at the individual-image level,
so that multiple photographs of the same lesion cannot appear in both training and test sets.
