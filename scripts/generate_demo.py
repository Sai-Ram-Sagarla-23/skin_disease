"""
Generates demonstration figures (requirement #14): for a handful of REAL
test-set images, shows the original image, ResNet Grad-CAM, ViT attention,
and the model's actual prediction + confidence. Nothing here is simulated —
every heatmap comes from a real forward/backward pass.

Selects, where available:
  - one correctly classified example
  - one difficult example (low softmax confidence)
  - one minority-class example
  - one incorrectly classified example

Usage:
    python scripts/generate_demo.py --config configs/config.yaml
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_FULL_NAMES, CLASS_NAMES, HAM10000Dataset
from src.explainability.attention_visualization import get_cls_attention_map, overlay_attention
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import get_val_test_transform
from src.utils.seed import get_device, set_seed


def select_demo_images(test_df: pd.DataFrame, y_true, y_pred, y_prob, n: int = 5):
    """Pick a diverse, real set of test images to demonstrate."""
    confidences = y_prob.max(axis=1)
    correct_mask = y_true == y_pred
    selected_idx = []

    # 1. A correctly classified, high-confidence example
    correct_idx = np.where(correct_mask)[0]
    if len(correct_idx) > 0:
        best = correct_idx[np.argmax(confidences[correct_idx])]
        selected_idx.append(best)

    # 2. A difficult (low-confidence) example
    remaining = [i for i in range(len(y_true)) if i not in selected_idx]
    if remaining:
        hardest = min(remaining, key=lambda i: confidences[i])
        selected_idx.append(hardest)

    # 3. A minority-class example (df or vasc are rarest in HAM10000)
    minority_classes = [CLASS_NAMES.index(c) for c in ("df", "vasc") if c in CLASS_NAMES]
    for mc in minority_classes:
        idxs = [i for i in range(len(y_true)) if y_true[i] == mc and i not in selected_idx]
        if idxs:
            selected_idx.append(idxs[0])
            break

    # 4. An incorrectly classified example, if any exist
    incorrect_idx = np.where(~correct_mask)[0]
    incorrect_idx = [i for i in incorrect_idx if i not in selected_idx]
    if len(incorrect_idx) > 0:
        selected_idx.append(incorrect_idx[0])

    # Fill up to n with random additional real examples if still short
    remaining = [i for i in range(len(y_true)) if i not in selected_idx]
    while len(selected_idx) < n and remaining:
        selected_idx.append(remaining.pop(0))

    return selected_idx[:n]


def make_demo_figure(image_path, true_label, pred_label, confidence, cam, attn_map, save_path):
    orig = np.array(Image.open(image_path).convert("RGB").resize((224, 224))) / 255.0

    cam_overlay = overlay_heatmap(orig, cam) if cam is not None else orig
    attn_overlay = overlay_attention(orig, attn_map) if attn_map is not None else orig

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(cam_overlay); axes[1].set_title("ResNet Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(attn_overlay); axes[2].set_title("ViT Attention"); axes[2].axis("off")

    correct = "✓ CORRECT" if true_label == pred_label else "✗ INCORRECT"
    fig.suptitle(
        f"True: {CLASS_FULL_NAMES[CLASS_NAMES[true_label]]} ({true_label})   |   "
        f"Predicted: {CLASS_FULL_NAMES[CLASS_NAMES[pred_label]]} ({pred_label})   |   "
        f"Confidence: {confidence*100:.2f}%   |   {correct}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device()
    out_dir = cfg["project"]["output_dir"]

    ckpt_path = os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("best_model.pth not found. RESULTS WILL BE GENERATED AFTER TRAINING. "
              "Run scripts/train_hybrid.py first.")
        return

    test_df = pd.read_csv(os.path.join(cfg["dataset"]["splits_dir"], "test.csv"))
    transform = get_val_test_transform(cfg)
    test_ds = HAM10000Dataset(test_df, transform=transform)

    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Run inference over the whole test set to select diverse examples
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label, _ = test_ds[i]
            logits = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            y_true.append(label)
            y_pred.append(int(probs.argmax()))
            y_prob.append(probs)
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    n_demo = cfg["explainability"]["num_demo_images"]
    selected = select_demo_images(test_df, y_true, y_pred, y_prob, n=n_demo)

    gradcam = GradCAM(model, model.resnet_branch.layer4)

    demo_dir = os.path.join(out_dir, "demonstrations")
    os.makedirs(demo_dir, exist_ok=True)

    for rank, idx in enumerate(selected, start=1):
        img_tensor, true_label, filepath = test_ds[idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        cam, pred_class = gradcam.generate(input_batch)
        attn_weights = model.vit_branch.get_last_attention()
        attn_map = None
        if attn_weights is not None:
            attn_map = get_cls_attention_map(attn_weights, cfg["data"]["image_size"])

        confidence = float(y_prob[idx].max())
        save_path = os.path.join(demo_dir, f"demo_{rank:02d}.png")
        make_demo_figure(filepath, true_label, int(y_pred[idx]), confidence, cam, attn_map, save_path)
        print(f"Saved {save_path}  (true={CLASS_NAMES[true_label]}, "
              f"pred={CLASS_NAMES[y_pred[idx]]}, conf={confidence*100:.1f}%)")

    print(f"\n{n_demo} demonstration figures saved to {demo_dir}/")


if __name__ == "__main__":
    main()
