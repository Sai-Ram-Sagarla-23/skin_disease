#!/usr/bin/env python
"""
Simple inference script (requirement #29).

Usage:
    python predict.py --image path/to/image.jpg

Output:
    Predicted Disease: <class>
    Confidence: <value>
    Explanation images saved to: outputs/prediction/
"""
import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset import CLASS_FULL_NAMES, CLASS_NAMES
from src.explainability.attention_visualization import get_cls_attention_map, overlay_attention
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import get_val_test_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_model(cfg, ckpt_path, device):
    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=False,
    ).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No trained model found at {ckpt_path}. Train the model first with "
            f"`python scripts/train_hybrid.py`, or point --checkpoint to a valid .pth file."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a skin lesion image")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to models/checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/prediction")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")

    model = load_model(cfg, ckpt_path, device)
    transform = get_val_test_transform(cfg)

    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.resnet_branch.layer4)
    cam, pred_idx = gradcam.generate(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    attn_weights = model.vit_branch.get_last_attention()
    attn_map = None
    if attn_weights is not None:
        attn_map = get_cls_attention_map(attn_weights, cfg["data"]["image_size"])

    print(f"Predicted Disease: {CLASS_FULL_NAMES[pred_class]} ({pred_class})")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nFull class probabilities:")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:6s} ({CLASS_FULL_NAMES[c]}): {probs[i]*100:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    orig = np.array(image.resize((cfg["data"]["image_size"], cfg["data"]["image_size"]))) / 255.0
    cam_overlay = overlay_heatmap(orig, cam)
    attn_overlay = overlay_attention(orig, attn_map) if attn_map is not None else orig

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(cam_overlay); axes[1].set_title("ResNet Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(attn_overlay); axes[2].set_title("ViT Attention"); axes[2].axis("off")
    fig.suptitle(f"Prediction: {pred_class}  |  Confidence: {confidence*100:.2f}%")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "explanation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nExplanation images saved to: {args.output_dir}/")
    print("\nDISCLAIMER: This is an academic research prototype, not a medical "
          "diagnosis tool. Consult a qualified dermatologist for any real "
          "skin concern.")


if __name__ == "__main__":
    main()
