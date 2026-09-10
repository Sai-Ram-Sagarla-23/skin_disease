"""
Robustness test (requirement #23): checks whether the trained model's
predictions remain reasonably stable under mild, realistic image
perturbations. Runs real inference on real test images under each
perturbation — nothing here is simulated or guessed.

Usage:
    python scripts/robustness_test.py --config configs/config.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageEnhance
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import get_val_test_transform
from src.utils.seed import get_device, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_perturbation(image: Image.Image, kind: str) -> Image.Image:
    if kind == "original":
        return image
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.3)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.3)
    if kind == "rotated":
        return image.rotate(15, expand=False, fillcolor=(128, 128, 128))
    if kind == "noise":
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 12, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_samples", type=int, default=30,
                         help="Number of real test images to evaluate robustness on")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device()

    ckpt_path = os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("best_model.pth not found. RESULTS WILL BE GENERATED AFTER TRAINING.")
        return

    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_df = pd.read_csv(os.path.join(cfg["dataset"]["splits_dir"], "test.csv"))
    sample_df = test_df.sample(n=min(args.n_samples, len(test_df)), random_state=cfg["project"]["seed"])

    transform = get_val_test_transform(cfg)
    perturbations = ["original", "brightness", "contrast", "rotated", "noise"]

    rows = []
    for _, row in sample_df.iterrows():
        base_img = Image.open(row["filepath"]).convert("RGB")
        preds = {}
        for kind in perturbations:
            perturbed = apply_perturbation(base_img, kind)
            tensor = transform(perturbed).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            preds[kind] = {"pred": CLASS_NAMES[int(probs.argmax())], "confidence": float(probs.max())}

        stable = all(preds[k]["pred"] == preds["original"]["pred"] for k in perturbations)
        rows.append({
            "filepath": row["filepath"],
            "true_class": CLASS_NAMES[int(row["label_idx"])],
            **{f"{k}_pred": v["pred"] for k, v in preds.items()},
            **{f"{k}_conf": v["confidence"] for k, v in preds.items()},
            "prediction_stable_across_all_perturbations": stable,
        })

    df = pd.DataFrame(rows)
    out_dir = cfg["project"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "robustness_test_results.csv")
    df.to_csv(out_csv, index=False)

    stability_rate = df["prediction_stable_across_all_perturbations"].mean()
    print(f"Prediction stability across all 4 perturbations: {stability_rate*100:.1f}% "
          f"of {len(df)} sampled test images")
    print(f"Full per-image results saved to {out_csv}")

    # Summary bar chart: how often the model's prediction flips under each perturbation
    flip_rates = {}
    for kind in perturbations[1:]:
        flip_rates[kind] = (df[f"{kind}_pred"] != df["original_pred"]).mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(flip_rates.keys()), list(flip_rates.values()), color="#C44E52")
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_title("Robustness: Prediction Flip Rate Under Perturbation")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "robustness_flip_rates.png"), dpi=150)
    plt.close(fig)

    print("Robustness figure saved to outputs/figures/robustness_flip_rates.png")


if __name__ == "__main__":
    main()
