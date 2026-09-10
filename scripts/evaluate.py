"""
Aggregates baseline + hybrid results into the final model comparison table
(requirement #19), and generates confusion matrix / ROC / PR curves for the
best (proposed) model on the real test set.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES
from src.preprocessing import build_dataloaders
from src.models.hybrid_model import HybridResNetViT
from src.training.evaluate import evaluate_model
from src.utils.seed import get_device, set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_precision_recall_curves,
    plot_roc_curves,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device()
    out_dir = cfg["project"]["output_dir"]

    # --- Combine baseline + hybrid comparison tables (requirement #19) ---
    frames = []
    baseline_csv = os.path.join(out_dir, "baseline_results.csv")
    hybrid_csv = os.path.join(out_dir, "hybrid_imbalance_comparison.csv")
    if os.path.exists(baseline_csv):
        frames.append(pd.read_csv(baseline_csv))
    else:
        print("WARNING: baseline_results.csv not found — run scripts/train_baselines.py first.")
    if os.path.exists(hybrid_csv):
        frames.append(pd.read_csv(hybrid_csv))
    else:
        print("WARNING: hybrid_imbalance_comparison.csv not found — run scripts/train_hybrid.py first.")

    if not frames:
        print("No results available yet. RESULTS WILL BE GENERATED AFTER TRAINING.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join(out_dir, "model_comparison.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined model comparison saved to {combined_path}")
    print(combined.to_string(index=False))

    plot_model_comparison_bar(combined, "Macro F1",
                               os.path.join(out_dir, "figures", "model_comparison_macro_f1.png"))
    plot_model_comparison_bar(combined, "Accuracy",
                               os.path.join(out_dir, "figures", "model_comparison_accuracy.png"))

    # --- Detailed evaluation of the best (proposed) model, if trained ---
    best_ckpt = os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(best_ckpt):
        print("\nbest_model.pth not found yet — run scripts/train_hybrid.py first. "
              "RESULTS WILL BE GENERATED AFTER TRAINING.")
        return

    splits_dir = cfg["dataset"]["splits_dir"]
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    _, _, test_loader = build_dataloaders(train_df, val_df, test_df, cfg)

    m_cfg = cfg["model"]
    model = HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    metrics = evaluate_model(model, test_loader, device, CLASS_NAMES)

    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES,
                           os.path.join(out_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES,
                           os.path.join(out_dir, "confusion_matrix_normalized.png"), normalize=True)

    if metrics["y_prob"] is not None:
        aucs = plot_roc_curves(metrics["y_true"], metrics["y_prob"], CLASS_NAMES,
                                os.path.join(out_dir, "roc_curve.png"))
        plot_precision_recall_curves(metrics["y_true"], metrics["y_prob"], CLASS_NAMES,
                                      os.path.join(out_dir, "precision_recall_curve.png"))
        print(f"ROC AUC per class: {aucs}")

    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        pd.DataFrame([metrics["per_class"]]).T.to_csv(f)

    print("\nFinal evaluation figures saved to outputs/. Done.")


if __name__ == "__main__":
    main()
