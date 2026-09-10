"""
Trains the proposed model: ResNet50 + ViT + attention-based fusion.

Also implements requirement #7: trains and compares three imbalance
handling strategies —
    Model A: standard Cross-Entropy
    Model B: class-weighted Cross-Entropy
    Model C: Focal Loss
so that the final report can honestly state whether imbalance-aware
training actually helps minority classes (rather than assuming it does).

The single best-performing variant (by macro F1 on validation) becomes the
project's "proposed hybrid" model used in the demo/inference scripts.

Usage:
    python scripts/train_hybrid.py --config configs/config.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES
from src.losses import get_loss_function
from src.models.hybrid_model import HybridResNetViT
from src.preprocessing import build_dataloaders
from src.training.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.metrics import metrics_to_flat_row
from src.utils.seed import get_device, set_seed
from src.utils.visualization import plot_fusion_weights


def load_splits(cfg):
    splits_dir = cfg["dataset"]["splits_dir"]
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    return train_df, val_df, test_df


def build_hybrid(cfg):
    m_cfg = cfg["model"]
    return HybridResNetViT(
        num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
        vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
        fusion_type="attention", dropout=m_cfg["dropout"], pretrained=m_cfg["pretrained"],
    )


def measure_branch_contributions(model, test_loader, device):
    """Runs a pass over the test set and averages the model's own learned
    ResNet/ViT contribution weights (requirement #12) — not manually set."""
    import torch
    model.eval()
    alphas_r, alphas_v = [], []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            model(images)
            ar, av = model.get_branch_contributions()
            if ar is not None:
                alphas_r.append(ar)
                alphas_v.append(av)
    if not alphas_r:
        return None, None
    return sum(alphas_r) / len(alphas_r), sum(alphas_v) / len(alphas_v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--losses", nargs="+", default=["ce", "weighted_ce", "focal"],
                         help="Which imbalance-handling variants to train (Models A/B/C)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device()

    train_df, val_df, test_df = load_splits(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, cfg)

    loss_label = {"ce": "Model_A_CE", "weighted_ce": "Model_B_WeightedCE", "focal": "Model_C_Focal"}
    results_rows = []
    best_overall = {"val_f1": -1, "name": None}
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)

    for loss_name in args.losses:
        run_name = f"hybrid_{loss_label.get(loss_name, loss_name)}"
        print(f"\n{'='*60}\nTraining proposed hybrid model — loss={loss_name} ({run_name})\n{'='*60}")

        model = build_hybrid(cfg).to(device)
        criterion = get_loss_function(loss_name, train_df["label_idx"].tolist(),
                                       cfg["model"]["num_classes"], cfg)

        history = train_model(model, train_loader, val_loader, criterion, cfg, device,
                               model_name=run_name, two_stage=True)

        metrics = evaluate_model(model, test_loader, device, CLASS_NAMES)
        row = metrics_to_flat_row(metrics, run_name, history["training_time_sec"])
        results_rows.append(row)

        alpha_r, alpha_v = measure_branch_contributions(model, test_loader, device)
        if alpha_r is not None:
            print(f"Learned branch contributions -> ResNet: {alpha_r:.3f}, ViT: {alpha_v:.3f}")
            plot_fusion_weights(
                alpha_r, alpha_v,
                os.path.join(cfg["project"]["output_dir"], "figures", f"{run_name}_fusion_weights.png"),
            )

        with open(os.path.join(cfg["project"]["output_dir"], f"{run_name}_history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)
        with open(os.path.join(cfg["project"]["output_dir"], f"{run_name}_test_metrics.json"), "w") as f:
            json.dump({k: v for k, v in metrics.items()
                       if k not in ("y_true", "y_pred", "y_prob", "filepaths")},
                      f, indent=2, default=str)

        if history["best_val_metric"] > best_overall["val_f1"]:
            best_overall = {"val_f1": history["best_val_metric"], "name": run_name}

    results_df = pd.DataFrame(results_rows)
    out_csv = os.path.join(cfg["project"]["output_dir"], "hybrid_imbalance_comparison.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nImbalance-strategy comparison saved to {out_csv}")

    if best_overall["name"] is not None:
        best_ckpt = os.path.join(cfg["project"]["checkpoint_dir"], f"{best_overall['name']}_best.pth")
        final_ckpt = os.path.join(cfg["project"]["checkpoint_dir"], "best_model.pth")
        import shutil
        shutil.copyfile(best_ckpt, final_ckpt)
        print(f"\nBest overall proposed-model variant: {best_overall['name']} "
              f"(val macro F1 = {best_overall['val_f1']:.4f})")
        print(f"Copied to {final_ckpt} — this is the checkpoint used by predict.py / app.py.")

        with open(os.path.join(cfg["project"]["output_dir"], "best_model_info.json"), "w") as f:
            json.dump(best_overall, f, indent=2)


if __name__ == "__main__":
    main()
