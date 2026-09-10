"""
Trains Baselines 1-4 (requirement #9):
  - ResNet18
  - ResNet50
  - ViT
  - ResNet50 + ViT (simple concatenation, no attention fusion)

Each model's best checkpoint, training history, and test-set metrics are
saved. These become the comparison rows in outputs/model_comparison.csv
alongside the proposed hybrid model (trained separately by train_hybrid.py).

Usage:
    python scripts/train_baselines.py --config configs/config.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES, HAM10000Dataset
from src.losses import get_loss_function
from src.models.hybrid_model import HybridResNetViT
from src.models.resnet_model import ResNetClassifier
from src.models.vit_model import ViTClassifier
from src.preprocessing import build_dataloaders
from src.training.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.metrics import metrics_to_flat_row
from src.utils.seed import get_device, set_seed


def load_splits(cfg):
    splits_dir = cfg["dataset"]["splits_dir"]
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    return train_df, val_df, test_df


def build_model(name, cfg):
    m_cfg = cfg["model"]
    if name == "resnet18":
        return ResNetClassifier("resnet18", m_cfg["num_classes"], m_cfg["pretrained"], m_cfg["dropout"])
    elif name == "resnet50":
        return ResNetClassifier("resnet50", m_cfg["num_classes"], m_cfg["pretrained"], m_cfg["dropout"])
    elif name == "vit":
        return ViTClassifier(m_cfg["vit_backbone"], m_cfg["num_classes"], m_cfg["pretrained"], m_cfg["dropout"])
    elif name == "resnet50_vit_concat":
        return HybridResNetViT(
            num_classes=m_cfg["num_classes"], resnet_backbone=m_cfg["resnet_backbone"],
            vit_backbone=m_cfg["vit_backbone"], fusion_dim=m_cfg["fusion_dimension"],
            fusion_type="concat", dropout=m_cfg["dropout"], pretrained=m_cfg["pretrained"],
        )
    else:
        raise ValueError(f"Unknown baseline: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Subset of baselines to train, e.g. --models resnet18 resnet50")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device()

    train_df, val_df, test_df = load_splits(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, cfg)

    baselines = args.models or cfg["baselines"]
    results_rows = []
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)

    for name in baselines:
        print(f"\n{'='*60}\nTraining baseline: {name}\n{'='*60}")
        model = build_model(name, cfg).to(device)
        criterion = get_loss_function("ce", train_df["label_idx"].tolist(),
                                       cfg["model"]["num_classes"], cfg)

        two_stage = hasattr(model, "freeze_backbones") or name in ("resnet18", "resnet50", "vit")
        # For plain classifiers, emulate two-stage via their .branch freeze methods
        if not hasattr(model, "freeze_backbones"):
            model.freeze_backbones = lambda m=model: m.branch.freeze_backbone(True)
            model.unfreeze_stage2 = lambda m=model: m.branch.unfreeze_last_block()

        history = train_model(model, train_loader, val_loader, criterion, cfg, device,
                               model_name=name, two_stage=True)

        metrics = evaluate_model(model, test_loader, device, CLASS_NAMES)
        row = metrics_to_flat_row(metrics, name, history["training_time_sec"])
        results_rows.append(row)

        with open(os.path.join(cfg["project"]["output_dir"], f"{name}_history.json"), "w") as f:
            json.dump({k: v for k, v in history.items()}, f, indent=2, default=str)
        with open(os.path.join(cfg["project"]["output_dir"], f"{name}_test_metrics.json"), "w") as f:
            json.dump({k: v for k, v in metrics.items()
                       if k not in ("y_true", "y_pred", "y_prob", "filepaths")},
                      f, indent=2, default=str)

        print(f"\n{name} TEST RESULTS: Accuracy={row['Accuracy']:.4f} "
              f"MacroF1={row['Macro F1']:.4f} AUC={row['AUC (macro)']}")

    results_df = pd.DataFrame(results_rows)
    out_csv = os.path.join(cfg["project"]["output_dir"], "baseline_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nAll baseline results saved to {out_csv}")


if __name__ == "__main__":
    main()
