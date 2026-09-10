"""
Loads the raw HAM10000 metadata, runs dataset analysis (requirement #5),
performs the patient/lesion-wise leakage-resistant split (requirement #6),
and saves everything needed by downstream training scripts.

Usage:
    python scripts/prepare_dataset.py --config configs/config.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES, analyze_dataset, load_metadata, patient_wise_split
from src.utils.seed import set_seed
from src.utils.visualization import plot_class_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    print("Loading HAM10000 metadata and resolving image paths...")
    df = load_metadata(cfg)

    print("\n=== Dataset Analysis ===")
    report = analyze_dataset(df, cfg)
    print(f"Number of images: {report['num_images']}")
    print(f"Number of classes: {report['num_classes']}")
    print(f"Class distribution: {report['class_distribution']}")
    print(f"Corrupt images found: {report['num_corrupt']}")
    if report["num_corrupt"] > 0:
        print(f"  -> Corrupt files: {report['corrupt_images']}")
    if "image_dimensions" in report:
        print(f"Image dimensions: {report['image_dimensions']}")
    print(f"Duplicate filenames: {report['duplicate_filenames']}")
    print(f"Note: {report['duplicate_check_note']}")

    # Drop any corrupt images before splitting
    if report["num_corrupt"] > 0:
        df = df[~df["filepath"].isin(report["corrupt_images"])].reset_index(drop=True)

    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    plot_class_distribution(
        report["class_distribution"],
        os.path.join(cfg["project"]["output_dir"], "figures", "class_distribution.png"),
    )

    print("\n=== Patient/Lesion-wise Split (leakage prevention) ===")
    train_df, val_df, test_df = patient_wise_split(df, cfg, seed=cfg["project"]["seed"])

    splits_dir = cfg["dataset"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    with open(os.path.join(cfg["project"]["output_dir"], "dataset_report.json"), "w") as f:
        json.dump({k: v for k, v in report.items() if k != "corrupt_images"}, f, indent=2, default=str)

    print(f"\nSplits saved to {splits_dir}/. Dataset report saved to "
          f"{cfg['project']['output_dir']}/dataset_report.json")
    print("Ready for training. Run: python scripts/train_baselines.py")


if __name__ == "__main__":
    main()
