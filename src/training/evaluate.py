"""
Evaluation loop: runs a trained model over a DataLoader and returns
predictions, probabilities, labels and filepaths — all real, nothing
simulated. Downstream code (metrics.py, visualization.py, scripts/) turns
this into confusion matrices, ROC curves, demo figures, etc.
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.metrics import compute_all_metrics


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for images, labels, paths in tqdm(loader, desc="inference", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.append(probs)
        all_paths.extend(list(paths))

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.concatenate(all_probs, axis=0),
        "filepaths": all_paths,
    }


def evaluate_model(model, loader: DataLoader, device: torch.device, class_names) -> Dict:
    """Full evaluation: inference + metrics.compute_all_metrics in one call."""
    inf = run_inference(model, loader, device)
    metrics = compute_all_metrics(
        inf["y_true"], inf["y_pred"], inf["y_prob"], class_names=class_names
    )
    metrics["y_true"] = inf["y_true"]
    metrics["y_pred"] = inf["y_pred"]
    metrics["y_prob"] = inf["y_prob"]
    metrics["filepaths"] = inf["filepaths"]
    return metrics
