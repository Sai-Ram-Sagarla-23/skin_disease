"""
Class-weighted Cross Entropy loss.

Weights are always derived from actual class frequencies in the *training*
split (never hand-picked), using the standard inverse-frequency scheme:

    w_c = N / (num_classes * n_c)

where N is total training samples and n_c is the number of training
samples in class c. This is the "Model B" (Class-weighted CE) referenced
in requirement #7, and is also used to build FocalLoss's alpha term.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from real label counts."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1  # avoid div-by-zero for an absent class
    n_total = counts.sum()
    weights = n_total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_class_weighted_ce_loss(labels: List[int], num_classes: int) -> nn.Module:
    weights = compute_class_weights(labels, num_classes)
    return nn.CrossEntropyLoss(weight=weights)


def class_distribution_report(labels: List[int], class_names: Dict[int, str]) -> Dict[str, int]:
    """Real (not fabricated) class counts, used both for reporting and for
    computing the loss weights above."""
    counts = np.bincount(labels, minlength=len(class_names))
    return {class_names[i]: int(counts[i]) for i in range(len(class_names))}
