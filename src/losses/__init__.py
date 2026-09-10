from typing import Dict, List

import torch.nn as nn

from .focal_loss import FocalLoss
from .weighted_loss import compute_class_weights


def get_loss_function(loss_name: str, train_labels: List[int], num_classes: int, cfg: Dict) -> nn.Module:
    """Factory used by all training scripts. `loss_name` in {"ce", "weighted_ce", "focal"}."""
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_ce":
        weights = compute_class_weights(train_labels, num_classes)
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "focal":
        weights = compute_class_weights(train_labels, num_classes)
        gamma = cfg["training"].get("focal_gamma", 2.0)
        return FocalLoss(alpha=weights, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
