"""
Focal Loss (Lin et al., 2017) for multiclass classification with class
imbalance. Down-weights easy, well-classified examples so the model spends
more learning capacity on hard / minority-class examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

alpha_t is the class weight (from actual class frequencies, never guessed),
gamma controls how strongly easy examples are down-weighted.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)
        log_p_t = (log_probs * targets_one_hot).sum(dim=-1)

        focal_term = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        loss = -focal_term * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
