"""
Full hybrid architecture:

    Input Image
        |
  +-----+-----+
  |           |
 ResNet50    ViT
  |           |
 Local       Global
 Features    Features
  |           |
  +-----+-----+
        |
Feature Projection
        |
Attention-Based Fusion  (or ConcatFusion for the ablation baseline)
        |
Fused Features
        |
Dropout / MLP Head
        |
    Softmax
        |
Disease Prediction
"""
from typing import Optional

import torch
import torch.nn as nn

from .fusion import AttentionFusion, ConcatFusion
from .resnet_model import ResNetBranch
from .vit_model import ViTBranch


class HybridResNetViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        resnet_backbone: str = "resnet50",
        vit_backbone: str = "vit_base_patch16_224",
        fusion_dim: int = 512,
        fusion_type: str = "attention",   # "attention" or "concat"
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.resnet_branch = ResNetBranch(backbone=resnet_backbone, pretrained=pretrained)
        self.vit_branch = ViTBranch(backbone=vit_backbone, pretrained=pretrained)

        if fusion_type == "attention":
            self.fusion = AttentionFusion(
                resnet_dim=self.resnet_branch.feat_dim,
                vit_dim=self.vit_branch.feat_dim,
                fusion_dim=fusion_dim,
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(
                resnet_dim=self.resnet_branch.feat_dim,
                vit_dim=self.vit_branch.feat_dim,
                fusion_dim=fusion_dim,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.fusion_type = fusion_type
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, num_classes),
        )

        # populated on each forward() call for explainability / logging
        self.last_alpha_resnet: Optional[torch.Tensor] = None
        self.last_alpha_vit: Optional[torch.Tensor] = None
        self.last_resnet_feature_map: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------
    # Staged transfer learning (requirement #10)
    # ---------------------------------------------------------------
    def freeze_backbones(self) -> None:
        self.resnet_branch.freeze_backbone(True)
        self.vit_branch.freeze_backbone(True)

    def unfreeze_stage2(self) -> None:
        """Stage 2: partially unfreeze the deepest layers of each backbone."""
        self.resnet_branch.unfreeze_last_block()
        self.vit_branch.unfreeze_last_block()

    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_pooled, resnet_feature_map = self.resnet_branch(x)
        vit_pooled, _ = self.vit_branch(x)

        fused, alpha_r, alpha_v = self.fusion(resnet_pooled, vit_pooled)

        self.last_alpha_resnet = alpha_r
        self.last_alpha_vit = alpha_v
        self.last_resnet_feature_map = resnet_feature_map

        logits = self.classifier(fused)
        return logits

    def get_branch_contributions(self):
        """Returns mean scalar contribution of each branch for the most
        recent forward pass (averaged over channels/batch if channel-wise
        gating was used). Used for requirement #12 visualization."""
        if self.last_alpha_resnet is None:
            return None, None
        alpha_r = self.last_alpha_resnet.mean().item()
        alpha_v = self.last_alpha_vit.mean().item()
        return alpha_r, alpha_v
