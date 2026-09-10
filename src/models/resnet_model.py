"""
ResNet feature-extraction branch (local / spatial representation).

Provides both a standalone classifier (used as Baseline 1 / 2) and a
feature-extractor mode used inside the hybrid model.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetBranch(nn.Module):
    """Wraps a torchvision ResNet as a feature extractor.

    forward() returns:
        - pooled_features: (B, feat_dim) global-average-pooled vector
        - feature_map: (B, C, H, W) last conv block output, used for Grad-CAM
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = tv_models.resnet50(weights=weights)
            self.feat_dim = 2048
        elif backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tv_models.resnet18(weights=weights)
            self.feat_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        # Keep everything up to (and including) the last conv block ("layer4")
        # so we can expose the spatial feature map for Grad-CAM.
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3,
        )
        self.layer4 = net.layer4  # exposed separately -> Grad-CAM target layer
        self.avgpool = net.avgpool

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.stem.parameters():
            p.requires_grad = not freeze
        for p in self.layer4.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_block(self) -> None:
        """Stage-2 fine-tuning: unfreeze only layer4 (the last conv block)."""
        for p in self.layer4.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        feature_map = self.layer4(x)          # (B, feat_dim, H, W) — Grad-CAM source
        pooled = self.avgpool(feature_map).flatten(1)  # (B, feat_dim)
        return pooled, feature_map


class ResNetClassifier(nn.Module):
    """Standalone ResNet classifier used as Baseline 1 (ResNet18) / Baseline 2 (ResNet50)."""

    def __init__(self, backbone: str = "resnet50", num_classes: int = 7,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.branch = ResNetBranch(backbone=backbone, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.branch.feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.branch(x)
        return self.classifier(pooled)

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """For Grad-CAM: returns the last conv feature map."""
        _, feature_map = self.branch(x)
        return feature_map
