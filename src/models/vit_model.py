"""
Vision Transformer feature-extraction branch (global / contextual representation).

Built on `timm` (free, open-source) with ImageNet-pretrained weights.
Exposes the [CLS] token embedding as the pooled global feature vector, and
the last self-attention block's attention weights for visualization.
"""
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn


class ViTBranch(nn.Module):
    """Wraps a timm ViT as a feature extractor with an attention-capturing hook."""

    def __init__(self, backbone: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.vit.num_features

        self._last_attn: Optional[torch.Tensor] = None
        self._register_attention_hook()

    def _register_attention_hook(self) -> None:
        """Hook the last transformer block's attention module so we can read
        the raw attention weights (used for attention-map visualization).
        timm's Attention module optionally stores attention if fused
        attention is disabled; we force eager attention for interpretability.
        """
        last_block = self.vit.blocks[-1]
        # Force eager (non-fused) attention implementation so weights are
        # materialized and can be captured by the hook.
        if hasattr(last_block.attn, "fused_attn"):
            last_block.attn.fused_attn = False

        def hook(module, input, output):
            # timm Attention forward computes `attn` internally; when
            # fused_attn=False it is exposed via module.attn if available,
            # otherwise we recompute a lightweight rollout-free version here.
            if hasattr(module, "attn_weights") and module.attn_weights is not None:
                self._last_attn = module.attn_weights.detach()

        last_block.attn.register_forward_hook(hook)
        self._patch_attention_module(last_block.attn)

    @staticmethod
    def _patch_attention_module(attn_module) -> None:
        """Monkeypatch timm's Attention.forward to stash softmax attention
        weights on the module (timm doesn't expose them by default).
        """
        orig_forward = attn_module.forward

        def patched_forward(x, attn_mask=None, is_causal=False, **kwargs):
            # attn_mask / is_causal are accepted (for API compatibility with
            # newer timm versions) but not used: standard ViT self-attention
            # over all patch + CLS tokens has no masking or causality.
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(
                B, N, 3, attn_module.num_heads, C // attn_module.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_module.q_norm(q), attn_module.k_norm(k)

            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)
            attn_module.attn_weights = attn  # (B, heads, N, N) stashed for hook/visualization
            attn = attn_module.attn_drop(attn)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)
            return x

        attn_module.forward = patched_forward

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.vit.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_block(self) -> None:
        """Stage-2 fine-tuning: unfreeze only the last transformer block + norm."""
        for p in self.vit.blocks[-1].parameters():
            p.requires_grad = True
        for p in self.vit.norm.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled = self.vit(x)  # (B, feat_dim) — CLS token after final norm
        return pooled, self._last_attn

    def get_last_attention(self) -> Optional[torch.Tensor]:
        """Returns (B, heads, N, N) softmax attention from the last block,
        captured during the most recent forward pass."""
        return self._last_attn


class ViTClassifier(nn.Module):
    """Standalone ViT classifier used as Baseline 3."""

    def __init__(self, backbone: str = "vit_base_patch16_224", num_classes: int = 7,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.branch = ViTBranch(backbone=backbone, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.branch.feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.branch(x)
        return self.classifier(pooled)
