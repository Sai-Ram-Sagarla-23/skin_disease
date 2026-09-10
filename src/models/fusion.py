"""
Attention-based feature fusion module.

This is the core research contribution described in the project brief:
instead of naively concatenating the ResNet (local) and ViT (global)
feature vectors, this module learns, per-sample, how much to weight each
branch before fusing them.

Mathematical formulation
-------------------------
Let r ∈ R^{d_r} be the pooled ResNet feature and v ∈ R^{d_v} the pooled ViT
(CLS token) feature for a given image.

1. Project both branches into a shared dimensionality d:
       R = W_r r + b_r,   R ∈ R^d
       V = W_v v + b_v,   V ∈ R^d

2. Compute a joint context vector:
       H = concat(R, V) ∈ R^{2d}

3. Compute unnormalized gating scores with a small MLP:
       [s_r, s_v] = W_2 · tanh(W_1 H + b_1) + b_2 ∈ R^2

4. Normalize with softmax to obtain the learned, sample-specific branch
   weights (these sum to 1 and are what requirement #12 calls
   "ResNet contribution" / "ViT contribution"):
       [α_r, α_v] = softmax([s_r, s_v])

5. Fuse:
       F = α_r ⊙ R + α_v ⊙ V         (element-wise weighted sum)

   (A learnable per-channel gate is also supported — see `gate_mode`.)

F ∈ R^d is then passed to the classification head (Dropout + Linear).

Two gate granularities are supported:
    - "scalar" : one (α_r, α_v) pair per sample (as in the formulation above)
    - "channel": a (α_r, α_v) pair *per feature channel* per sample, giving
                 the model finer-grained control (a form of channel-wise
                 gated attention). This is the default, richer mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, resnet_dim: int, vit_dim: int, fusion_dim: int = 512,
                 gate_mode: str = "channel", dropout: float = 0.1):
        super().__init__()
        assert gate_mode in ("scalar", "channel")
        self.gate_mode = gate_mode
        self.fusion_dim = fusion_dim

        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_out_dim = fusion_dim if gate_mode == "channel" else 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, gate_out_dim * 2),  # 2 = one score per branch
        )

    def forward(self, resnet_feat: torch.Tensor, vit_feat: torch.Tensor):
        """
        Args:
            resnet_feat: (B, resnet_dim)
            vit_feat:    (B, vit_dim)
        Returns:
            fused:        (B, fusion_dim)
            alpha_resnet: (B,) or (B, fusion_dim) learned ResNet weight
            alpha_vit:    (B,) or (B, fusion_dim) learned ViT weight
        """
        R = self.resnet_proj(resnet_feat)   # (B, d)
        V = self.vit_proj(vit_feat)         # (B, d)

        H = torch.cat([R, V], dim=-1)       # (B, 2d)
        scores = self.gate_mlp(H)           # (B, 2) or (B, 2d)

        if self.gate_mode == "scalar":
            scores = scores.view(-1, 2)                  # (B, 2)
            alphas = F.softmax(scores, dim=-1)            # (B, 2)
            alpha_r, alpha_v = alphas[:, 0:1], alphas[:, 1:2]
        else:  # channel-wise
            scores = scores.view(-1, 2, self.fusion_dim)  # (B, 2, d)
            alphas = F.softmax(scores, dim=1)              # softmax over the 2 branches, per channel
            alpha_r, alpha_v = alphas[:, 0, :], alphas[:, 1, :]

        fused = alpha_r * R + alpha_v * V   # (B, d)
        return fused, alpha_r, alpha_v


class ConcatFusion(nn.Module):
    """Simple concatenation baseline (requirement #9 Baseline 4 / ablation
    Experiment 3) — used to demonstrate why the attention mechanism is an
    actual improvement rather than an assumed one."""

    def __init__(self, resnet_dim: int, vit_dim: int, fusion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(resnet_dim + vit_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, resnet_feat: torch.Tensor, vit_feat: torch.Tensor):
        fused = self.proj(torch.cat([resnet_feat, vit_feat], dim=-1))
        return fused, None, None
