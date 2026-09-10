"""
Vision Transformer attention-map visualization (requirement #13).

Uses the real softmax attention weights captured by ViTBranch's forward
hook (src/models/vit_model.py) — specifically, the [CLS] token's attention
to every patch token in the last transformer block, averaged over heads.
This is a standard, widely used approximation of "where the transformer
looked" and is NOT a fabricated/simulated heatmap.
"""
import cv2
import numpy as np
import torch


def get_cls_attention_map(attn_weights: torch.Tensor, image_size: int, patch_size: int = 16) -> np.ndarray:
    """
    Args:
        attn_weights: (1, num_heads, N, N) softmax attention from the last
                      ViT block, where N = 1 (CLS) + num_patches.
        image_size: input image side length (e.g. 224)
        patch_size: ViT patch size (16 for vit_base_patch16_224)
    Returns:
        (H, W) attention heatmap resized to image_size, normalized [0, 1]
    """
    # Average over attention heads
    attn = attn_weights.mean(dim=1)  # (1, N, N)

    # CLS token (index 0) attention to all patch tokens (index 1:)
    cls_attn = attn[0, 0, 1:]  # (num_patches,)

    grid_size = image_size // patch_size
    cls_attn = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    if cls_attn.max() > cls_attn.min():
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    else:
        cls_attn = np.zeros_like(cls_attn)

    cls_attn = cv2.resize(cls_attn, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    cls_attn = np.clip(cls_attn, 0, 1)
    return cls_attn


def overlay_attention(rgb_image: np.ndarray, attn_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    return np.clip(overlay, 0, 1)
