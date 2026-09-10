"""
Grad-CAM for the ResNet branch (requirement #13).

We implement Grad-CAM directly (rather than depending on exact internal
wiring of the `grad-cam` package with our custom hybrid forward) so it
works for both the standalone ResNet baselines and the ResNet branch inside
the hybrid model. Heatmaps are generated from real gradients/activations
of a real forward+backward pass — never fabricated.
"""
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Args:
            input_tensor: (1, 3, H, W) single preprocessed image
            class_idx: target class; if None, uses the model's own top prediction
        Returns:
            cam: (H, W) heatmap normalized to [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # forward populates activations via hook

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        # Global-average-pool gradients over spatial dims -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        input_size = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        return cam, class_idx


def overlay_heatmap(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb_image: (H, W, 3) float in [0,1]. cam: (H, W) float in [0,1]."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    return np.clip(overlay, 0, 1)
