"""Grad-CAM visual attribution on 2D audio spectrogram representations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger("bird_classification.gradcam")


class GradCAM:
    """Computes Grad-CAM saliency heatmaps for convolutional audio backbones."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Find default target layer if not explicitly supplied
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._hook_handles = []
        self._hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Find the last Conv2d layer in the architecture."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model for Grad-CAM.")
        return last_conv

    def _save_activation(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        waveform: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Generate Grad-CAM heatmap and original spectrogram.
        
        Args:
            waveform: Tensor of shape (1, T) or (T,).
            target_class_idx: Class index to explain (defaults to model prediction).
            
        Returns:
            Tuple of (cam_heatmap [F, T], spectrogram [F, T], predicted_class_idx).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(next(self.model.parameters()).device)

        # Clear existing gradients
        self.model.zero_grad()

        # Get spectrogram
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            spec = raw_model.frontend(waveform).squeeze().cpu().numpy()

        # Forward pass
        logits = self.model(waveform)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        # Compute importance weights: Global average pooling over gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU to keep only features that positively correlate with class

        # Interpolate CAM to match input spectrogram dimensions
        cam_resized = F.interpolate(cam, size=spec.shape, mode="bilinear", align_corners=False)
        cam_np = cam_resized.squeeze().cpu().numpy()

        # Normalize CAM between 0 and 1
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, spec, target_class_idx

    def plot_and_save(
        self,
        waveform: torch.Tensor,
        class_names: List[str],
        save_path: Union[str, Path],
        target_class_idx: Optional[int] = None,
        title: Optional[str] = None,
    ) -> None:
        """Generate and save Grad-CAM overlay plot."""
        cam, spec, pred_idx = self.generate_cam(waveform, target_class_idx)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        class_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
        plot_title = title or f"Grad-CAM: Predicted {class_label}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # 1. Base Spectrogram
        ax1.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax1.set_ylabel("Mel Bins")
        ax1.set_title("Input Mel Spectrogram")

        # 2. Grad-CAM Overlay
        ax2.imshow(spec, aspect="auto", origin="lower", cmap="magma", alpha=0.6)
        im = ax2.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.4)
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bins")
        ax2.set_title(plot_title)
        fig.colorbar(im, ax=[ax1, ax2], fraction=0.046, pad=0.04, label="Relevance")

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=200)
        plt.close()
        logger.info(f"Saved Grad-CAM visualization to {save_path}.")

    def remove_hooks(self) -> None:
        """Remove PyTorch hook handles."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

