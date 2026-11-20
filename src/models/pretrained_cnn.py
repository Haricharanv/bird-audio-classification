"""Pretrained 2D CNN backbones from timm with on-GPU audio frontend."""

from __future__ import annotations

from typing import Any, Dict, List
import timm
import torch
import torch.nn as nn

from src.data.augment import SpecAugment
from src.features.melspec import build_frontend


class PretrainedCNN(nn.Module):
    """Pretrained timm backbone audio classifier (ResNet, EfficientNet, ConvNeXt, etc.)."""

    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # 1. On-GPU audio feature extraction frontend
        self.frontend = build_frontend(cfg)

        # 2. SpecAugment module
        spec_cfg = cfg.augment.spec if hasattr(cfg, "augment") else {}
        self.spec_augment = SpecAugment(
            time_mask_num=int(getattr(spec_cfg, "time_mask_num", 2)),
            time_mask_max_len=int(getattr(spec_cfg, "time_mask_max_len", 40)),
            freq_mask_num=int(getattr(spec_cfg, "freq_mask_num", 2)),
            freq_mask_max_len=int(getattr(spec_cfg, "freq_mask_max_len", 24)),
            prob=float(getattr(spec_cfg, "spec_aug_prob", 0.5)),
        )

        backbone_name = str(getattr(cfg.model, "backbone", "efficientnet_b0"))
        pretrained = bool(getattr(cfg.model, "pretrained", True))
        in_chans = int(getattr(cfg.model, "in_chans", 1))
        dropout_p = float(getattr(cfg.model, "dropout", 0.2))

        # 3. Create timm backbone
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning class logits."""
        spec = self.frontend(x)  # (B, 1, F, T)
        if self.training:
            spec = self.spec_augment(spec)
        logits = self.backbone(spec)
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        """Separate backbone and classifier head for discriminative learning rates."""
        head_names = ["classifier", "fc", "head"]
        head_params = []
        backbone_params = []

        for name, param in self.backbone.named_parameters():
            if any(h in name for h in head_names):
                head_params.append(param)
            else:
                backbone_params.append(param)

        if not head_params:
            # Fallback if head name didn't match standard prefixes
            head_params = list(self.backbone.get_classifier().parameters())
            head_ids = set(id(p) for p in head_params)
            backbone_params = [p for p in self.parameters() if id(p) not in head_ids]

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def freeze_backbone(self) -> None:
        """Freeze backbone weights during initial warmup epochs."""
        head_names = ["classifier", "fc", "head"]
        for name, param in self.backbone.named_parameters():
            if not any(h in name for h in head_names):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

