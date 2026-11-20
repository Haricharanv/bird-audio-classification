"""4-block Baseline Convolutional Neural Network with on-GPU audio frontend."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn

from src.features.melspec import build_frontend
from src.data.augment import SpecAugment


class ConvBlock(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> AvgPool2d."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BaselineCNN(nn.Module):
    """Baseline Audio Classifier: GPU frontend + 4 ConvBlocks + Global Pooling + FC."""

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

        in_chans = int(getattr(cfg.model, "in_chans", 1))
        dropout_p = float(getattr(cfg.model, "dropout", 0.2))

        # 3. 4-block CNN backbone (64 -> 128 -> 256 -> 512)
        self.block1 = ConvBlock(in_chans, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)

        # 4. Global pooling & classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spectrogram and backbone feature map."""
        spec = self.frontend(x)  # (B, 1, F, T)
        if self.training:
            spec = self.spec_augment(spec)
        f = self.block1(spec)
        f = self.block2(f)
        f = self.block3(f)
        f = self.block4(f)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning logits (B, num_classes)."""
        feat = self.extract_features(x)
        pooled = self.global_pool(feat).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for discriminative learning rates."""
        head_params = list(self.fc.parameters())
        head_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def freeze_backbone(self) -> None:
        """Freeze feature extractor parameters."""
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        for param in self.block3.parameters():
            param.requires_grad = False
        for param in self.block4.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze feature extractor parameters."""
        for param in self.parameters():
            param.requires_grad = True

