"""Audio Spectrogram Transformer (AST) wrapper using HuggingFace transformers with fallback."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.augment import SpecAugment
from src.features.melspec import build_frontend
from src.utils.logging import get_logger

logger = get_logger("bird_classification.ast")


class FallbackASTModel(nn.Module):
    """Lightweight patch transformer fallback if HuggingFace weights cannot be downloaded."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 4):
        super().__init__()
        # Patch projection (16x16 patch)
        self.patch_embed = nn.Conv2d(1, hidden_size, kernel_size=(16, 16), stride=(16, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, input_values: torch.Tensor) -> Any:
        # input_values: (B, 1024, 128) -> reshape to (B, 1, 128, 1024)
        x = input_values.permute(0, 2, 1).unsqueeze(1)
        patches = self.patch_embed(x)  # (B, hidden, 8, 64)
        B, C, H, W = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)  # (B, 512, hidden)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        out = self.encoder(tokens)
        
        class Output:
            pass
        res = Output()
        res.pooler_output = out[:, 0]  # CLS token
        return res


class ASTAudioClassifier(nn.Module):
    """Audio Spectrogram Transformer (AST) wrapper for fine-tuning on bird bioacoustics."""

    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # 1. On-GPU audio feature extraction frontend
        self.frontend = build_frontend(cfg)

        # 2. SpecAugment
        spec_cfg = cfg.augment.spec if hasattr(cfg, "augment") else {}
        self.spec_augment = SpecAugment(
            time_mask_num=int(getattr(spec_cfg, "time_mask_num", 2)),
            time_mask_max_len=int(getattr(spec_cfg, "time_mask_max_len", 40)),
            freq_mask_num=int(getattr(spec_cfg, "freq_mask_num", 2)),
            freq_mask_max_len=int(getattr(spec_cfg, "freq_mask_max_len", 24)),
            prob=float(getattr(spec_cfg, "spec_aug_prob", 0.5)),
        )

        hf_id = str(getattr(cfg.model, "hf_model_id", "MIT/ast-finetuned-audioset-10-10-0.4593"))
        dropout_p = float(getattr(cfg.model, "dropout", 0.2))
        grad_ckpt = bool(getattr(cfg.model, "gradient_checkpointing", True))

        # 3. Load HuggingFace AST or initialize fallback
        hidden_size = 768
        self.is_hf = False
        try:
            from transformers import ASTConfig, ASTModel
            # Try loading pretrained weights or config
            try:
                self.ast = ASTModel.from_pretrained(hf_id)
                self.is_hf = True
                logger.info(f"Loaded pretrained AST model: {hf_id}")
            except Exception as e:
                logger.warning(f"Failed to load weights for {hf_id} ({e}); instantiating from ASTConfig.")
                config = ASTConfig()
                self.ast = ASTModel(config)
                self.is_hf = True

            if grad_ckpt and hasattr(self.ast, "gradient_checkpointing_enable"):
                self.ast.gradient_checkpointing_enable()
            hidden_size = self.ast.config.hidden_size
        except Exception as e:
            logger.warning(f"HuggingFace AST unavailable ({e}); falling back to local patch transformer.")
            self.ast = FallbackASTModel(hidden_size=768)

        # 4. Classification head
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning class logits."""
        # 1. Frontend feature extraction -> (B, 1, 128, 501)
        spec = self.frontend(x)
        if self.training:
            spec = self.spec_augment(spec)

        # 2. Resize/interpolate time dimension to match AST standard input (128 mels x 1024 time frames)
        if spec.shape[-1] != 1024:
            spec = F.interpolate(spec, size=(128, 1024), mode="bilinear", align_corners=False)

        # AST expects input_values of shape (B, 1024, 128)
        ast_input = spec.squeeze(1).permute(0, 2, 1)

        # 3. Transformer backbone forward pass
        if self.is_hf:
            outputs = self.ast(input_values=ast_input)
            pooled = outputs.pooler_output
            if pooled is None:
                # Use CLS token if pooler_output is None
                pooled = outputs.last_hidden_state[:, 0]
        else:
            outputs = self.ast(input_values=ast_input)
            pooled = outputs.pooler_output

        # 4. Classification head
        normed = self.layer_norm(pooled)
        dropped = self.dropout(normed)
        logits = self.classifier(dropped)
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        """Separate transformer backbone parameters from classification head."""
        head_params = list(self.classifier.parameters()) + list(self.layer_norm.parameters())
        head_ids = set(id(p) for p in head_params)
        backbone_params = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def freeze_backbone(self) -> None:
        """Freeze transformer backbone weights."""
        for param in self.ast.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze transformer backbone weights."""
        for param in self.parameters():
            param.requires_grad = True

