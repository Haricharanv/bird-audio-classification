"""Wav2Vec2 baseline model with frozen temporal encoder and trainable linear head."""

from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.utils.logging import get_logger

logger = get_logger("bird_classification.wav2vec2")


class FallbackWav2Vec2Encoder(nn.Module):
    """Fallback 7-layer 1D CNN feature extractor simulating Wav2Vec2 front-end."""

    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, out_dim, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Any:
        # x: (B, T) -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x)  # (B, out_dim, T_sub)
        
        class Output:
            pass
        out = Output()
        out.last_hidden_state = feat.transpose(1, 2)  # (B, T_sub, out_dim)
        return out


class Wav2Vec2AudioClassifier(nn.Module):
    """Wav2Vec 2.0 audio classifier: Frozen feature encoder + Mean pooling + Linear head."""

    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # Wav2Vec2 was pretrained at 16 kHz: resample if input is 32 kHz
        self.sr = int(getattr(cfg.audio, "sample_rate", 32000))
        if self.sr != 16000:
            self.resampler = T.Resample(orig_freq=self.sr, new_freq=16000)
        else:
            self.resampler = nn.Identity()

        hidden_size = 768
        self.is_hf = False
        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
            try:
                self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self.is_hf = True
                logger.info("Loaded pretrained facebook/wav2vec2-base.")
            except Exception as e:
                logger.warning(f"Failed to fetch weights for wav2vec2 ({e}); instantiating config.")
                conf = Wav2Vec2Config()
                self.encoder = Wav2Vec2Model(conf)
                self.is_hf = True
            hidden_size = self.encoder.config.hidden_size
        except Exception as e:
            logger.warning(f"transformers.Wav2Vec2Model unavailable ({e}); using local CNN encoder.")
            self.encoder = FallbackWav2Vec2Encoder(out_dim=768)

        # Freeze encoder completely (comparison only benchmark)
        for param in self.encoder.parameters():
            param.requires_grad = False

        dropout_p = float(getattr(cfg.model, "dropout", 0.2))
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking raw waveform (B, T) and returning class logits."""
        # Resample to 16 kHz for wav2vec2
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        x_16k = self.resampler(x)

        # Frozen feature extraction
        with torch.no_grad():
            outputs = self.encoder(x_16k)
            hidden_states = outputs.last_hidden_state  # (B, T_sub, hidden_size)

        # Temporal mean pooling
        pooled = torch.mean(hidden_states, dim=1)      # (B, hidden_size)
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)              # (B, num_classes)
        return logits

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> List[Dict[str, Any]]:
        # Only classification head is trained
        return [{"params": list(self.classifier.parameters()), "lr": head_lr}]

    def freeze_backbone(self) -> None:
        pass

    def unfreeze_backbone(self) -> None:
        pass

