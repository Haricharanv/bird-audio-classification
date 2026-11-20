"""Unit tests for model architectures forward pass, loss functions, and gradient flow."""

import pytest
import torch

from src.models.ast_wrapper import ASTAudioClassifier
from src.models.baseline_cnn import BaselineCNN
from src.models.crnn import CRNN
from src.models.pretrained_cnn import PretrainedCNN
from src.models.registry import build_model
from src.training.losses import CrossEntropyWithSmoothing, FocalLoss, build_loss
from src.utils.config import Config


@pytest.fixture
def base_model_cfg():
    return Config({
        "task": "multiclass",
        "audio": {
            "feature_type": "logmel",
            "sample_rate": 32000,
            "duration_sec": 5.0,
            "n_fft": 1024,
            "hop_length": 320,
            "win_length": 1024,
            "n_mels": 128,
            "fmin": 20.0,
            "fmax": 16000.0,
            "standardize": True,
        },
        "augment": {
            "spec": {
                "spec_aug_prob": 0.0,
            }
        },
        "model": {
            "in_chans": 1,
            "num_classes": 10,
            "dropout": 0.1,
            "pretrained": False,
        },
        "train": {
            "loss": "cross_entropy",
            "label_smoothing": 0.05,
        },
    })


@pytest.fixture
def dummy_waveforms():
    # 2 audio clips of 5 seconds (160,000 samples @ 32 kHz)
    return torch.randn(2, 160000)


def test_baseline_cnn_forward_and_backward(base_model_cfg, dummy_waveforms):
    base_model_cfg.model.name = "baseline_cnn"
    model = BaselineCNN(base_model_cfg, num_classes=10)
    
    out = model(dummy_waveforms)
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any()

    # Verify backward pass and gradients
    loss = out.sum()
    loss.backward()
    assert model.fc.weight.grad is not None
    assert not torch.isnan(model.fc.weight.grad).any()


def test_timm_cnn_forward_and_freeze(base_model_cfg, dummy_waveforms):
    base_model_cfg.model.name = "timm_cnn"
    base_model_cfg.model.backbone = "resnet18"
    base_model_cfg.model.pretrained = False
    model = PretrainedCNN(base_model_cfg, num_classes=10)

    out = model(dummy_waveforms)
    assert out.shape == (2, 10)

    # Test parameter separation
    groups = model.get_param_groups(head_lr=1e-3, backbone_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-3

    # Test backbone freezing
    model.freeze_backbone()
    trainable_backbone = [p for p in groups[0]["params"] if p.requires_grad]
    assert len(trainable_backbone) == 0, "Backbone parameters should be frozen!"
    
    model.unfreeze_backbone()
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(all_trainable) == len(list(model.parameters()))


def test_crnn_forward(base_model_cfg, dummy_waveforms):
    base_model_cfg.model.name = "crnn"
    base_model_cfg.model.backbone = "resnet18"
    base_model_cfg.model.pretrained = False
    base_model_cfg.model.gru_hidden_size = 64
    base_model_cfg.model.gru_layers = 1
    model = CRNN(base_model_cfg, num_classes=10)

    out = model(dummy_waveforms)
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any()


def test_ast_forward(base_model_cfg, dummy_waveforms):
    base_model_cfg.model.name = "ast"
    model = ASTAudioClassifier(base_model_cfg, num_classes=10)

    out = model(dummy_waveforms)
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any()


def test_build_model_registry(base_model_cfg, dummy_waveforms):
    base_model_cfg.model.name = "baseline_cnn"
    model = build_model(base_model_cfg, num_classes=15)
    assert isinstance(model, BaselineCNN)
    out = model(dummy_waveforms)
    assert out.shape == (2, 15)


def test_losses_1d_and_2d():
    logits = torch.randn(4, 5, requires_grad=True)
    
    # 1. CrossEntropy with 1D hard targets
    targets_1d = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    ce_loss = CrossEntropyWithSmoothing(label_smoothing=0.05)
    l1 = ce_loss(logits, targets_1d)
    assert l1.dim() == 0
    assert l1.item() > 0.0

    # 2. CrossEntropy with 2D soft/Mixup targets
    targets_2d = torch.softmax(torch.randn(4, 5), dim=-1)
    l2 = ce_loss(logits, targets_2d)
    assert l2.dim() == 0
    assert l2.item() > 0.0

    # 3. Focal Loss
    focal = FocalLoss(gamma=2.0)
    lf = focal(logits, targets_1d)
    assert lf.dim() == 0
    assert lf.item() > 0.0

