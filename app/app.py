"""Gradio Web Application for Bird Species Audio Classification."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_audio
from src.models.registry import build_model
from src.utils.config import Config, load_config
from src.utils.io import load_checkpoint
from src.utils.logging import get_logger

logger = get_logger("gradio_app")

# Global model cache
MODEL_CACHE: Dict[str, Any] = {}


def find_default_checkpoint() -> Optional[Path]:
    """Search for the most recent checkpoint in runs/."""
    runs_dir = PROJECT_ROOT / "runs"
    if runs_dir.is_dir():
        checkpoints = sorted(list(runs_dir.glob("**/best_model.pt")), key=os.path.getmtime, reverse=True)
        if checkpoints:
            return checkpoints[0]
    return None


def get_model_and_classes() -> Tuple[torch.nn.Module, list[str], torch.device]:
    """Load and cache model, species mapping, and device."""
    if "model" in MODEL_CACHE:
        return MODEL_CACHE["model"], MODEL_CACHE["class_names"], MODEL_CACHE["device"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_env = os.environ.get("CHECKPOINT_PATH")
    if ckpt_env and Path(ckpt_env).is_file():
        ckpt_path = Path(ckpt_env)
    else:
        ckpt_path = find_default_checkpoint()

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(
            "No trained model checkpoint found! Train a model first via `python scripts/train.py` "
            "or set the CHECKPOINT_PATH environment variable."
        )

    logger.info(f"Loading checkpoint for Gradio app: {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(ckpt.get("config", {}))
    if not cfg:
        cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    num_classes = len(class_names)
    cfg.model.num_classes = num_classes

    model = build_model(cfg, num_classes=num_classes)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.to(device)
    model.eval()

    MODEL_CACHE["model"] = model
    MODEL_CACHE["class_names"] = class_names
    MODEL_CACHE["device"] = device
    MODEL_CACHE["cfg"] = cfg

    return model, class_names, device


def predict_bird_audio(audio_filepath: str) -> Tuple[Dict[str, float], plt.Figure, plt.Figure]:
    """Run model inference on uploaded/recorded audio, compute spectrogram and timeline.
    
    Args:
        audio_filepath: Temporary path to input audio file.
        
    Returns:
        Tuple of (top_3_probabilities_dict, spectrogram_figure, timeline_figure).
    """
    if not audio_filepath:
        return {}, plt.figure(), plt.figure()

    model, class_names, device = get_model_and_classes()
    sr = 32000
    window_sec = 5.0
    hop_sec = 2.5
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    # 1. Load and process audio
    waveform = load_audio(audio_filepath, target_sr=sr, trim_silence=False)
    total_len = len(waveform)

    if total_len < window_samples:
        repeats = int(np.ceil(window_samples / max(1, total_len)))
        windows = [np.tile(waveform, repeats)[:window_samples]]
        timestamps = [0.0]
    else:
        windows = []
        timestamps = []
        for start in range(0, total_len - window_samples + 1, hop_samples):
            windows.append(waveform[start : start + window_samples])
            timestamps.append(start / sr)

        tail_start = total_len - window_samples
        if tail_start > 0 and (total_len - window_samples) % hop_samples != 0:
            windows.append(waveform[tail_start:total_len])
            timestamps.append(tail_start / sr)

    # 2. Model inference
    windows_tensor = torch.from_numpy(np.stack(windows, axis=0)).float().to(device)
    with torch.no_grad():
        logits = model(windows_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (num_windows, num_classes)

    # 3. Overall Top-3 species prediction (averaged across all windows)
    mean_probs = np.mean(probs, axis=0)
    top_indices = np.argsort(mean_probs)[::-1][:3]
    top_3_dict = {class_names[idx]: float(mean_probs[idx]) for idx in top_indices}

    # 4. Generate Spectrogram Plot
    raw_model = model.module if hasattr(model, "module") else model
    sample_crop = windows_tensor[:1]
    with torch.no_grad():
        spec = raw_model.frontend(sample_crop).squeeze().cpu().numpy()

    fig_spec, ax_spec = plt.subplots(figsize=(8, 3.5))
    im = ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax_spec.set_title("5-Second Log-Mel Spectrogram")
    ax_spec.set_xlabel("Time Frames")
    ax_spec.set_ylabel("Mel Frequency Bins")
    fig_spec.colorbar(im, ax=ax_spec, format="%+2.0f dB")
    fig_spec.tight_layout()

    # 5. Generate Timeline Plot of per-window predictions
    fig_timeline, ax_timeline = plt.subplots(figsize=(8, 3.5))
    top_per_window_idx = np.argmax(probs, axis=1)
    top_per_window_conf = np.max(probs, axis=1)
    top_per_window_labels = [class_names[idx] for idx in top_per_window_idx]

    ax_timeline.plot(timestamps, top_per_window_conf, marker="o", color="#1f77b4", linewidth=2)
    ax_timeline.fill_between(timestamps, top_per_window_conf, alpha=0.15, color="#1f77b4")

    for t, conf, label in zip(timestamps, top_per_window_conf, top_per_window_labels):
        ax_timeline.annotate(
            f"{label}\n({conf:.0%})",
            (t, conf),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax_timeline.set_title("Per-Window Top Bird Species Timeline")
    ax_timeline.set_xlabel("Recording Timestamp (seconds)")
    ax_timeline.set_ylabel("Confidence")
    ax_timeline.set_ylim(0.0, 1.1)
    ax_timeline.grid(True, linestyle="--", alpha=0.5)
    fig_timeline.tight_layout()

    return top_3_dict, fig_spec, fig_timeline


def create_demo_audio_examples(examples_dir: Path) -> list[list[str]]:
    """Create synthetic audio clips if no example files exist."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    ex1 = examples_dir / "sample_bird_chirp_1.ogg"
    ex2 = examples_dir / "sample_bird_chirp_2.ogg"

    if not ex1.is_file():
        sr = 32000
        t = np.linspace(0, 5, sr * 5)
        # Chirp synthesis
        f0, f1 = 2000, 6000
        chirp = 0.5 * np.sin(2 * np.pi * (f0 * t + ((f1 - f0) / 10) * (t**2 % 1.0)))
        sf.write(str(ex1), chirp.astype(np.float32), sr)

    if not ex2.is_file():
        sr = 32000
        t = np.linspace(0, 8, sr * 8)
        chirp2 = 0.4 * np.sin(2 * np.pi * 3500 * t) + 0.1 * np.random.randn(len(t))
        sf.write(str(ex2), chirp2.astype(np.float32), sr)

    return [[str(ex1)], [str(ex2)]]


def build_app():
    import gradio as gr

    examples_dir = PROJECT_ROOT / "app" / "examples"
    examples = create_demo_audio_examples(examples_dir)

    with gr.Blocks(title="Bird Species Audio Classifier | BirdCLEF+ 2025") as demo:
        gr.Markdown(
            "# 🐦 Bird Species Audio Classification (Bioacoustics)\n"
            "Upload an audio recording or record directly from your microphone to classify bird species "
            "using deep neural networks trained on the BirdCLEF+ 2025 dataset."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Input Audio Recording (.ogg, .wav, .flac)",
                )
                predict_btn = gr.Button("Classify Audio", variant="primary")
                gr.Examples(examples=examples, inputs=audio_input)

            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=3, label="Top-3 Predicted Species")
                spec_output = gr.Plot(label="Log-Mel Spectrogram")
                timeline_output = gr.Plot(label="Sliding Window Confidence Timeline")

        predict_btn.click(
            fn=predict_bird_audio,
            inputs=audio_input,
            outputs=[label_output, spec_output, timeline_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
