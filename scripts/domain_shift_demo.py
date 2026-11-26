"""Qualitative domain shift demo: run a trained model over train_soundscapes and plot per-window predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_audio
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.io import load_checkpoint
from src.utils.logging import get_logger

logger = get_logger("domain_shift_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-shift analysis over unlabeled soundscapes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config.")
    parser.add_argument("--num_soundscapes", type=int, default=3, help="Number of soundscapes to inspect.")
    parser.add_argument("--output_dir", type=str, default="reports/domain_shift", help="Output directory for plots.")
    parser.add_argument("overrides", nargs="*", help="CLI overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_sp = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_sp[i] for i in range(len(class_to_idx))]

    num_classes = len(class_to_idx)
    cfg.model.num_classes = num_classes
    cfg.model.pretrained = False

    # Load model
    model = build_model(cfg, num_classes=num_classes)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.to(device)
    model.eval()

    raw_dir = Path(cfg.data.raw_dir)
    soundscapes_dir = raw_dir / "train_soundscapes"
    sc_files = sorted(list(soundscapes_dir.glob("*.ogg"))) if soundscapes_dir.is_dir() else []

    if not sc_files:
        logger.warning(f"No soundscapes found in {soundscapes_dir}. Creating a synthetic soundscape for demo.")
        synth_path = out_dir / "mock_soundscape.ogg"
        y_synth = np.random.randn(32000 * 30).astype(np.float32) * 0.1
        sf.write(str(synth_path), y_synth, 32000)
        sc_files = [synth_path]

    sample_sc_files = sc_files[: args.num_soundscapes]
    sr = 32000
    window_samples = 160000  # 5s
    hop_samples = 80000      # 2.5s

    for sc_file in sample_sc_files:
        logger.info(f"Analyzing soundscape: {sc_file.name}...")
        y = load_audio(sc_file, target_sr=sr, trim_silence=False)
        total_len = len(y)
        if total_len < window_samples:
            continue

        windows = []
        timestamps = []
        for start in range(0, total_len - window_samples + 1, hop_samples):
            windows.append(y[start : start + window_samples])
            timestamps.append(start / sr)

        windows_tensor = torch.from_numpy(np.stack(windows, axis=0)).float().to(device)
        with torch.no_grad():
            logits = model(windows_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (W, num_classes)

        # Plot timeline of per-window top predictions
        top_indices = np.argmax(probs, axis=1)
        top_confidences = np.max(probs, axis=1)
        top_labels = [class_names[idx] for idx in top_indices]

        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, top_confidences, marker="o", color="darkcyan", linewidth=2)
        plt.fill_between(timestamps, top_confidences, alpha=0.2, color="darkcyan")
        for t, conf, label in zip(timestamps, top_confidences, top_labels):
            if conf > 0.15:  # Label prominent detections
                plt.annotate(
                    f"{label}\n({conf:.0%})",
                    (t, conf),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

        plt.title(f"Domain Shift Analysis: {sc_file.name}\nSliding Window (5s, 2.5s hop) Top Predictions")
        plt.xlabel("Audio Timeline (seconds)")
        plt.ylabel("Prediction Confidence")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_path = out_dir / f"domain_shift_{sc_file.stem}.png"
        plt.savefig(str(plot_path), dpi=200)
        plt.close()
        logger.info(f"Saved domain shift timeline plot to {plot_path}.")

    print(f"\nDomain shift demo complete! Plots saved to {out_dir}")


if __name__ == "__main__":
    main()

