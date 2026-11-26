"""CLI inference script for bird audio recordings supporting single models and ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_audio
from src.models.ensemble import EnsembleModel, load_ensemble_from_checkpoints
from src.models.registry import build_model
from src.utils.audio_inference import aggregate_window_probs, slice_sliding_windows
from src.utils.config import load_config
from src.utils.io import load_checkpoint
from src.utils.logging import format_epoch_table, get_logger

logger = get_logger("infer")


def predict_audio_file(
    audio_path: Path,
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    sample_rate: int = 32000,
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
    is_multilabel: bool = False,
) -> Dict[str, Any]:
    """Perform sliding-window inference on a single audio file and average predictions.
    
    Args:
        audio_path: Path to audio file.
        model: Loaded PyTorch model or Ensemble.
        class_names: List of species names indexed by class ID.
        device: PyTorch device.
        sample_rate: Sampling rate in Hz.
        window_sec: Duration of each analysis window in seconds.
        hop_sec: Hop size in seconds.
        is_multilabel: Multilabel flag.
        
    Returns:
        Dictionary with recording predictions, top species, and window timeline.
    """
    waveform = load_audio(audio_path, target_sr=sample_rate, trim_silence=False)
    windows_np, timestamps = slice_sliding_windows(
        waveform,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    windows_tensor = torch.from_numpy(windows_np).float().to(device)

    model.eval()
    with torch.no_grad():
        out = model(windows_tensor)
        if isinstance(model, EnsembleModel):
            window_probs = out.cpu().numpy()
        else:
            if is_multilabel:
                window_probs = torch.sigmoid(out).cpu().numpy()
            else:
                window_probs = torch.softmax(out, dim=-1).cpu().numpy()

    # Aggregate probabilities over sliding windows weighted toward active vocalizations
    mean_probs = aggregate_window_probs(window_probs, windows=windows_np, mode="confidence_energy")
    top_indices = np.argsort(mean_probs)[::-1]

    # Timeline per-window top predictions
    timeline = []
    for t, probs in zip(timestamps, window_probs):
        w_top_idx = int(np.argmax(probs))
        timeline.append({
            "timestamp_sec": float(t),
            "top_species": class_names[w_top_idx],
            "confidence": float(probs[w_top_idx]),
        })

    return {
        "filename": audio_path.name,
        "probabilities": mean_probs,
        "top_indices": top_indices,
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on audio files.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to single model checkpoint (.pt).")
    parser.add_argument("--ensemble_checkpoints", type=str, nargs="+", default=None, help="List of checkpoint paths for ensemble.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Configuration path.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file or directory of audio files.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional destination path for prediction CSV.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional destination path for prediction JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

    # Load Model (single checkpoint or ensemble)
    if args.ensemble_checkpoints:
        logger.info(f"Loading ensemble of {len(args.ensemble_checkpoints)} models...")
        model = load_ensemble_from_checkpoints(args.ensemble_checkpoints, cfg, device=device)
        # Load class map from first checkpoint
        first_ckpt = torch.load(args.ensemble_checkpoints[0], map_location="cpu")
        class_to_idx = first_ckpt.get("class_to_idx", {})
    elif args.checkpoint:
        from src.utils.config import Config
        logger.info(f"Loading single checkpoint: {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        class_to_idx = ckpt.get("class_to_idx", {})
        ckpt_cfg = ckpt.get("config")
        model_cfg = Config(ckpt_cfg) if isinstance(ckpt_cfg, dict) and ckpt_cfg else cfg
        if "model" in model_cfg:
            model_cfg.model.pretrained = False
        num_classes = len(class_to_idx) or int(getattr(model_cfg.model, "num_classes", 206))
        model = build_model(model_cfg, num_classes=num_classes)
        load_checkpoint(args.checkpoint, model=model, device=device)
    else:
        raise ValueError("Must provide either --checkpoint or --ensemble_checkpoints!")

    model.to(device)
    model.eval()

    if not class_to_idx:
        raise ValueError("Could not find 'class_to_idx' mapping in checkpoint!")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # Gather audio files to process
    input_path = Path(args.audio_path)
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = sorted(list(input_path.glob("*.ogg")) + list(input_path.glob("*.wav")) + list(input_path.glob("*.flac")))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    logger.info(f"Found {len(audio_files)} audio file(s) for inference.")

    records = []
    json_results = []

    for af in audio_files:
        pred_res = predict_audio_file(
            audio_path=af,
            model=model,
            class_names=class_names,
            device=device,
            sample_rate=int(getattr(cfg.audio, "sample_rate", 32000)),
            window_sec=float(getattr(cfg.audio, "duration_sec", 5.0)),
            hop_sec=float(getattr(cfg.audio, "val_hop_sec", 2.5)),
            is_multilabel=is_multilabel,
        )

        top_k = min(args.top_k, len(class_names))
        top_k_indices = pred_res["top_indices"][:top_k]
        top_k_preds = [
            {"species": class_names[idx], "probability": float(pred_res["probabilities"][idx])}
            for idx in top_k_indices
        ]

        row = {"filename": af.name}
        for rank, item in enumerate(top_k_preds, start=1):
            row[f"top_{rank}_species"] = item["species"]
            row[f"top_{rank}_prob"] = f"{item['probability']:.4f}"
        records.append(row)

        json_results.append({
            "filename": af.name,
            "top_predictions": top_k_preds,
            "timeline": pred_res["timeline"],
        })

        # Display prediction in console
        headers = ["Rank", "Species", "Confidence"]
        rows = [[f"#{r}", item["species"], f"{item['probability']:.2%}"] for r, item in enumerate(top_k_preds, start=1)]
        print(f"\n--- Inference: {af.name} ---")
        print(format_epoch_table(headers, rows))

    if args.output_csv:
        out_csv_path = Path(args.output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv_path, index=False)
        logger.info(f"Saved predictions table to {out_csv_path}.")

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved predictions JSON to {out_json_path}.")


if __name__ == "__main__":
    main()

