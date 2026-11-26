"""CLI script for rigorous recording-level evaluation on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BirdDataset
from src.evaluation.analysis import generate_full_evaluation_report
from src.evaluation.gradcam import GradCAM
from src.models.registry import build_model
from src.utils.audio_inference import aggregate_window_probs
from src.utils.config import load_config
from src.utils.io import load_checkpoint
from src.utils.logging import format_epoch_table, get_logger

logger = get_logger("evaluate")


def evaluate_recording_sliding_windows(
    model: torch.nn.Module,
    dataset: BirdDataset,
    device: torch.device,
    is_multilabel: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate dataset by averaging sliding window predictions for each recording.
    
    Args:
        model: Trained PyTorch model.
        dataset: BirdDataset instantiated with use_sliding_windows=True.
        device: Evaluation device.
        is_multilabel: Multi-label task flag.
        
    Returns:
        Tuple of (y_true, y_pred_probs) averaged per recording.
    """
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    logger.info(f"Evaluating {len(dataset)} recordings using sliding 5s windows (2.5s hop)...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            windows_waveform = sample["waveform"].to(device)  # (num_windows, target_samples)
            target = sample["target"].numpy()

            # Pass all 5s windows of this recording through the model
            logits = model(windows_waveform)
            if is_multilabel:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Aggregate probabilities across windows weighted toward active calls
            windows_np = sample["waveform"].numpy()
            recording_prob = aggregate_window_probs(probs, windows=windows_np, mode="confidence_energy")
            all_preds.append(recording_prob)
            all_targets.append(target)

    y_pred_probs = np.stack(all_preds, axis=0)  # (N, num_classes)
    y_true = np.array(all_targets) if not is_multilabel else np.stack(all_targets, axis=0)
    return y_true, y_pred_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on held-out split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to evaluate on.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation reports.")
    parser.add_argument("overrides", nargs="*", help="Key=value configuration overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", {})
    if not class_to_idx:
        raise ValueError("Checkpoint does not contain 'class_to_idx' mapping!")

    num_classes = len(class_to_idx)
    cfg.model.num_classes = num_classes
    cfg.model.pretrained = False

    # Build model and load weights
    model = build_model(cfg, num_classes=num_classes)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.to(device)
    model.eval()

    # Load metadata splits
    splits_path = Path(cfg.data.metadata_dir) / "splits.csv"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing splits at: {splits_path}")

    splits_df = pd.read_csv(splits_path, dtype={"primary_label": str, "filename": str})
    eval_df = splits_df[splits_df["split"] == args.split].reset_index(drop=True)
    logger.info(f"Evaluating split '{args.split}': {len(eval_df)} recordings across {num_classes} classes.")

    is_multilabel = str(getattr(cfg, "task", "multiclass")).lower() == "multilabel"

    # Instantiate evaluation dataset with sliding windows
    dataset = BirdDataset(
        df=eval_df,
        cfg=cfg,
        split=args.split,
        species_to_idx=class_to_idx,
        use_sliding_windows=True,
    )

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"evaluation_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run window-averaged evaluation
    y_true, y_pred_probs = evaluate_recording_sliding_windows(
        model=model,
        dataset=dataset,
        device=device,
        is_multilabel=is_multilabel,
    )

    # 2. Generate comprehensive reports and plots
    frontend = model.frontend if hasattr(model, "frontend") else None
    if frontend is not None:
        frontend.cfg = cfg
    metrics = generate_full_evaluation_report(
        y_true=y_true,
        y_pred_probs=y_pred_probs,
        df_meta=eval_df,
        class_to_idx=class_to_idx,
        output_dir=out_dir,
        frontend=frontend,
        is_multilabel=is_multilabel,
    )

    # 3. Grad-CAM diagnostic on first recording
    try:
        sample_wf = dataset[0]["waveform"]
        if sample_wf.dim() == 2:
            sample_wf = sample_wf[0]  # Take first 5s window
        gradcam = GradCAM(model)
        idx_to_sp = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_sp[i] for i in range(len(class_to_idx))]
        gradcam.plot_and_save(
            waveform=sample_wf,
            class_names=class_names,
            save_path=out_dir / "sample_gradcam.png",
        )
        gradcam.remove_hooks()
    except Exception as e:
        logger.warning(f"Grad-CAM generation skipped: {e}")

    # Print Formatted Metric Tables
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({args.split.upper()} SPLIT)")
    print("=" * 60)
    headers = ["Metric", "Score"]
    rows = [
        ["Macro-F1", metrics["macro_f1"]],
        ["Weighted-F1", metrics["weighted_f1"]],
        ["Top-1 Accuracy", metrics["top1_accuracy"]],
        ["Top-3 Accuracy", metrics["top3_accuracy"]],
        ["Macro ROC-AUC", metrics["roc_auc"]],
    ]
    print(format_epoch_table(headers, rows))

    if "per_collection" in metrics:
        print("\n" + "=" * 60)
        print("PER-COLLECTION BREAKDOWN (XC vs iNat vs CSA)")
        print("=" * 60)
        coll_headers = ["Collection", "Files", "Macro-F1", "Top-1 Acc", "Top-3 Acc", "ROC-AUC"]
        coll_rows = [
            [
                c["collection"],
                c["recordings"],
                c["macro_f1"],
                c["top1_acc"],
                c["top3_acc"],
                c["roc_auc"],
            ]
            for c in metrics["per_collection"]
        ]
        print(format_epoch_table(coll_headers, coll_rows))

    print(f"\nAll report artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
