"""Data preprocessing, metadata ingestion, taxa/rating filtering, and audio chunk extraction."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from src.utils.io import check_disk_space_or_abort, save_json
from src.utils.logging import get_logger

logger = get_logger("bird_classification.preprocess")


def parse_secondary_labels(val: Any) -> List[str]:
    """Parse stringified secondary_labels list safely.
    
    Args:
        val: Raw string, list, or NaN from CSV.
        
    Returns:
        List of secondary species label strings.
    """
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "[]":
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            # Fallback: remove brackets and split by comma
            cleaned = val.strip("[]'\" ")
            if cleaned:
                return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]
    return []


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 32000,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
    peak_norm: bool = True,
) -> np.ndarray:
    """Read audio file, convert to mono, resample to target_sr, trim silence, and peak-normalize.
    
    Args:
        file_path: Path to audio file (.ogg, .wav, .flac).
        target_sr: Desired sampling rate in Hz (default 32000).
        trim_silence: If True, trims leading and trailing silence.
        trim_top_db: Threshold in decibels below peak to consider silence.
        peak_norm: If True, scales waveform so peak absolute amplitude is 1.0.
        
    Returns:
        1D float32 numpy array containing processed waveform.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        # soundfile is fast and handles 32kHz ogg files natively
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        logger.warning(f"soundfile failed for {path} ({e}), falling back to librosa")
        waveform, sr = librosa.load(str(path), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[:, np.newaxis]
        else:
            waveform = waveform.T

    # Convert multi-channel to mono by averaging channels
    if waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
    else:
        waveform = waveform[:, 0]

    # Resample if audio sample rate differs from target
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")

    # Trim leading and trailing silence
    if trim_silence and len(waveform) > 0:
        trimmed, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
        if len(trimmed) > 0:
            waveform = trimmed

    # Peak normalization
    if peak_norm and len(waveform) > 0:
        peak = np.max(np.abs(waveform))
        if peak > 1e-7:
            waveform = waveform / peak

    return waveform.astype(np.float32)


def compute_chunk_rms(chunk: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) energy of an audio chunk."""
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk))))


def extract_energy_chunks(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    chunk_sec: float = 5.0,
    hop_sec: float = 2.5,
    min_rms: float = 1e-4,
) -> List[Tuple[int, np.ndarray, float]]:
    """Extract 5-second chunks using a sliding window, sorted by descending RMS energy.
    
    Args:
        waveform: 1D audio waveform.
        sample_rate: Audio sampling rate in Hz.
        chunk_sec: Duration of each chunk in seconds.
        hop_sec: Sliding window hop size in seconds.
        min_rms: Minimum RMS energy threshold to skip near-silent audio segments.
        
    Returns:
        List of tuples: (start_sample_idx, chunk_array, rms_energy) sorted by highest energy.
    """
    chunk_len = int(chunk_sec * sample_rate)
    hop_len = int(hop_sec * sample_rate)
    total_len = len(waveform)

    # If waveform is shorter than chunk_len, repeat-pad
    if total_len < chunk_len:
        if total_len == 0:
            return []
        repeats = int(np.ceil(chunk_len / total_len))
        padded = np.tile(waveform, repeats)[:chunk_len]
        rms = compute_chunk_rms(padded)
        if rms >= min_rms:
            return [(0, padded, rms)]
        return []

    chunks: List[Tuple[int, np.ndarray, float]] = []
    for start in range(0, total_len - chunk_len + 1, hop_len):
        chunk = waveform[start : start + chunk_len]
        rms = compute_chunk_rms(chunk)
        if rms >= min_rms:
            chunks.append((start, chunk, rms))

    # Also include the tail chunk if not already aligned
    tail_start = total_len - chunk_len
    if tail_start > 0 and (not chunks or chunks[-1][0] != tail_start):
        chunk = waveform[tail_start:total_len]
        rms = compute_chunk_rms(chunk)
        if rms >= min_rms:
            chunks.append((tail_start, chunk, rms))

    # Sort chunks by descending RMS energy
    chunks.sort(key=lambda x: x[2], reverse=True)
    return chunks


def build_metadata(cfg: Any) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Load, filter, and harmonize BirdCLEF+ 2025 metadata according to configuration.
    
    Args:
        cfg: Configuration object containing data paths and filter settings.
        
    Returns:
        Tuple of (filtered_dataframe, species_to_idx_mapping, dropped_species_list).
    """
    raw_dir = Path(cfg.data.raw_dir)
    train_csv_path = raw_dir / "train.csv"
    tax_csv_path = raw_dir / "taxonomy.csv"

    if not train_csv_path.is_file():
        raise FileNotFoundError(f"Missing train.csv at: {train_csv_path}")

    # Read ALL label columns as str to avoid mixed numeric/string types (e.g. iNat numeric IDs)
    dtype_dict = {
        "primary_label": str,
        "secondary_labels": str,
        "filename": str,
        "author": str,
        "collection": str,
    }
    df = pd.read_csv(train_csv_path, dtype=dtype_dict)
    initial_rows = len(df)
    logger.info(f"Loaded train.csv: {initial_rows} recordings.")

    # Parse secondary labels
    df["secondary_labels"] = df["secondary_labels"].apply(parse_secondary_labels)

    # Clean collection string
    if "collection" in df.columns:
        df["collection"] = df["collection"].fillna("XC").astype(str).str.strip()
    else:
        df["collection"] = "XC"

    # Merge taxonomy.csv if present to obtain class_name (Aves | Amphibia | Mammalia | Insecta)
    if tax_csv_path.is_file():
        tax_df = pd.read_csv(tax_csv_path, dtype=str)
        if "primary_label" in tax_df.columns and "class_name" in tax_df.columns:
            tax_sub = tax_df[["primary_label", "class_name"]].drop_duplicates(subset=["primary_label"])
            df = df.merge(tax_sub, on="primary_label", how="left")
            logger.info("Merged taxonomy.csv successfully.")
    
    if "class_name" not in df.columns:
        # Default fallback: assume all are Aves if taxonomy is absent
        df["class_name"] = "Aves"
    else:
        df["class_name"] = df["class_name"].fillna("Aves")

    # Apply taxa filter (default keeps Aves only; empty list keeps all species)
    taxa_filter = list(getattr(cfg.data, "taxa_filter", ["Aves"]) or [])
    if taxa_filter:
        before_taxa = len(df)
        df = df[df["class_name"].isin(taxa_filter)].copy()
        logger.info(f"Taxa filter {taxa_filter}: kept {len(df)} of {before_taxa} rows.")

    # Rating filter: applies strictly to collection == 'XC'; iNat and CSA have no ratings
    min_rating = float(getattr(cfg.data, "min_rating", 0.0))
    if min_rating > 0.0 and "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        # Retain all non-XC, or XC with rating >= min_rating
        keep_mask = (df["collection"] != "XC") | (df["rating"] >= min_rating)
        dropped_ratings = len(df) - keep_mask.sum()
        df = df[keep_mask].copy()
        logger.info(f"Rating filter (XC rating >= {min_rating}): dropped {dropped_ratings} low-rated XC files.")

    # Minimum files per species filter
    min_files = int(getattr(cfg.data, "min_files_per_species", 10))
    species_counts = df["primary_label"].value_counts()
    retained_species = species_counts[species_counts >= min_files].index.tolist()
    dropped_species = species_counts[species_counts < min_files].index.tolist()

    if dropped_species:
        metadata_dir = Path(cfg.data.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        dropped_log_path = metadata_dir / "dropped_species.log"
        with open(dropped_log_path, "w", encoding="utf-8") as f:
            f.write(f"# Species dropped with < {min_files} recordings (Total: {len(dropped_species)})\n")
            for sp in dropped_species:
                f.write(f"{sp}: {species_counts[sp]} files\n")
        logger.info(f"Dropped {len(dropped_species)} species with < {min_files} files (logged to {dropped_log_path}).")

    df = df[df["primary_label"].isin(retained_species)].copy()

    # Debug mode downsampling
    if getattr(cfg, "debug", False):
        debug_n_species = int(getattr(cfg.data, "debug_species", 5))
        debug_files = int(getattr(cfg.data, "debug_files_per_species", 20))
        top_species = df["primary_label"].value_counts().head(debug_n_species).index.tolist()
        df = df[df["primary_label"].isin(top_species)].copy()
        df = df.groupby("primary_label", group_keys=False).head(debug_files).reset_index(drop=True)
        logger.info(f"DEBUG MODE ENABLED: sample reduced to {len(df)} files across {debug_n_species} species.")

    # Build species to index mapping
    unique_species = sorted(df["primary_label"].unique())
    species_to_idx = {sp: idx for idx, sp in enumerate(unique_species)}
    df["target"] = df["primary_label"].map(species_to_idx)

    # Save filtered metadata and index map
    metadata_dir = Path(cfg.data.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_dir / "metadata.csv", index=False)
    save_json(species_to_idx, metadata_dir / "species_to_idx.json")
    logger.info(f"Saved {len(df)} records across {len(species_to_idx)} species to {metadata_dir / 'metadata.csv'}.")

    return df, species_to_idx, dropped_species


def cache_energy_chunks(
    df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Pre-chunk audio files into float16 .npy segments and cache under cache_dir.
    
    Capped at max_chunks_per_species, selecting the highest RMS energy chunks.
    Verifies disk space caps before writing.
    
    Args:
        df: Metadata DataFrame.
        cfg: Configuration object.
        
    Returns:
        DataFrame cataloging all created chunks.
    """
    raw_dir = Path(cfg.data.raw_dir)
    audio_dir = raw_dir / "train_audio"
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sr = int(getattr(cfg.audio, "sample_rate", 32000))
    duration_sec = float(getattr(cfg.audio, "duration_sec", 5.0))
    target_samples = int(duration_sec * sr)
    max_chunks_per_species = int(getattr(cfg.data, "max_chunks_per_species", 200))
    max_cache_gb = float(getattr(cfg.data, "max_cache_gb", 18.0))

    # Pre-estimate storage: (num_species * max_chunks * target_samples * 2 bytes [float16])
    num_species = df["primary_label"].nunique()
    bytes_per_chunk = target_samples * 2  # float16 is 2 bytes
    estimated_total_bytes = num_species * max_chunks_per_species * bytes_per_chunk
    estimated_gb = estimated_total_bytes / (1024.0**3)

    logger.info(f"Estimated maximum cache footprint: {estimated_gb:.2f} GB (safety limit: {max_cache_gb:.2f} GB).")
    check_disk_space_or_abort(cache_dir, incoming_gb=estimated_gb, max_allowed_gb=max_cache_gb)

    chunk_records: List[Dict[str, Any]] = []

    # Group recordings by species to enforce max_chunks_per_species
    for species, group in df.groupby("primary_label"):
        species_cache_dir = cache_dir / str(species)
        species_cache_dir.mkdir(parents=True, exist_ok=True)
        species_chunks: List[Tuple[str, int, np.ndarray, float, Dict[str, Any]]] = []

        for _, row in group.iterrows():
            filename = str(row["filename"])
            audio_path = audio_dir / filename
            if not audio_path.is_file():
                # Attempt search inside species subfolder if structure is train_audio/<species>/<filename>
                audio_path = audio_dir / str(species) / filename
            if not audio_path.is_file():
                continue

            try:
                waveform = load_audio(
                    audio_path,
                    target_sr=sr,
                    trim_silence=bool(getattr(cfg.audio, "trim_silence", True)),
                    trim_top_db=float(getattr(cfg.audio, "trim_top_db", 30.0)),
                    peak_norm=bool(getattr(cfg.audio, "peak_normalize", True)),
                )
            except Exception as err:
                logger.warning(f"Skipping unreadable file {audio_path}: {err}")
                continue

            # Extract chunks sorted by RMS
            extracted = extract_energy_chunks(
                waveform,
                sample_rate=sr,
                chunk_sec=duration_sec,
                hop_sec=float(getattr(cfg.audio, "val_hop_sec", 2.5)),
            )

            for c_idx, (start_sample, chunk_arr, rms_val) in enumerate(extracted):
                species_chunks.append((
                    filename,
                    start_sample,
                    chunk_arr,
                    rms_val,
                    row.to_dict(),
                ))

        # Sort all chunks of this species by RMS energy and keep the top max_chunks_per_species
        species_chunks.sort(key=lambda x: x[3], reverse=True)
        selected_chunks = species_chunks[:max_chunks_per_species]

        for idx, (fn, start_s, chunk_arr, rms_val, meta_dict) in enumerate(selected_chunks):
            stem = Path(fn).stem
            chunk_filename = f"{stem}_c{idx:03d}.npy"
            chunk_path = species_cache_dir / chunk_filename

            # Save as float16 to conserve memory and disk
            np.save(str(chunk_path), chunk_arr.astype(np.float16))

            record = {
                "chunk_path": str(chunk_path.relative_to(cache_dir.parent)),
                "filename": fn,
                "primary_label": species,
                "target": meta_dict.get("target"),
                "author": meta_dict.get("author", "Unknown"),
                "collection": meta_dict.get("collection", "XC"),
                "rms_energy": rms_val,
                "start_sample": start_s,
            }
            chunk_records.append(record)

    chunks_df = pd.DataFrame(chunk_records)
    chunks_meta_path = Path(cfg.data.metadata_dir) / "cached_chunks.csv"
    chunks_df.to_csv(chunks_meta_path, index=False)
    logger.info(f"Successfully cached {len(chunks_df)} chunks across {num_species} species to {cache_dir}.")
    return chunks_df
