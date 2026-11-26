"""Download research-grade bird vocalizations from iNaturalist into BirdCLEF-compatible layout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import soundfile as sf

# 8 distinctive bird species with abundant research-grade iNaturalist recordings
DEFAULT_SPECIES: List[Tuple[str, str, str]] = [
    ("common_blackbird", "Common Blackbird", "Turdus merula"),
    ("northern_cardinal", "Northern Cardinal", "Cardinalis cardinalis"),
    ("american_crow", "American Crow", "Corvus brachyrhynchos"),
    ("common_cuckoo", "Common Cuckoo", "Cuculus canorus"),
    ("house_sparrow", "House Sparrow", "Passer domesticus"),
    ("mourning_dove", "Mourning Dove", "Zenaida macroura"),
    ("great_tit", "Great Tit", "Parus major"),
    ("american_robin", "American Robin", "Turdus migratorius"),
]

USER_AGENT = "bird-audio-classification-edu/1.0 (https://github.com/)"


def fetch_observations(sci_name: str, page: int = 1, per_page: int = 80) -> List[Dict[str, Any]]:
    """Query iNaturalist API v1 for research-grade observations containing audio."""
    params = urllib.parse.urlencode({
        "taxon_name": sci_name,
        "sounds": "true",
        "quality_grade": "research",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "votes",
    })
    url = f"https://api.inaturalist.org/v1/observations?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("results", [])


def download_and_convert_clip(
    sound_url: str,
    dest_ogg: Path,
    max_duration_sec: int = 20,
    min_duration_sec: float = 2.5,
    min_rms: float = 0.005,
    max_download_bytes: int = 12 * 1024 * 1024,
) -> Optional[float]:
    """Download an audio URL, transcode to 32kHz mono OGG, and verify RMS energy."""
    if not sound_url:
        return None

    req = urllib.request.Request(sound_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_len = resp.headers.get("Content-Length")
            if content_len and int(content_len) > max_download_bytes:
                return None
            raw_bytes = resp.read(max_download_bytes + 1)
            if len(raw_bytes) > max_download_bytes or len(raw_bytes) < 2048:
                return None
    except Exception:
        return None

    suffix = Path(urllib.parse.urlparse(sound_url).path).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp_in:
        tmp_in.write(raw_bytes)
        tmp_in.flush()

        dest_ogg.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            tmp_in.name,
            "-t",
            str(max_duration_sec),
            "-ac",
            "1",
            "-ar",
            "32000",
            "-c:a",
            "libvorbis",
            "-q:a",
            "4",
            str(dest_ogg),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode != 0 or not dest_ogg.is_file():
            dest_ogg.unlink(missing_ok=True)
            return None

    try:
        audio, sr = sf.read(str(dest_ogg), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        duration = len(audio) / float(sr)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
        if duration < min_duration_sec or rms < min_rms or np.isnan(audio).any():
            dest_ogg.unlink(missing_ok=True)
            return None
        return duration
    except Exception:
        dest_ogg.unlink(missing_ok=True)
        return None


def collect_species_recordings(
    label: str,
    common_name: str,
    sci_name: str,
    audio_root: Path,
    target_count: int = 20,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Collect target_count distinct-recordist audio clips for one bird species."""
    species_dir = audio_root / label
    species_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[str, str, str]] = []
    seen_authors = set()
    seen_obs = set()

    for page in (1, 2, 3):
        try:
            results = fetch_observations(sci_name, page=page, per_page=80)
        except Exception as e:
            print(f"[{label}] Warning: failed page {page}: {e}")
            break
        if not results:
            break

        for obs in results:
            obs_id = str(obs.get("id", ""))
            if obs_id in seen_obs:
                continue
            author = str((obs.get("user") or {}).get("login") or f"inat_{obs_id}").strip()
            if author in seen_authors:
                continue
            sounds = obs.get("sounds") or []
            sound_url = None
            for s in sounds:
                u = s.get("file_url")
                if u:
                    sound_url = u
                    break
            if not sound_url:
                continue

            seen_obs.add(obs_id)
            seen_authors.add(author)
            candidates.append((obs_id, author, sound_url))
            if len(candidates) >= target_count * 2:
                break
        if len(candidates) >= target_count * 2:
            break
        time.sleep(0.3)

    rows: List[Dict[str, Any]] = []
    pending: List[Tuple[str, str, str, str, Path]] = []
    for obs_id, author, url in candidates:
        if len(rows) >= target_count:
            break
        rel_name = f"{label}/{label}_{obs_id}.ogg"
        out_path = audio_root / rel_name
        if out_path.is_file() and out_path.stat().st_size > 1000:
            rows.append({
                "primary_label": label,
                "secondary_labels": "[]",
                "filename": rel_name,
                "author": author,
                "collection": "iNat",
                "rating": "",
            })
        else:
            pending.append((obs_id, author, url, rel_name, out_path))

    cursor = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(rows) < target_count and cursor < len(pending):
            needed = target_count - len(rows)
            batch = pending[cursor : cursor + needed]
            cursor += len(batch)
            future_map = {
                pool.submit(download_and_convert_clip, url, out_path): (obs_id, author, rel_name, out_path)
                for obs_id, author, url, rel_name, out_path in batch
            }
            for fut in as_completed(future_map):
                obs_id, author, rel_name, out_path = future_map[fut]
                dur = fut.result()
                if dur is not None and len(rows) < target_count:
                    rows.append({
                        "primary_label": label,
                        "secondary_labels": "[]",
                        "filename": rel_name,
                        "author": author,
                        "collection": "iNat",
                        "rating": "",
                    })
                elif len(rows) >= target_count:
                    out_path.unlink(missing_ok=True)

    print(f"[{label}] ({common_name} / {sci_name}): collected {len(rows)} recordings.", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real bird audio dataset from iNaturalist.")
    parser.add_argument("--out_dir", type=str, default="../data/real_birds", help="Output dataset directory.")
    parser.add_argument("--per_species", type=int, default=20, help="Recordings to collect per species.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    audio_root = out_dir / "train_audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    tax_rows: List[Dict[str, str]] = []

    for label, common_name, sci_name in DEFAULT_SPECIES:
        rows = collect_species_recordings(
            label=label,
            common_name=common_name,
            sci_name=sci_name,
            audio_root=audio_root,
            target_count=args.per_species,
            max_workers=args.workers,
        )
        all_rows.extend(rows)
        tax_rows.append({
            "primary_label": label,
            "common_name": common_name,
            "scientific_name": sci_name,
            "class_name": "Aves",
        })

    train_df = pd.DataFrame(all_rows)
    tax_df = pd.DataFrame(tax_rows)

    train_df.to_csv(out_dir / "train.csv", index=False)
    tax_df.to_csv(out_dir / "taxonomy.csv", index=False)
    print(f"\nSaved {len(train_df)} total recordings across {len(tax_df)} species to {out_dir}")


if __name__ == "__main__":
    main()
