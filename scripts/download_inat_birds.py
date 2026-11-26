"""Download research-grade bird vocalizations from iNaturalist into BirdCLEF-compatible layout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import soundfile as sf

# Curated species presets with abundant research-grade iNaturalist recordings
SPECIES_PRESETS: Dict[str, List[Tuple[str, str, str]]] = {
    "indian16": [
        ("indian_peafowl", "Indian Peafowl", "Pavo cristatus"),
        ("asian_koel", "Asian Koel", "Eudynamys scolopaceus"),
        ("house_crow", "House Crow", "Corvus splendens"),
        ("common_myna", "Common Myna", "Acridotheres tristis"),
        ("rose_ringed_parakeet", "Rose-ringed Parakeet", "Psittacula krameri"),
        ("house_sparrow", "House Sparrow", "Passer domesticus"),
        ("rock_pigeon", "Rock Pigeon", "Columba livia"),
        ("spotted_dove", "Spotted Dove", "Spilopelia chinensis"),
        ("coppersmith_barbet", "Coppersmith Barbet", "Psilopogon haemacephalus"),
        ("red_vented_bulbul", "Red-vented Bulbul", "Pycnonotus cafer"),
        ("white_throated_kingfisher", "White-throated Kingfisher", "Halcyon smyrnensis"),
        ("oriental_magpie_robin", "Oriental Magpie-Robin", "Copsychus saularis"),
        ("common_tailorbird", "Common Tailorbird", "Orthotomus sutorius"),
        ("black_kite", "Black Kite", "Milvus migrans"),
        ("greater_coucal", "Greater Coucal", "Centropus sinensis"),
        ("common_cuckoo", "Common Cuckoo", "Cuculus canorus"),
    ],
    "core8": [
        ("common_blackbird", "Common Blackbird", "Turdus merula"),
        ("northern_cardinal", "Northern Cardinal", "Cardinalis cardinalis"),
        ("american_crow", "American Crow", "Corvus brachyrhynchos"),
        ("common_cuckoo", "Common Cuckoo", "Cuculus canorus"),
        ("house_sparrow", "House Sparrow", "Passer domesticus"),
        ("mourning_dove", "Mourning Dove", "Zenaida macroura"),
        ("great_tit", "Great Tit", "Parus major"),
        ("american_robin", "American Robin", "Turdus migratorius"),
    ],
}

DEFAULT_SPECIES = SPECIES_PRESETS["indian16"]
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
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """Collect target_count distinct-recordist audio clips for one bird species."""
    species_dir = audio_root / label
    species_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[str, str, str]] = []
    seen_authors = set()
    seen_obs = set()

    for page in range(1, 7):
        try:
            results = fetch_observations(sci_name, page=page, per_page=80)
        except Exception as e:
            print(f"[{label}] Warning: failed page {page}: {e}", flush=True)
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
            sound_url = next((s.get("file_url") for s in sounds if s.get("file_url")), None)
            if not sound_url:
                continue

            seen_obs.add(obs_id)
            seen_authors.add(author)
            candidates.append((obs_id, author, sound_url))
            if len(candidates) >= target_count * 2:
                break
        if len(candidates) >= target_count * 2:
            break
        time.sleep(0.25)

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


def _slugify_species_label(common_name: str, sci_name: str, seen_labels: set[str]) -> str:
    """Create a unique snake_case primary_label from common and scientific names."""
    base = re.sub(r"[^a-z0-9]+", "_", common_name.lower()).strip("_")
    if not base:
        base = re.sub(r"[^a-z0-9]+", "_", sci_name.lower()).strip("_") or "unknown_species"
    if base not in seen_labels:
        return base
    sci_slug = re.sub(r"[^a-z0-9]+", "_", sci_name.lower()).strip("_")
    if sci_slug and sci_slug not in seen_labels:
        return sci_slug
    idx = 2
    while f"{base}_{idx}" in seen_labels:
        idx += 1
    return f"{base}_{idx}"


def fetch_india_top_species(top_n: int, place_id: int = 6681) -> List[Tuple[str, str, str]]:
    """Query iNaturalist for the top N most-recorded bird species with audio in India."""
    if top_n < 1:
        raise ValueError(f"--auto_india_top must be >= 1, got {top_n}")

    species: List[Tuple[str, str, str]] = []
    seen_labels: set[str] = set()
    seen_sci: set[str] = set()
    page = 1

    while len(species) < top_n:
        per_page = min(200, max(50, top_n - len(species)))
        params = urllib.parse.urlencode({
            "taxon_id": 3,
            "place_id": place_id,
            "sounds": "true",
            "quality_grade": "research",
            "per_page": per_page,
            "page": page,
        })
        url = f"https://api.inaturalist.org/v1/observations/species_counts?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "BirdAudioClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            taxon = item.get("taxon") or {}
            sci_name = str(taxon.get("name", "")).strip()
            if not sci_name or sci_name in seen_sci:
                continue
            common_name = str(taxon.get("preferred_common_name") or sci_name).strip()
            label = _slugify_species_label(common_name, sci_name, seen_labels)
            seen_labels.add(label)
            seen_sci.add(sci_name)
            species.append((label, common_name, sci_name))
            if len(species) >= top_n:
                break

        if len(results) < per_page:
            break
        page += 1
        time.sleep(0.3)

    print(
        f"Resolved top {len(species)} Indian bird species with research-grade audio "
        f"(iNaturalist place_id={place_id}).",
        flush=True,
    )
    return species


def load_species_list(
    preset: str,
    species_csv: Optional[str],
    auto_india_top: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """Resolve species list from custom CSV, auto India top-N query, or built-in preset."""
    if species_csv:
        df = pd.read_csv(species_csv, dtype=str)
        required = {"primary_label", "common_name", "scientific_name"}
        if not required.issubset(df.columns):
            raise ValueError(f"Custom species CSV must contain columns: {sorted(required)}")
        return [
            (str(r["primary_label"]).strip(), str(r["common_name"]).strip(), str(r["scientific_name"]).strip())
            for _, r in df.iterrows()
        ]
    if auto_india_top is not None:
        return fetch_india_top_species(auto_india_top)
    return SPECIES_PRESETS[preset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real bird audio dataset from iNaturalist.")
    parser.add_argument("--out_dir", type=str, default="../data/indian_birds", help="Output dataset directory.")
    parser.add_argument("--per_species", type=int, default=20, help="Recordings to collect per species.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download threads.")
    parser.add_argument("--preset", type=str, default="indian16", choices=sorted(SPECIES_PRESETS.keys()), help="Built-in species preset.")
    parser.add_argument(
        "--auto_india_top",
        type=int,
        default=None,
        help="Dynamically fetch the top N most-recorded Indian bird species with audio from iNaturalist (e.g. 50 or 100).",
    )
    parser.add_argument("--species_csv", type=str, default=None, help="Optional CSV with primary_label,common_name,scientific_name.")
    args = parser.parse_args()

    species_list = load_species_list(args.preset, args.species_csv, auto_india_top=args.auto_india_top)
    out_dir = Path(args.out_dir).resolve()
    audio_root = out_dir / "train_audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    tax_rows: List[Dict[str, str]] = []

    for label, common_name, sci_name in species_list:
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
