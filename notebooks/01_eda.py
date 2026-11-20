# %% [markdown]
# # Exploratory Data Analysis: BirdCLEF+ 2025 Bioacoustic Dataset
# 
# This interactive percent-format script examines species class balance,
# taxonomic classifications, collection source nuances, quality ratings,
# and acoustic properties of recordings in the BirdCLEF+ 2025 benchmark.

# %%
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import librosa

sns.set_theme(style="whitegrid", palette="muted")

# %% [markdown]
# ### 1. Dataset Configuration and Path Setup

# %%
# Configurable paths: Kaggle default vs local repository
DATA_DIR = Path("/kaggle/input/birdclef-2025")
if not DATA_DIR.exists():
    DATA_DIR = Path("../data/mock_raw")  # Fallback to local data

train_csv = DATA_DIR / "train.csv"
tax_csv = DATA_DIR / "taxonomy.csv"
audio_dir = DATA_DIR / "train_audio"
soundscape_dir = DATA_DIR / "train_soundscapes"

print(f"Data directory: {DATA_DIR}")
print(f"train.csv exists: {train_csv.is_file()}")
print(f"taxonomy.csv exists: {tax_csv.is_file()}")

# %% [markdown]
# ### 2. Metadata Ingestion and Taxonomic Harmony

# %%
# Read string datatypes for all label fields (preserving iNat numeric IDs)
df = pd.read_csv(train_csv, dtype={"primary_label": str, "secondary_labels": str, "filename": str, "author": str})
print(f"Total recordings in train.csv: {len(df):,}")
display_cols = ["primary_label", "secondary_labels", "collection", "rating", "filename", "author"]
print(df[[c for c in display_cols if c in df.columns]].head(5))

# Merge taxonomy if available
if tax_csv.is_file():
    tax_df = pd.read_csv(tax_csv, dtype=str)
    df = df.merge(tax_df[["primary_label", "class_name"]].drop_duplicates("primary_label"), on="primary_label", how="left")
else:
    df["class_name"] = "Aves"

df["class_name"] = df["class_name"].fillna("Aves")

# %% [markdown]
# ### 3. Class Balance and Long-Tail Species Distribution

# %%
species_counts = df["primary_label"].value_counts()
print(f"Total Unique Species: {len(species_counts)}")
print(f"Min recordings per species: {species_counts.min()}")
print(f"Max recordings per species: {species_counts.max()}")
print(f"Median recordings per species: {species_counts.median():.1f}")

plt.figure(figsize=(10, 4))
plt.plot(range(len(species_counts)), species_counts.values, color="crimson", linewidth=2)
plt.fill_between(range(len(species_counts)), species_counts.values, alpha=0.2, color="crimson")
plt.title("Species Frequency Distribution (Long-Tail Imbalance)", fontsize=13, fontweight="bold")
plt.xlabel("Species Rank")
plt.ylabel("Recording Count")
plt.axhline(10, color="black", linestyle="--", label="Default min_files_per_species (10)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Taxonomic Class Breakdown (Aves vs Non-Birds)

# %%
taxa_counts = df["class_name"].value_counts()
print("Recordings by Taxonomic Class:")
print(taxa_counts)

plt.figure(figsize=(6, 4))
sns.barplot(x=taxa_counts.index, y=taxa_counts.values, hue=taxa_counts.index, palette="viridis", legend=False)
plt.title("Distribution of Taxonomic Classes", fontsize=12, fontweight="bold")
plt.ylabel("Number of Recordings")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Recording Sources (XC vs iNaturalist vs CSA) and Quality Ratings

# %%
collection_counts = df["collection"].value_counts()
print("Collection Source Breakdown:")
print(collection_counts)

# Analyze ratings for Xeno-Canto (XC)
if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    xc_ratings = df[df["collection"] == "XC"]["rating"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="collection", hue="collection", palette="Set2", ax=ax1, legend=False)
    ax1.set_title("Recordings per Source Platform")

    sns.histplot(xc_ratings, bins=10, kde=True, color="teal", ax=ax2)
    ax2.set_title("Quality Rating Distribution (Xeno-Canto Only)")
    ax2.set_xlabel("Rating (0.0 to 5.0)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 6. Audio File Inspection and Log-Mel Spectrogram Visualization

# %%
# Select a sample file
sample_row = df.iloc[0]
sample_file = audio_dir / sample_row["filename"]
if not sample_file.is_file():
    sample_file = audio_dir / sample_row["primary_label"] / sample_row["filename"]

if sample_file.is_file():
    y, sr = sf.read(str(sample_file), dtype="float32", always_2d=True)
    y_mono = np.mean(y, axis=1) if y.shape[1] > 1 else y[:, 0]
    duration_sec = len(y_mono) / sr

    print(f"Inspecting file: {sample_row['filename']}")
    print(f"Sample Rate: {sr} Hz | Duration: {duration_sec:.2f} s | Channels: {y.shape[1]}")

    # Compute 5-second log-mel spectrogram
    crop = y_mono[: min(len(y_mono), 32000 * 5)]
    melspec = librosa.feature.melspectrogram(
        y=crop, sr=32000, n_fft=1024, hop_length=320, n_mels=128, fmin=20, fmax=16000
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max, top_db=80.0)

    fig, (ax_w, ax_m) = plt.subplots(2, 1, figsize=(10, 6))
    time_axis = np.linspace(0, len(crop) / 32000, len(crop))
    ax_w.plot(time_axis, crop, color="steelblue", linewidth=0.8)
    ax_w.set_title(f"Waveform: {sample_row['primary_label']} ({sample_row['filename']})")
    ax_w.set_ylabel("Amplitude")

    im = ax_m.imshow(log_melspec, aspect="auto", origin="lower", cmap="magma")
    ax_m.set_title("Log-Mel Spectrogram (128 mels, 32 kHz)")
    ax_m.set_xlabel("Frames")
    ax_m.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im, ax=[ax_w, ax_m], format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 7. Unlabeled Soundscape Background Noise Inspection

# %%
soundscapes = list(soundscape_dir.glob("*.ogg"))
print(f"Unlabeled Soundscape Files available: {len(soundscapes)}")

if soundscapes:
    sc = soundscapes[0]
    y_sc, sr_sc = sf.read(str(sc), dtype="float32", always_2d=True)
    y_sc_mono = np.mean(y_sc, axis=1) if y_sc.shape[1] > 1 else y_sc[:, 0]
    rms_energy = np.sqrt(np.mean(np.square(y_sc_mono)))
    print(f"Soundscape: {sc.name} | Length: {len(y_sc_mono)/sr_sc:.1f} s | Global RMS: {rms_energy:.5f}")

    # Plot 10-second slice
    slice_10s = y_sc_mono[: min(len(y_sc_mono), 32000 * 10)]
    plt.figure(figsize=(10, 3))
    plt.plot(np.linspace(0, len(slice_10s) / sr_sc, len(slice_10s)), slice_10s, color="forestgreen", linewidth=0.7)
    plt.title(f"Soundscape Ambient Background Sample: {sc.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# %%
print("Exploratory Data Analysis complete.")

