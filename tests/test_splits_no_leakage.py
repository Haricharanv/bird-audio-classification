"""Unit tests verifying leakage-free splits across filenames and authors."""

import pandas as pd
import pytest

from src.data.splits import (
    sanitize_author_groups,
    split_by_author,
    split_by_filename,
)
from src.utils.config import Config


@pytest.fixture
def mock_metadata():
    """Construct mock metadata with varied species counts and overlapping authors."""
    data = []
    # 5 frequent species (10 files each)
    for sp_idx in range(5):
        sp_name = f"species_{sp_idx}"
        for f_idx in range(10):
            # 2 authors per species
            author = f"Author_{sp_idx}_{f_idx % 2}"
            data.append({
                "filename": f"rec_{sp_name}_{f_idx}.ogg",
                "primary_label": sp_name,
                "author": author,
                "collection": "XC",
            })
    # 1 rare species with 2 files (too few to stratify across 3 splits)
    data.append({
        "filename": "rec_rare_0.ogg",
        "primary_label": "species_rare",
        "author": "RareAuthor_A",
        "collection": "iNat",
    })
    data.append({
        "filename": "rec_rare_1.ogg",
        "primary_label": "species_rare",
        "author": "RareAuthor_B",
        "collection": "iNat",
    })
    # Several recordings with Unknown / NaN author
    for u_idx in range(4):
        data.append({
            "filename": f"rec_unknown_{u_idx}.ogg",
            "primary_label": "species_0",
            "author": "Unknown" if u_idx % 2 == 0 else None,
            "collection": "CSA",
        })

    return pd.DataFrame(data)


def test_split_by_filename_no_leakage(mock_metadata):
    df_split = split_by_filename(
        mock_metadata,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    
    assert "split" in df_split.columns
    assert set(df_split["split"].unique()).issubset({"train", "val", "test"})

    train_files = set(df_split[df_split["split"] == "train"]["filename"])
    val_files = set(df_split[df_split["split"] == "val"]["filename"])
    test_files = set(df_split[df_split["split"] == "test"]["filename"])

    # Strict disjointness
    assert len(train_files.intersection(val_files)) == 0, "Filename overlap between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Filename overlap between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Filename overlap between val and test!"

    # Rare species (< 3 recordings) must go to train-only
    rare_splits = set(df_split[df_split["primary_label"] == "species_rare"]["split"])
    assert rare_splits == {"train"}, "Rare species with < 3 files must be placed in train split only!"


def test_split_by_author_no_leakage(mock_metadata):
    df_split = split_by_author(
        mock_metadata,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    sanitized_authors = sanitize_author_groups(df_split)
    train_authors = set(sanitized_authors[df_split["split"] == "train"])
    val_authors = set(sanitized_authors[df_split["split"] == "val"])
    test_authors = set(sanitized_authors[df_split["split"] == "test"])

    # Strict author disjointness
    assert len(train_authors.intersection(val_authors)) == 0, "Author overlap between train and val!"
    assert len(train_authors.intersection(test_authors)) == 0, "Author overlap between train and test!"
    assert len(val_authors.intersection(test_authors)) == 0, "Author overlap between val and test!"

    # Filenames must also be strictly disjoint
    train_files = set(df_split[df_split["split"] == "train"]["filename"])
    val_files = set(df_split[df_split["split"] == "val"]["filename"])
    test_files = set(df_split[df_split["split"] == "test"]["filename"])

    assert len(train_files.intersection(val_files)) == 0
    assert len(train_files.intersection(test_files)) == 0
    assert len(val_files.intersection(test_files)) == 0


def test_unknown_authors_treated_as_individuals(mock_metadata):
    sanitized = sanitize_author_groups(mock_metadata)
    unknown_rows = mock_metadata[
        mock_metadata["author"].fillna("Unknown").str.lower().isin(["unknown", "none", "nan"])
    ]
    
    # Assert every unknown author row received a distinct synthetic ID
    assigned_ids = sanitized.loc[unknown_rows.index]
    assert assigned_ids.nunique() == len(unknown_rows), "Unknown authors must not be grouped together!"

