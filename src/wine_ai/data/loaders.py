"""Canonical dataset loaders for the Wine AI project."""

from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import pandas as pd

from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RAW_DATASET = DATA_ROOT / "raw" / "wine_scraped_125k.parquet"
PROCESSED_DATASET = DATA_ROOT / "processed" / "wine_training_dataset_v1.parquet"
DESCRIPTIONS_DATASET = DATA_ROOT / "raw" / "wine_descriptions_230k.parquet"
IMAGE_ROOT = DATA_ROOT / "processed" / "wine_images_organized" / "raw_flat" / "wine_images"
SPLITS_PATH = DATA_ROOT / "processed" / "train_val_test_splits.json"


@dataclass(frozen=True)
class WineDataset:
    """A container for train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def load_latest(cls) -> 'WineDataset':
        """Load the latest processed dataset with train/validation/test splits."""
        return load_dataset_with_splits()

    def iter_name_sequences(self) -> Iterator[str]:
        """Yield wine names for sequence modelling."""

        yield from self.train["name"].astype(str)

    def iter_description_pairs(self) -> Iterator[Tuple[str, str]]:
        """Yield (name, description) pairs for conditional generation."""

        frame = self.train[["name", "description"]].dropna()
        for _, row in frame.iterrows():
            yield row["name"], row["description"]


class WineImageLoader:
    """Utility for resolving paths to processed wine images."""

    def __init__(self, image_root: Path | None = None) -> None:
        self._image_root = image_root or IMAGE_ROOT

    @property
    def image_root(self) -> Path:
        return self._image_root

    def exists(self, image_filename: str) -> bool:
        return (self._image_root / image_filename).exists()

    def path_for(self, image_filename: str) -> Path:
        path = self._image_root / image_filename
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    def iter_existing(self, filenames: Iterable[str]) -> Iterator[Path]:
        for filename in filenames:
            path = self._image_root / filename
            if path.exists():
                yield path


def load_training_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical training dataframe from Parquet."""

    dataset_path = path or PROCESSED_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    frame = pd.read_parquet(dataset_path)
    expected_columns = {"name", "description", "price", "wine_category", "region", "image_filename", "url"}
    missing = expected_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset missing columns: {missing}")
    return frame


def load_dataset_with_splits(path: Path | None = None, use_hf: bool = True) -> WineDataset:
    """Load wine dataset with train/validation/test splits.

    Args:
        path: Local parquet file path (fallback)
        use_hf: Use HuggingFace dataset if True (default)
    """

    if use_hf:
        try:
            print("📦 Loading dataset from HuggingFace: cipher982/wine-text-126k")
            hf_dataset = load_dataset("cipher982/wine-text-126k")

            # Convert HF dataset splits to pandas DataFrames
            train = hf_dataset["train"].to_pandas()
            validation = hf_dataset["validation"].to_pandas() if "validation" in hf_dataset else pd.DataFrame()
            test = hf_dataset["test"].to_pandas() if "test" in hf_dataset else pd.DataFrame()

            print(f"   ✅ Loaded: {len(train):,} train, {len(validation):,} validation, {len(test):,} test")
            return WineDataset(train=train, validation=validation, test=test)

        except Exception as e:
            print(f"⚠️  HuggingFace dataset failed: {e}")
            print("   Falling back to local files...")

    # Fallback to local file loading
    frame = load_training_dataframe(path)

    if SPLITS_PATH.exists():
        with SPLITS_PATH.open("r", encoding="utf-8") as fh:
            indices = json.load(fh)
        train = frame.loc[indices["train"]].reset_index(drop=True)
        validation = frame.loc[indices["validation"]].reset_index(drop=True)
        test = frame.loc[indices["test"]].reset_index(drop=True)
    else:  # fallback to naive split
        shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_cut = int(len(shuffled) * 0.8)
        val_cut = train_cut + int(len(shuffled) * 0.1)
        train = shuffled.iloc[:train_cut]
        validation = shuffled.iloc[train_cut:val_cut]
        test = shuffled.iloc[val_cut:]

    return WineDataset(train=train, validation=validation, test=test)


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load the legacy raw dataset from Parquet or pickle backup."""

    dataset_path = path or RAW_DATASET
    if dataset_path.exists():
        return pd.read_parquet(dataset_path)

    legacy_pickle = DATA_ROOT / "raw" / "legacy" / "wine_scraped_125k.pickle.gz"
    if not legacy_pickle.exists():
        raise FileNotFoundError(f"No raw dataset found at {dataset_path} or {legacy_pickle}")

    with gzip.open(legacy_pickle, "rb") as fh:
        frame = pickle.load(fh)  # type: ignore[name-defined]
    if not isinstance(frame, pd.DataFrame):  # pragma: no cover - legacy guard
        frame = pd.DataFrame(frame)
    return frame


def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    """Convert a pandas frame into a Hugging Face dataset."""
    return Dataset.from_pandas(frame, preserve_index=False)


__all__ = [
    "WineDataset",
    "WineImageLoader",
    "load_training_dataframe",
    "load_dataset_with_splits",
    "load_raw_dataframe",
    "to_hf_dataset",
]
