"""Dataset loader for the Wine AI project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd
from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class WineDataset:
    """A container for train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def load_latest(cls) -> 'WineDataset':
        """Load the latest dataset from HuggingFace."""
        return load_dataset_with_splits()

    def iter_name_sequences(self) -> Iterator[str]:
        """Yield wine names for sequence modelling."""
        yield from self.train["name"].astype(str)

    def iter_description_pairs(self) -> Iterator[Tuple[str, str]]:
        """Yield (name, description) pairs for conditional generation."""
        frame = self.train[["name", "description"]].dropna()
        for _, row in frame.iterrows():
            yield row["name"], row["description"]


def load_dataset_with_splits() -> WineDataset:
    """Load wine dataset from HuggingFace."""
    print("📦 Loading dataset from HuggingFace: cipher982/wine-text-126k")
    hf_dataset = load_dataset("cipher982/wine-text-126k")

    train = hf_dataset["train"].to_pandas()
    validation = hf_dataset["validation"].to_pandas() if "validation" in hf_dataset else pd.DataFrame()
    test = hf_dataset["test"].to_pandas() if "test" in hf_dataset else pd.DataFrame()

    print(f"   ✅ Loaded: {len(train):,} train, {len(validation):,} validation, {len(test):,} test")
    return WineDataset(train=train, validation=validation, test=test)


def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    """Convert a pandas frame into a Hugging Face dataset."""
    return Dataset.from_pandas(frame, preserve_index=False)


__all__ = [
    "WineDataset",
    "load_dataset_with_splits",
    "to_hf_dataset",
]
