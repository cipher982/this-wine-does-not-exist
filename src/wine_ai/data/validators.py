"""Data validation utilities for ensuring dataset integrity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .loaders import IMAGE_ROOT, load_training_dataframe


@dataclass
class DataQualityReport:
    """Structured representation of validation results."""

    duplicate_names: List[str]
    invalid_image_paths: List[str]
    price_outliers: List[int]
    empty_descriptions: int
    missing_image_files: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "duplicate_names": self.duplicate_names,
            "invalid_image_paths": self.invalid_image_paths,
            "price_outliers": self.price_outliers,
            "empty_descriptions": self.empty_descriptions,
            "missing_image_files": self.missing_image_files,
        }


def validate_dataset(frame: pd.DataFrame | None = None, *, price_bounds: tuple[float, float] = (5.0, 5000.0)) -> DataQualityReport:
    """Run a series of validation checks over the processed dataset."""

    data = frame if frame is not None else load_training_dataframe()

    duplicates = data["name"].value_counts()
    duplicate_names = duplicates[duplicates > 1].index.tolist()

    invalid_paths = [
        path
        for path in data["image_filename"].astype(str)
        if not path.endswith((".jpg", ".jpeg", ".png"))
    ]

    min_price, max_price = price_bounds
    outlier_indices = data.index[(data["price"].notna()) & ((data["price"] < min_price) | (data["price"] > max_price))]

    empty_descriptions = int((data["description"].astype(str).str.strip() == "").sum())

    image_root = IMAGE_ROOT
    missing_files = int(sum(not (image_root / fn).exists() for fn in data["image_filename"].astype(str)))

    return DataQualityReport(
        duplicate_names=duplicate_names,
        invalid_image_paths=invalid_paths,
        price_outliers=outlier_indices.tolist(),
        empty_descriptions=empty_descriptions,
        missing_image_files=missing_files,
    )


__all__ = ["DataQualityReport", "validate_dataset"]
