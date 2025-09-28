#!/usr/bin/env python3
"""Convert legacy data dumps into the modern Parquet format."""

from __future__ import annotations

from wine_ai.data import build_training_dataset, migrate_legacy_sources


def main() -> None:
    print("Converting legacy sources...")
    migrate_legacy_sources()
    print("Building training dataset...")
    dataset_path = build_training_dataset()
    print(f"Processed dataset written to {dataset_path}")


if __name__ == "__main__":
    main()
