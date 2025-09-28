#!/usr/bin/env python3
"""Run data validation checks against the processed dataset."""

from __future__ import annotations

from wine_ai.data import validate_dataset


def main() -> None:
    report = validate_dataset()
    payload = report.as_dict()
    print("Duplicate names:", len(payload["duplicate_names"]))
    print("Invalid image paths:", len(payload["invalid_image_paths"]))
    print("Price outliers:", len(payload["price_outliers"]))
    print("Empty descriptions:", payload["empty_descriptions"])
    print("Missing image files:", payload["missing_image_files"])


if __name__ == "__main__":
    main()
