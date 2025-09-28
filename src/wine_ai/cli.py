"""Top-level CLI helpers exposed via `wine-ai` entry points."""

from __future__ import annotations

from .data import build_training_dataset, migrate_legacy_sources, validate_dataset


def migrate_main() -> None:
    migrate_legacy_sources()
    path = build_training_dataset()
    print(f"Processed dataset written to {path}")


def validate_main() -> None:
    report = validate_dataset()
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")
