# Wine AI Dataset

The largest publicly available wine bottle image + text dataset, refreshed for 2025-era
multimodal research.

## Quick Start
```bash
# Install dependencies (requires Python 3.11+ and uv)
uv pip install -e .[dev]

# Regenerate processed datasets if needed
uv run python scripts/migrate_legacy_data.py

# Inspect the dataset and validate integrity
uv run python scripts/validate_data.py

# Train a language model using the modern pipeline
uv run wine-train --config configs/train_description.yaml
```

## Dataset Highlights
- 125,787 unique wines with complete metadata
- 107,821 validated bottle images (17,966 missing locally)
- Rich tasting descriptions, URLs, prices, and HTML archives
- Deterministic train/validation/test splits stored in `data/processed/train_val_test_splits.json`

## Repository Layout
```
data/
  raw/                # Immutable Parquet exports + metadata
  processed/          # Clean datasets and organized image storage
  external/           # Synthetic GPT-2 outputs and augmentation artefacts
src/wine_ai/          # Python package for data, models, training, inference
scripts/              # Utility scripts (migration, validation, env setup)
experiments/          # Historical and modern research runs
notebooks/            # Exploratory analysis and prototyping
```

## Why This Matters
- **Irreplaceable data** – wine.com changed structure post-2020; this scrape cannot be repeated.
- **Research value** – largest paired wine image + text corpus for multimodal modelling.
- **Commercial potential** – recommendation engines, tasting note generation, virtual label design.

See `DATASET.md` for detailed documentation and `CONTRIBUTING.md` for development guidelines.
