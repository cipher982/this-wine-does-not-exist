# Wine AI Dataset

The largest publicly available wine bottle image + text dataset, refreshed for 2025-era
multimodal research.

## Quick Start
```bash
# Install dependencies (requires Python 3.11+ and uv)
uv pip install -e .[dev]

# Load the dataset in Python
python -c "from wine_ai.data.loaders import WineDataset; ds = WineDataset.load_latest(); print(f'{len(ds.train)} train samples')"

# Validate data integrity
uv run wine-validate

# Train a language model (download model first)
uv run huggingface-cli download distilgpt2
uv run wine-train --config configs/test_training.yaml --no-wandb
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
  processed/          # Clean datasets and flat image storage
  external/           # Synthetic GPT-2 outputs and augmentation artefacts
src/wine_ai/          # Python package for data, models, training, inference
configs/              # Training configurations (YAML)
experiments/          # Historical and modern research runs
notebooks/            # Exploratory analysis and prototyping (includes quick-start)
```

## Python API
```python
from wine_ai.data.loaders import WineDataset, WineImageLoader

# Load dataset with train/val/test splits
dataset = WineDataset.load_latest()
print(f"Train: {len(dataset.train)} samples")

# Access images
loader = WineImageLoader()
sample = dataset.train.iloc[0]
if loader.exists(sample['image_filename']):
    image_path = loader.path_for(sample['image_filename'])
```

## Why This Matters
- **Irreplaceable data** – wine.com changed structure post-2020; this scrape cannot be repeated.
- **Research value** – largest paired wine image + text corpus for multimodal modelling.
- **Commercial potential** – recommendation engines, tasting note generation, virtual label design.

See `DATASET.md` for detailed documentation and `CONTRIBUTING.md` for development guidelines.
