# Data Classification: Raw vs Processed

## Raw (Immutable)
- `data/raw/wine_scraped_125k.parquet`
- `data/raw/wine_descriptions_230k.parquet`
- `data/raw/legacy/` (original pickles/CSVs)
- `data/raw/metadata/` (scraping log, data dictionary)

## Processed (Regenerable)
- `data/processed/wine_training_dataset_v1.parquet`
- `data/processed/train_val_test_splits.json`
- `data/processed/data_quality_report.json`
- `data/processed/wine_images_organized/`

## External / Generated
- `data/external/synthetic_wines_gpt2/`

## Archive Strategy
- Keep `wine-dataset-20250922.tar.gz` plus the fresh `tmp/wine-project-backup-20250928.tar.gz`.
- Regenerate processed artefacts via `wine-migrate` CLI when code changes.
