# Wine Dataset Documentation

## Data Lineage
This dataset originated from a comprehensive wine.com scrape conducted between
November 2019 and March 2020. The original pickle/CSV exports are preserved under
`data/raw/legacy/` while the modern Parquet files live in `data/raw/`.

## File Formats
- **Primary format**: Parquet (Snappy compressed, columnar, fast)
- **Images**: JPEG (various sizes between 300–800px)
- **Legacy format**: Pickle/CSV (deprecated, retained for provenance only)

## Core Files
| Path | Purpose |
| ---- | ------- |
| `data/raw/wine_scraped_125k.parquet` | Canonical wine metadata (125,787 rows) |
| `data/raw/wine_descriptions_230k.parquet` | Supplemental descriptions (230,487 rows) |
| `data/processed/wine_training_dataset_v1.parquet` | Clean training dataset with categories & regions |
| `data/processed/wine_images_organized/` | Structured image storage with symlinks for category/region |
| `data/processed/train_val_test_splits.json` | Deterministic dataset splits |
| `data/processed/data_quality_report.json` | Validation summary (coverage, prices, categories) |

## Dataset Splits
- **Training**: 86,257 wines (80%)
- **Validation**: 10,782 wines (10%)
- **Test**: 10,782 wines (10%)
- **Missing images**: 17,966 wines (documented in `validation_manifest.csv`)

## Quality Issues
- Some descriptions contain residual HTML or marketing boilerplate.
- Price data reflects 2019–2020 retail values.
- Image quality varies; a subset of downloads failed or produced placeholders.
- Regional tags are heuristic and should be confirmed before production use.

## Rebuilding the Dataset
```bash
uv run python scripts/migrate_legacy_data.py
uv run python scripts/validate_data.py
```

## Metadata
Additional lineage information is stored in `data/raw/metadata/`:
- `scraping_log.json`
- `data_dictionary.json`

## Contact
For access to the full dataset or archival tarballs contact David Rose
(`<david@example.com>`).
