# Data Lineage

The Wine AI dataset was generated from a full scrape of wine.com pages conducted between
November 2019 and March 2020. The raw pickle and CSV archives are preserved in
`data/raw/legacy/` while the modern Parquet exports live in `data/raw/`.

1. `migrate_legacy_data.py` converts the legacy pickle/CSV files to Parquet.
2. `build_training_dataset` enriches the raw data with heuristic wine category and
   region labels, producing `data/processed/wine_training_dataset_v1.parquet`.
3. `train_val_test_splits.json` captures deterministic split indices for ML workflows.
4. `data_quality_report.json` summarizes validation metrics including missing images and
   price statistics.

The 12 GB `wine-dataset-20250922.tar.gz` archive remains as a long-term backup snapshot.
