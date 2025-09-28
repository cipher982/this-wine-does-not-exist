# Contributing

1. **Environment**
   - Install dependencies with `uv pip install -e .[dev]`.
   - Use Python 3.11+ for parity with the training pipeline.

2. **Data Safety**
   - Never modify files under `data/raw/legacy/`.
   - Regenerate processed assets via `scripts/migrate_legacy_data.py` instead of editing Parquet manually.

3. **Code Style**
   - Run `ruff` and `black` before committing.
   - Add type hints and minimal docstrings for non-obvious logic.

4. **Testing**
   - Execute `pytest` for unit tests.
   - Use the validation script to ensure dataset integrity after major changes.

5. **Pull Requests**
   - Summarize data impacts, including counts of added/removed records.
   - Attach updated validation reports when touching ingestion pipelines.
