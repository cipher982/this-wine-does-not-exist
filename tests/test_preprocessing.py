from wine_ai.data.preprocessors import build_training_dataset, migrate_legacy_sources
from wine_ai.data.loaders import RAW_DATASET, PROCESSED_DATASET

def test_migrate_and_build(tmp_path):
    migrate_legacy_sources()
    assert RAW_DATASET.exists()
    build_training_dataset()
    assert PROCESSED_DATASET.exists()
