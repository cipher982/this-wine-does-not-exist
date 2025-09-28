from pathlib import Path

import pandas as pd

from wine_ai.data.loaders import load_training_dataframe, WineImageLoader

def test_load_training_dataframe():
    frame = load_training_dataframe()
    assert not frame.empty
    assert set(["name", "description", "price", "wine_category", "region", "image_filename", "url"]).issubset(frame.columns)


def test_image_loader_paths(tmp_path):
    loader = WineImageLoader()
    frame = load_training_dataframe()
    sample = frame[frame["image_filename"].apply(loader.exists)].iloc[0]
    path = loader.path_for(sample["image_filename"])
    assert path.exists()
