"""Wine AI Dataset and Models Package.

A comprehensive toolkit for working with wine image+text data,
including data loading, preprocessing, validation, and model utilities.
"""

from __future__ import annotations

from .data.loaders import WineDataset, WineImageLoader, load_training_dataframe
from .models.text_generation import WineGPT
from .models.image_generation import WineStyleGAN
from .models.multimodal import MultimodalWineModel

__all__ = [
    "WineDataset",
    "WineImageLoader",
    "WineGPT",
    "WineStyleGAN",
    "MultimodalWineModel",
    "load_training_dataframe",
]

__version__ = "1.0.0"
