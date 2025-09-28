"""High-level sampling helpers for legacy wine data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..data.loaders import load_training_dataframe


@dataclass
class GenerationRequest:
    wine_type: Optional[str] = None
    sweetness: Optional[float] = None
    price_range: Optional[tuple[float, float]] = None
    seed: Optional[int] = None
    count: int = 1
    dataset_path: Optional[Path] = None


def _select_candidate(frame: pd.DataFrame, request: GenerationRequest, rng: random.Random) -> pd.Series:
    candidates = frame
    if request.wine_type:
        mask = candidates["wine_category"].astype(str).str.lower().str.contains(request.wine_type.lower())
        candidates = candidates[mask]
    if request.price_range:
        low, high = request.price_range
        candidates = candidates[(candidates["price"] >= low) & (candidates["price"] <= high)]
    if candidates.empty:
        candidates = frame
    idx = rng.randrange(len(candidates))
    return candidates.iloc[idx]


def _hydrate_response(row: pd.Series) -> Dict[str, object]:
    return {
        "name": row.get("name"),
        "description": row.get("description", ""),
        "price": float(row.get("price", 0.0) or 0.0),
        "wine_type": row.get("wine_category"),
        "region": row.get("region"),
        "image_filename": row.get("image_filename"),
        "url": row.get("url"),
    }


def generate_wines(request: GenerationRequest) -> List[Dict[str, object]]:
    """Sample wines from the processed dataset using lightweight filters."""

    frame = load_training_dataframe(request.dataset_path)
    rng = random.Random(request.seed)

    responses: List[Dict[str, object]] = []
    for _ in range(max(1, request.count)):
        row = _select_candidate(frame, request, rng)
        responses.append(_hydrate_response(row))

    return responses


__all__ = ["GenerationRequest", "generate_wines"]
