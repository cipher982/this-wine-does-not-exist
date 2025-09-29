"""Simple data augmentation helpers for experimentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import random


@dataclass
class AugmentationConfig:
    noise_probability: float = 0.1
    shuffle_probability: float = 0.05


def jitter_description(description: str, config: AugmentationConfig | None = None) -> str:
    """Inject lightweight variation into a wine description."""

    config = config or AugmentationConfig()
    tokens = description.split()
    rng = random.Random(42)

    if not tokens:
        return description

    augmented: List[str] = []
    for token in tokens:
        if rng.random() < config.noise_probability:
            augmented.append(token.upper())
        else:
            augmented.append(token)

    if rng.random() < config.shuffle_probability:
        rng.shuffle(augmented)

    return " ".join(augmented)


def augment_corpus(corpus: Iterable[str], config: AugmentationConfig | None = None) -> List[str]:
    """Apply augmentation to a collection of descriptions."""

    config = config or AugmentationConfig()
    return [jitter_description(item, config=config) for item in corpus]


__all__ = ["AugmentationConfig", "jitter_description", "augment_corpus"]
