"""Classification utilities for wine categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ClassificationConfig:
    max_iter: int = 200


class WineClassifier:
    """Lightweight scikit-learn classifier for wine types."""

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig()
        self.model = LogisticRegression(max_iter=self.config.max_iter)

    def fit(self, features: np.ndarray, labels: Iterable[str]) -> None:
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> List[str]:
        return self.model.predict(features).tolist()


__all__ = ["WineClassifier", "ClassificationConfig"]
