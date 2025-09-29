"""Multimodal models that join text and imagery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MultimodalConfig:
    text_backbone: str = "distilgpt2"
    vision_backbone: str = "openai/clip-vit-base-patch32"


class MultimodalWineModel:
    """High-level placeholder for multimodal experimentation."""

    def __init__(self, config: MultimodalConfig | None = None) -> None:
        self.config = config or MultimodalConfig()
        self._model: Dict[str, Any] = {}

    def build(self) -> None:
        """Instantiate the multimodal backbone."""

        # In a future iteration, integrate Hugging Face multi-modal architectures.
        self._model = {
            "text_backbone": self.config.text_backbone,
            "vision_backbone": self.config.vision_backbone,
        }

    def summary(self) -> Dict[str, Any]:
        if not self._model:
            self.build()
        return self._model


__all__ = ["MultimodalWineModel", "MultimodalConfig"]
