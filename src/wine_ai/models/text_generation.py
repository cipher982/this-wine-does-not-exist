"""Text generation utilities for wine descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class WineGPTConfig:
    model_name: str = "distilgpt2"
    max_new_tokens: int = 120
    temperature: float = 0.85


class WineGPT:
    """Wrapper around a Hugging Face causal language model."""

    def __init__(self, config: WineGPTConfig | None = None) -> None:
        self.config = config or WineGPTConfig()
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            self._model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def generate(self, prompt: str, *, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate a tasting description given a prompt."""

        self._ensure_model()
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        assert self._tokenizer is not None and self._model is not None  # for type-checkers

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


__all__ = ["WineGPT", "WineGPTConfig"]
