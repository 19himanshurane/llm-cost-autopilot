"""Abstract provider interface. Every real provider adapter implements call()."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import ModelConfig
from app.response import Response


class BaseProvider(ABC):
    @abstractmethod
    def call(self, prompt: str, model_config: ModelConfig, **kwargs) -> Response:
        """Send `prompt` to the model and return a standardized Response."""
        raise NotImplementedError
