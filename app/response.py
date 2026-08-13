"""Standardized response object every provider adapter must return."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Response:
    text: str
    model_name: str            # registry key, e.g. "gpt-4o"
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    was_mocked: bool = False
    raw_finish_reason: str | None = None
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "model_name": self.model_name,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_ms": self.latency_ms,
            "was_mocked": self.was_mocked,
            "raw_finish_reason": self.raw_finish_reason,
            "timestamp": self.timestamp,
        }
