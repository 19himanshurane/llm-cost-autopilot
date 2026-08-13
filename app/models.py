"""
Model registry for LLM Cost Autopilot.

Defines the ModelConfig dataclass and populates it with real-world pricing
for the models we route across. Pricing is per-token (converted from the
common "per 1M tokens" quotes providers publish) so cost math downstream
never has to think about units.

IMPORTANT: LLM provider pricing changes frequently. The numbers below were
accurate as of mid-2025 list pricing. Before trusting the dashboard's dollar
figures, check https://openai.com/api/pricing, https://www.anthropic.com/pricing,
and https://console.groq.com/docs/models and update the values below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ModelConfig:
    """Everything the router needs to know about one callable model."""

    name: str                      # short internal key, e.g. "gpt-4o"
    provider: Provider
    model_id: str                  # the string the provider's API expects
    cost_per_input_token: float    # USD
    cost_per_output_token: float   # USD
    avg_latency_ms: int            # rough baseline, refined by real measurements later
    quality_tier: QualityTier
    context_window: int = 128_000
    notes: str = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


def _per_million(usd_per_million: float) -> float:
    return usd_per_million / 1_000_000


# ---------------------------------------------------------------------------
# Registry. Add/edit models here - nothing else in the codebase should
# hardcode a model name or price.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_input_token=_per_million(2.50),
        cost_per_output_token=_per_million(10.00),
        avg_latency_ms=1800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        notes="Flagship OpenAI model. Reserve for Tier 3 (complex reasoning).",
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_input_token=_per_million(0.15),
        cost_per_output_token=_per_million(0.60),
        avg_latency_ms=900,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        notes="Cheap, fast. Good Tier 2 default.",
    ),
    "claude-sonnet": ModelConfig(
        name="claude-sonnet",
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        cost_per_input_token=_per_million(3.00),
        cost_per_output_token=_per_million(15.00),
        avg_latency_ms=1600,
        quality_tier=QualityTier.HIGH,
        context_window=200_000,
        notes="Strong reasoning + writing. Tier 2/3 depending on budget.",
    ),
    "claude-haiku": ModelConfig(
        name="claude-haiku",
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-20250514",
        cost_per_input_token=_per_million(0.80),
        cost_per_output_token=_per_million(4.00),
        avg_latency_ms=700,
        quality_tier=QualityTier.LOW,
        context_window=200_000,
        notes="Cheapest Anthropic option.",
    ),
    "llama3-groq": ModelConfig(
        name="llama3-groq",
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-versatile",
        cost_per_input_token=_per_million(0.59),
        cost_per_output_token=_per_million(0.79),
        avg_latency_ms=400,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        notes="Free-tier cloud model via Groq (rate-limited, not token-limited). Good Tier 2 option.",
    ),
    "llama3-8b-groq": ModelConfig(
        name="llama3-8b-groq",
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        cost_per_input_token=_per_million(0.05),
        cost_per_output_token=_per_million(0.08),
        avg_latency_ms=250,
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        notes="Free-tier cloud model via Groq. Small/fast - real Tier 1 option, no local compute needed.",
    ),
}


def get_model(name: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'. Known models: {list(MODEL_REGISTRY)}"
        ) from exc


def models_by_tier(tier: QualityTier) -> list[ModelConfig]:
    return [m for m in MODEL_REGISTRY.values() if m.quality_tier == tier]
