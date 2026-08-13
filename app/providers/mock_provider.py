"""
Mock provider — used automatically when a real API key is missing, or when
FORCE_MOCK_MODE=true. Lets you build and test the entire routing/eval/logging
pipeline before you have paid API access.

It simulates realistic-ish token counts and latency so cost math and
dashboards behave sensibly, but the output text is a canned placeholder —
never use mock mode to evaluate actual model quality.
"""

from __future__ import annotations

import hashlib
import random
import time

from app.models import ModelConfig
from app.response import Response


def _fake_token_count(text: str) -> int:
    # Rough heuristic: ~1.3 tokens per word, same approximation OpenAI's
    # own docs suggest for English text.
    words = max(1, len(text.split()))
    return int(words * 1.3)


def call(prompt: str, model_config: ModelConfig, **kwargs) -> Response:
    start = time.monotonic()

    # Deterministic-ish "randomness" per prompt+model so repeat runs are
    # comparable, but different models still produce different-looking output.
    seed = int(hashlib.sha256(f"{prompt}:{model_config.name}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    input_tokens = _fake_token_count(prompt)
    # Cheaper/lower-tier models tend to be terser in this simulation.
    tier_multiplier = {"high": 1.4, "medium": 1.0, "low": 0.7}[model_config.quality_tier.value]
    output_tokens = int(rng.randint(40, 220) * tier_multiplier)

    output_text = (
        f"[MOCK RESPONSE from {model_config.name}] "
        f"This is a simulated {model_config.quality_tier.value}-tier answer to a "
        f"{input_tokens}-token prompt. Replace with a real API key in .env to "
        f"get actual model output."
    )

    # Simulate latency without actually sleeping the full amount in test runs.
    simulated_latency_ms = model_config.avg_latency_ms + rng.randint(-200, 400)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    cost = model_config.estimate_cost(input_tokens, output_tokens)

    return Response(
        text=output_text,
        model_name=model_config.name,
        provider=model_config.provider.value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=max(simulated_latency_ms, elapsed_ms),
        was_mocked=True,
        raw_finish_reason="mock_stop",
    )
