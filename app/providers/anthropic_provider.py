"""Real Anthropic adapter. Only imported/used when ANTHROPIC_API_KEY is set."""

from __future__ import annotations

import os
import time

from app.models import ModelConfig
from app.response import Response


def call(prompt: str, model_config: ModelConfig, **kwargs) -> Response:
    import anthropic  # local import so the package is optional until needed

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    start = time.monotonic()
    message = client.messages.create(
        model=model_config.model_id,
        max_tokens=kwargs.pop("max_tokens", 1024),
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    text = "".join(block.text for block in message.content if block.type == "text")
    cost = model_config.estimate_cost(
        message.usage.input_tokens, message.usage.output_tokens
    )

    return Response(
        text=text,
        model_name=model_config.name,
        provider=model_config.provider.value,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        was_mocked=False,
        raw_finish_reason=message.stop_reason,
    )
