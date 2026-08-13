"""Real OpenAI adapter. Only imported/used when OPENAI_API_KEY is set."""

from __future__ import annotations

import os
import time

from app.models import ModelConfig
from app.response import Response


def call(prompt: str, model_config: ModelConfig, **kwargs) -> Response:
    from openai import OpenAI  # local import so the package is optional until needed

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    start = time.monotonic()
    completion = client.chat.completions.create(
        model=model_config.model_id,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    choice = completion.choices[0]
    usage = completion.usage

    cost = model_config.estimate_cost(usage.prompt_tokens, usage.completion_tokens)

    return Response(
        text=choice.message.content or "",
        model_name=model_config.name,
        provider=model_config.provider.value,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        was_mocked=False,
        raw_finish_reason=choice.finish_reason,
    )
