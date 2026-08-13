"""
The single entry point every other part of the system calls:

    from app.client import send_request
    response = send_request("Summarize this...", get_model("gpt-4o-mini"))

send_request() hides provider-specific plumbing behind one function and
automatically falls back to the mock provider when a real key isn't
available, so the rest of the pipeline (router, evaluator, dashboard) can be
built and tested before you've paid for any API access.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from app.models import ModelConfig, Provider
from app.providers import mock_provider, openai_provider, anthropic_provider, groq_provider
from app.response import Response

load_dotenv()


def _force_mock() -> bool:
    return os.environ.get("FORCE_MOCK_MODE", "false").lower() == "true"


def _should_mock(model_config: ModelConfig) -> bool:
    if _force_mock():
        return True
    if model_config.provider == Provider.OPENAI:
        return not os.environ.get("OPENAI_API_KEY")
    if model_config.provider == Provider.ANTHROPIC:
        return not os.environ.get("ANTHROPIC_API_KEY")
    if model_config.provider == Provider.GROQ:
        return not os.environ.get("GROQ_API_KEY")
    raise ValueError(f"Unhandled provider: {model_config.provider}")


def send_request(prompt: str, model_config: ModelConfig, **kwargs) -> Response:
    """
    Send `prompt` to the model described by `model_config` and return a
    standardized Response, regardless of which provider backs it.

    Extra `kwargs` are passed through to the underlying provider SDK call
    (e.g. temperature, max_tokens) - ignored by the mock provider.
    """
    if _should_mock(model_config):
        return mock_provider.call(prompt, model_config, **kwargs)

    if model_config.provider == Provider.OPENAI:
        return openai_provider.call(prompt, model_config, **kwargs)
    if model_config.provider == Provider.ANTHROPIC:
        return anthropic_provider.call(prompt, model_config, **kwargs)
    if model_config.provider == Provider.GROQ:
        return groq_provider.call(prompt, model_config, **kwargs)

    raise ValueError(f"Unhandled provider: {model_config.provider}")
