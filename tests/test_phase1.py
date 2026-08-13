import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.client import send_request
from app.models import MODEL_REGISTRY, get_model


def test_registry_has_five_models():
    assert len(MODEL_REGISTRY) == 6


def test_get_model_unknown_raises():
    try:
        get_model("does-not-exist")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cost_estimate_scales_with_tokens():
    model = get_model("gpt-4o")
    assert model.estimate_cost(1000, 0) < model.estimate_cost(2000, 0)


def test_send_request_falls_back_to_mock_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FORCE_MOCK_MODE", "true")
    model = get_model("gpt-4o-mini")
    response = send_request("hello world", model)
    assert response.was_mocked is True
    assert response.model_name == "gpt-4o-mini"
    assert response.cost_usd >= 0
    assert response.input_tokens > 0

