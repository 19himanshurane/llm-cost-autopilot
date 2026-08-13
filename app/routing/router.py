"""
Phase 2, Step 4: The router.

Ties everything from Phase 2 together: takes a raw prompt, classifies its
complexity tier (Step 3's trained model), looks up which model that tier
should go to (routing_config.yaml), and returns the ModelConfig ready to
hand to Phase 1's send_request().

This is the file that turns "a pile of separate pieces" into an actual
auto-routing system: Router.route(prompt) is Phase 2's entire job in one
function call.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import yaml

from app.models import ModelConfig, get_model
from app.routing.features import featurize_prompt
from app.routing.tiers import ComplexityTier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "routing_config.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "classifier.joblib"

_TIER_KEY_BY_ENUM = {
    ComplexityTier.TIER_1_SIMPLE: "tier_1_simple",
    ComplexityTier.TIER_2_MODERATE: "tier_2_moderate",
    ComplexityTier.TIER_3_COMPLEX: "tier_3_complex",
}


class Router:
    """Loads the trained classifier + routing config once at startup, then
    reuses them for every classify()/route() call -- re-reading the YAML
    file or reloading the model from disk on every single request would be
    wasteful."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python -m scripts.train_classifier` first."
            )
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found.")

        self.classifier = joblib.load(model_path)
        with config_path.open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def classify(self, prompt: str) -> ComplexityTier:
        """Prompt in, predicted ComplexityTier out."""
        features = featurize_prompt(prompt)
        predicted_tier_int = int(self.classifier.predict([features])[0])
        return ComplexityTier(predicted_tier_int)

    def route(self, prompt: str) -> tuple[ComplexityTier, ModelConfig]:
        """Prompt in, (predicted tier, chosen ModelConfig) out. This is the
        single call the rest of the system (and later, the FastAPI
        endpoint in Phase 5) will actually use."""
        tier = self.classify(prompt)
        tier_key = _TIER_KEY_BY_ENUM[tier]
        model_name = self.config["routing"][tier_key]
        return tier, get_model(model_name)


def load_default_router() -> "Router":
    return Router()
