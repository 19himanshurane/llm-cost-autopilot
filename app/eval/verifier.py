"""
Phase 3, Step 2: The verifier.

Takes a prompt + the response the (cheap) routed model gave, quietly asks
the top-tier reference model ("gpt-4o", pulled from routing_config.yaml's
tier_3_complex slot -- so it stays swappable, same idea as Phase 2) the
same prompt, and compares the two answers using Step 1's similarity score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.client import send_request
from app.eval.scorer import AGREEMENT_THRESHOLD, similarity_score
from app.models import get_model
from app.response import Response
from app.routing.router import Router
from app.routing.tiers import ComplexityTier


@dataclass
class VerificationResult:
    prompt: str
    tier: ComplexityTier
    cheap_response: Response
    reference_response: Response
    similarity: float
    passed: bool
    timestamp: float = field(default_factory=time.time)

    @property
    def quality_gap(self) -> float:
        """0.0 = perfect agreement, 1.0 = total disagreement."""
        return 1.0 - self.similarity

    @property
    def cost_delta_usd(self) -> float:
        """Extra $ spent verifying, on top of what the cheap call cost."""
        return self.reference_response.cost_usd

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "tier": int(self.tier),
            "cheap_model_name": self.cheap_response.model_name,
            "reference_model_name": self.reference_response.model_name,
            "similarity": round(self.similarity, 4),
            "quality_gap": round(self.quality_gap, 4),
            "passed": self.passed,
            "verification_cost_usd": round(self.cost_delta_usd, 8),
            "timestamp": self.timestamp,
        }


def verify(prompt: str, tier: ComplexityTier, cheap_response: Response, router: Router) -> VerificationResult | None:
    """
    Re-runs `prompt` through the reference (Tier 3) model and scores
    agreement with `cheap_response`. Returns None if the cheap model WAS
    already the reference model -- no point double-checking a model
    against itself, that would always "pass" trivially.
    """
    reference_model_name = router.config["routing"]["tier_3_complex"]
    if cheap_response.model_name == reference_model_name:
        return None

    reference_model = get_model(reference_model_name)
    reference_response = send_request(prompt, reference_model)

    score = similarity_score(cheap_response.text, reference_response.text)
    passed = score >= AGREEMENT_THRESHOLD

    return VerificationResult(
        prompt=prompt,
        tier=tier,
        cheap_response=cheap_response,
        reference_response=reference_response,
        similarity=score,
        passed=passed,
    )
