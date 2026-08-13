"""
Phase 3, Step 3: Auto-escalation.
Phase 4, Step 1: now also logs every request to SQLite.

The actual end-to-end pipeline: route -> answer cheaply -> verify -> if the
cheap answer diverged too much from the reference model, swap in the
reference answer instead. Logs every escalation event to a JSONL audit
trail, AND logs every single request (escalated or not) to the SQLite
database Phase 4's dashboard reads from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.client import send_request
from app.db.logger import log_request
from app.eval.verifier import VerificationResult, verify
from app.models import get_model
from app.response import Response
from app.routing.router import Router

ESCALATION_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "escalation_log.jsonl"

# The model every request WOULD have used with no routing at all -- per
# the original spec, this is explicitly "what it would have cost using
# GPT-4o for everything," not just whichever model happens to be priciest
# per token (we initially got this wrong: claude-sonnet is actually more
# expensive per output token than gpt-4o in our registry, which silently
# produced a baseline the spec never asked for). Hardcoded on purpose.
BASELINE_MODEL = get_model("gpt-4o")


@dataclass
class RoutedResult:
    response: Response          # the response the user actually gets back
    verification: VerificationResult | None
    escalated: bool


def log_escalation(verification: VerificationResult) -> None:
    """Append one escalation event as a JSON line -- a simple, dependency-
    free append-only audit trail, separate from the SQLite log."""
    event = {
        "prompt": verification.prompt,
        "tier": int(verification.tier),
        "original_model": verification.cheap_response.model_name,
        "escalated_model": verification.reference_response.model_name,
        "similarity": round(verification.similarity, 4),
        "quality_gap": round(verification.quality_gap, 4),
        "cost_delta_usd": round(verification.cost_delta_usd, 8),
        "timestamp": time.time(),
    }
    ESCALATION_LOG_PATH.parent.mkdir(exist_ok=True)
    with ESCALATION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def route_with_verification(prompt: str, router: Router) -> RoutedResult:
    """The full Phase 1+2+3+4 pipeline in one call: route, answer, verify,
    escalate if needed, log to SQLite. This is what Phase 5's API endpoint
    will call."""
    tier, model = router.route(prompt)
    cheap_response = send_request(prompt, model)

    verification = verify(prompt, tier, cheap_response, router)

    if verification is None or verification.passed:
        quality_score = verification.similarity if verification else None
        log_request(prompt, tier, cheap_response, BASELINE_MODEL,
                     was_escalated=False, quality_score=quality_score)
        return RoutedResult(response=cheap_response, verification=verification, escalated=False)

    # Escalate: we already have the better answer from verification, no
    # need to call anything again.
    log_escalation(verification)
    log_request(prompt, tier, verification.reference_response, BASELINE_MODEL,
                was_escalated=True, quality_score=verification.similarity)
    return RoutedResult(
        response=verification.reference_response,
        verification=verification,
        escalated=True,
    )
