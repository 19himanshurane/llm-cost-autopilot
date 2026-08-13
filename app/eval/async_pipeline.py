"""
Phase 3, Step 2 (completed): make verification genuinely asynchronous.
Phase 4, Step 1: log to SQLite immediately, update once verification finishes.
Phase 5: now returns (tier, response) so the API layer can report which
tier a request was classified into, not just the final answer.

route_with_verification() in pipeline.py is synchronous -- it waits for
the reference-model call before returning anything. route_with_async_
verification() fixes this: it answers with the cheap model and returns
THAT immediately, while verification (and escalation logging) keeps
running in a background thread the caller never waits on.
"""

from __future__ import annotations

import threading

from app.client import send_request
from app.db.logger import log_request, update_escalation
from app.eval.pipeline import BASELINE_MODEL, log_escalation
from app.eval.verifier import verify
from app.response import Response
from app.routing.router import Router
from app.routing.tiers import ComplexityTier


def _verify_in_background(prompt: str, tier, cheap_response: Response, router: Router, row_id: int) -> None:
    verification = verify(prompt, tier, cheap_response, router)
    if verification is None:
        return
    if not verification.passed:
        log_escalation(verification)
    update_escalation(row_id, was_escalated=not verification.passed, quality_score=verification.similarity)


def route_with_async_verification(prompt: str, router: Router) -> tuple[ComplexityTier, Response]:
    """Returns (tier, cheap_response) IMMEDIATELY. Verification, escalation
    logging, and the SQLite quality-score update all happen in a
    background thread the caller never waits on."""
    tier, model = router.route(prompt)
    cheap_response = send_request(prompt, model)

    row_id = log_request(prompt, tier, cheap_response, BASELINE_MODEL,
                          was_escalated=False, quality_score=None)

    thread = threading.Thread(
        target=_verify_in_background,
        args=(prompt, tier, cheap_response, router, row_id),
        daemon=True,
    )
    thread.start()

    return tier, cheap_response
