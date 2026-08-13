"""
Manual smoke test for Phase 3: proves route_with_verification() works both
when the cheap model's answer passes verification, and when it's forced to
fail (and correctly escalates).

Run from the project root:
    python -m scripts.check_phase3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routing.router import load_default_router
from app.eval.pipeline import route_with_verification
import app.eval.verifier as verifier_module

PROMPT = "What is the capital of France?"


def main() -> None:
    router = load_default_router()

    print("=== Normal run (real threshold) ===")
    result = route_with_verification(PROMPT, router)
    print(f"escalated: {result.escalated}")
    print(f"final model used: {result.response.model_name}")
    if result.verification:
        print(f"similarity: {result.verification.similarity:.4f}")

    print("\n=== Forced-failure run (threshold cranked to 0.99) ===")
    verifier_module.AGREEMENT_THRESHOLD = 0.99
    result2 = route_with_verification(PROMPT, router)
    print(f"escalated: {result2.escalated}")
    print(f"final model used: {result2.response.model_name}")
    if result2.verification:
        print(f"similarity: {result2.verification.similarity:.4f}")


if __name__ == "__main__":
    main()
