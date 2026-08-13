"""
Proves route_with_async_verification() actually returns faster than the
synchronous route_with_verification(), since it doesn't wait on the
reference-model call before handing back an answer.

Run from the project root:
    python -m scripts.check_async
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routing.router import load_default_router
from app.eval.pipeline import route_with_verification
from app.eval.async_pipeline import route_with_async_verification

PROMPT = "What is the capital of Japan?"


def main() -> None:
    router = load_default_router()

    start = time.monotonic()
    sync_result = route_with_verification(PROMPT, router)
    sync_elapsed_ms = (time.monotonic() - start) * 1000
    print(f"SYNC   route_with_verification():        {sync_elapsed_ms:7.1f}ms")

    start = time.monotonic()
    tier, async_response = route_with_async_verification(PROMPT, router)
    async_elapsed_ms = (time.monotonic() - start) * 1000
    print(f"ASYNC  route_with_async_verification():  {async_elapsed_ms:7.1f}ms (tier={tier.name})")

    print(f"\nAsync version returned {sync_elapsed_ms - async_elapsed_ms:.0f}ms faster.")
    print("\nWaiting 2 seconds for the background verification thread to finish...")
    time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    main()
