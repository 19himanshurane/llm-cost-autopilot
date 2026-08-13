"""
Phase 6, Step 1: Realistic load test.

Runs 500+ prompts through the full pipeline (route -> answer -> verify ->
escalate -> log) and reports throughput + final aggregate stats -- this is
the dataset the final case study numbers are drawn from.

Volume approach, stated honestly: takes the 211 hand-templated prompts
from Phase 2's labeled_prompts.csv and generates 3 variations of each
(as-is, with an appended "please be concise" instruction, and with
prepended work context) to reach 633 total requests. This is a legitimate
volume/throughput-testing technique -- it genuinely exercises the full
pipeline at higher load -- but it is NOT a claim that all 633 are
independently unique scenarios. Worth being precise about in the case
study, not just in this comment.

Run from the project root:
    python -m scripts.load_test
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.logger import get_connection
from app.eval.pipeline import route_with_verification
from app.routing.router import load_default_router

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABELED_CSV = DATA_DIR / "labeled_prompts.csv"

VARIANTS = [
    lambda p: p,
    lambda p: f"{p}\n\nPlease be concise.",
    lambda p: f"For context, this is for a work project. {p}",
]


def build_load_test_prompts() -> list[str]:
    with LABELED_CSV.open(encoding="utf-8") as f:
        base_prompts = [row["prompt"] for row in csv.DictReader(f)]
    prompts = []
    for variant_fn in VARIANTS:
        prompts.extend(variant_fn(p) for p in base_prompts)
    return prompts


def main() -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM requests")
    conn.close()
    print("Cleared existing rows from requests table for a clean load test run.")

    prompts = build_load_test_prompts()
    print(f"Prepared {len(prompts)} prompts for the load test.")

    router = load_default_router()

    start = time.monotonic()
    escalated_count = 0
    for i, prompt in enumerate(prompts, start=1):
        result = route_with_verification(prompt, router)
        if result.escalated:
            escalated_count += 1
        if i % 50 == 0 or i == len(prompts):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  {i}/{len(prompts)} processed ({rate:.1f} req/s)...")

    elapsed = time.monotonic() - start
    print(f"\nDone: {len(prompts)} requests in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} req/s)")
    print(f"Escalations: {escalated_count} ({escalated_count / len(prompts):.1%})")


if __name__ == "__main__":
    main()
