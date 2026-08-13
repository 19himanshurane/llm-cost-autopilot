"""
Phase 4 support: populate data/autopilot.db with a realistic batch of
requests, so Step 2's dashboard has actual volume to visualize instead of
2-3 test rows.

Resets the `requests` table first (DELETE, not dropping the file -- keeps
the schema, just clears old rows) so repeated runs give you a clean,
consistent dataset rather than piling up duplicates from earlier testing.

Runs every prompt from data/labeled_prompts.csv (211 prompts spanning all
three tiers) through the full route -> verify -> escalate -> log pipeline.
This is a small preview of Phase 6's full load test, just enough to make
Phase 4's dashboard meaningful.

Run from the project root:
    python -m scripts.populate_demo_data
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


def main() -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM requests")
    conn.close()
    print("Cleared existing rows from requests table.")

    with LABELED_CSV.open(encoding="utf-8") as f:
        prompts = [row["prompt"] for row in csv.DictReader(f)]

    router = load_default_router()

    start = time.monotonic()
    escalated_count = 0
    for i, prompt in enumerate(prompts, start=1):
        result = route_with_verification(prompt, router)
        if result.escalated:
            escalated_count += 1
        if i % 25 == 0 or i == len(prompts):
            print(f"  {i}/{len(prompts)} processed...")

    elapsed = time.monotonic() - start
    print(f"\nDone: {len(prompts)} requests logged to data/autopilot.db in {elapsed:.1f}s.")
    print(f"Escalations: {escalated_count} ({escalated_count / len(prompts):.1%})")


if __name__ == "__main__":
    main()
