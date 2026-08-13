"""
Quick smoke test: run a couple of requests through the pipeline and
confirm they land in data/autopilot.db.

Run from the project root:
    python -m scripts.check_db_logging
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routing.router import load_default_router
from app.eval.pipeline import route_with_verification
from app.db.logger import DB_PATH


def main() -> None:
    router = load_default_router()

    route_with_verification("What is the capital of Japan?", router)
    route_with_verification(
        "Design a system architecture for a real-time chat app supporting "
        "1 million concurrent users, explaining key trade-offs.",
        router,
    )

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT id, tier, model_name, cost_usd, baseline_cost_usd, was_escalated "
        "FROM requests ORDER BY id DESC LIMIT 5"
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"Database: {DB_PATH}")
    print(f"{'id':>4} {'tier':>4} {'model_name':>15} {'cost_usd':>12} {'baseline_cost':>14} {'escalated':>10}")
    for row in rows:
        print(f"{row[0]:>4} {row[1]:>4} {row[2]:>15} {row[3]:>12.6f} {row[4]:>14.6f} {row[5]:>10}")


if __name__ == "__main__":
    main()
