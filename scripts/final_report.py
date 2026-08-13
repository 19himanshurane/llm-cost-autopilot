"""
Phase 6, Step 2: Final cost savings report.

Reads data/autopilot.db and prints the headline numbers for the case
study -- same math as the dashboard/API, plain text for easy copy-paste.

Run from the project root:
    python -m scripts.final_report
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.db.logger import DB_PATH


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM requests", conn)
    conn.close()

    total_cost = df["cost_usd"].sum()
    baseline_cost = df["baseline_cost_usd"].sum()
    savings = baseline_cost - total_cost
    savings_pct = savings / baseline_cost if baseline_cost else 0

    print(f"Total requests:        {len(df)}")
    print(f"Actual cost:           ${total_cost:.4f}")
    print(f"Baseline (all GPT-4o): ${baseline_cost:.4f}")
    print(f"Saved:                 ${savings:.4f} ({savings_pct:.1%})")
    print(f"Escalation rate:       {df['was_escalated'].mean():.1%}")
    print()
    print("Routing distribution:")
    dist = df["model_name"].value_counts()
    for model, count in dist.items():
        print(f"  {model:15s} {count:4d} ({count / len(df):.1%})")
    print()
    print("Requests by tier:")
    tier_labels = {1: "Tier 1 (Simple)", 2: "Tier 2 (Moderate)", 3: "Tier 3 (Complex)"}
    tier_dist = df["tier"].map(tier_labels).value_counts()
    for tier, count in tier_dist.items():
        print(f"  {tier:20s} {count:4d} ({count / len(df):.1%})")


if __name__ == "__main__":
    main()
