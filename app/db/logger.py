"""
Phase 4, Step 1: SQLite logging.

Every request that goes through the router gets one row in a local SQLite
database -- a single-file database (data/autopilot.db) that needs no
server to run. Unlike escalation_log.jsonl (append-only text, fine for a
simple audit trail), a real database lets Step 2's dashboard efficiently
ask aggregate questions ("total cost this week", "escalation rate by
tier") without reading and parsing an entire file by hand.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "autopilot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    tier INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    was_escalated INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    baseline_model_name TEXT NOT NULL,
    baseline_cost_usd REAL NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def hash_prompt(prompt: str) -> str:
    """Store a hash instead of raw prompt text by default -- a reasonable
    habit for a system that might see real user data later, even though
    for this project everything is synthetic anyway."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def log_request(
    prompt: str,
    tier,
    response,
    baseline_model,
    was_escalated: bool = False,
    quality_score: float | None = None,
) -> int:
    """
    Insert one row, return its id.

    `baseline_model` is the model this request WOULD have used if every
    single request went to the most expensive model, no routing at all --
    that's exactly what Phase 4 Step 2's "you saved $X compared to sending
    everything to gpt-4o" headline number needs to compute against.
    """
    baseline_cost = baseline_model.estimate_cost(response.input_tokens, response.output_tokens)

    conn = get_connection()
    with conn:
        cursor = conn.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, tier, model_name, provider,
                input_tokens, output_tokens, cost_usd, latency_ms,
                was_escalated, quality_score, baseline_model_name, baseline_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                hash_prompt(prompt),
                int(tier),
                response.model_name,
                response.provider,
                response.input_tokens,
                response.output_tokens,
                response.cost_usd,
                response.latency_ms,
                int(was_escalated),
                quality_score,
                baseline_model.name,
                baseline_cost,
            ),
        )
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def update_escalation(row_id: int, was_escalated: bool, quality_score: float) -> None:
    """Called by the async pipeline once background verification finishes,
    to fill in fields that weren't known yet when the row was first logged."""
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE requests SET was_escalated = ?, quality_score = ? WHERE id = ?",
            (int(was_escalated), quality_score, row_id),
        )
    conn.close()
