"""
Phase 4, Step 1: request logging.

Every request that goes through the router gets one row in a database.
Unlike escalation_log.jsonl (append-only text, fine for a simple audit
trail), a real database lets Step 2's dashboard efficiently ask aggregate
questions ("total cost this week", "escalation rate by tier") without
reading and parsing an entire file by hand.

Two backends, chosen automatically by whether DATABASE_URL is set:

- No DATABASE_URL (local dev, tests): a local SQLite file at
  data/autopilot.db. Zero setup, exactly how this started.
- DATABASE_URL set (production / Render): a real Postgres database (e.g.
  a free Neon project). This exists because Render's free web services
  have no persistent disk -- the local SQLite file gets wiped every time
  the container restarts (which happens automatically after ~15 minutes
  of inactivity), so every deployed dashboard kept coming up empty.
  Postgres storage survives restarts, so logged data actually persists.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "autopilot.db"
DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA_SQLITE = """
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

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS requests (
    id SERIAL PRIMARY KEY,
    timestamp DOUBLE PRECISION NOT NULL,
    prompt_hash TEXT NOT NULL,
    tier INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd DOUBLE PRECISION NOT NULL,
    latency_ms INTEGER NOT NULL,
    was_escalated INTEGER NOT NULL DEFAULT 0,
    quality_score DOUBLE PRECISION,
    baseline_model_name TEXT NOT NULL,
    baseline_cost_usd DOUBLE PRECISION NOT NULL
);
"""


def using_postgres() -> bool:
    return bool(DATABASE_URL)


def get_connection():
    """Returns a DB-API connection, schema already ensured to exist.
    Callers that need to stay backend-agnostic (reading with pandas, for
    example) can just use this and never check which backend is active."""
    if using_postgres():
        import psycopg

        conn = psycopg.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(SCHEMA_POSTGRES)
        conn.commit()
        return conn

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA_SQLITE)
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

    values = (
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
    )

    conn = get_connection()

    if using_postgres():
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO requests
                   (timestamp, prompt_hash, tier, model_name, provider,
                    input_tokens, output_tokens, cost_usd, latency_ms,
                    was_escalated, quality_score, baseline_model_name, baseline_cost_usd)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                values,
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        return row_id

    with conn:
        cursor = conn.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, tier, model_name, provider,
                input_tokens, output_tokens, cost_usd, latency_ms,
                was_escalated, quality_score, baseline_model_name, baseline_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def update_escalation(row_id: int, was_escalated: bool, quality_score: float) -> None:
    """Called by the async pipeline once background verification finishes,
    to fill in fields that weren't known yet when the row was first logged."""
    conn = get_connection()

    if using_postgres():
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE requests SET was_escalated = %s, quality_score = %s WHERE id = %s",
                (int(was_escalated), quality_score, row_id),
            )
        conn.commit()
        conn.close()
        return

    with conn:
        conn.execute(
            "UPDATE requests SET was_escalated = ?, quality_score = ? WHERE id = ?",
            (int(was_escalated), quality_score, row_id),
        )
    conn.close()
