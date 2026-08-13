"""
Phase 5: FastAPI service.

Step 1: POST /v1/completions -- the main endpoint end users call.
Step 2: GET /v1/models, GET /v1/stats, PUT /v1/routing-config -- operational
        endpoints for inspecting and reconfiguring the system without a
        redeploy.

Run from the project root:
    uvicorn app.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db.logger import DB_PATH
from app.eval.async_pipeline import route_with_async_verification
from app.models import MODEL_REGISTRY
from app.routing.router import DEFAULT_CONFIG_PATH, load_default_router
from app.routing.tiers import TIER_DEFINITIONS

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes each request to the cheapest model that can handle it.",
    version="0.1.0",
)

# Loaded ONCE when the server starts, reused for every request.
router = load_default_router()


# ---------------------------------------------------------------------------
# Step 1: POST /v1/completions
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send.")


class CompletionResponse(BaseModel):
    text: str
    selected_model: str
    provider: str
    tier: int
    tier_label: str
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    was_mocked: bool


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest) -> CompletionResponse:
    """The one endpoint end users actually call. They send a prompt, we
    decide the model, they get back the answer plus an explanation."""
    tier, response = route_with_async_verification(request.prompt, router)
    tier_def = TIER_DEFINITIONS[tier]

    reason = (
        f"Classified as {tier_def.label} (Tier {int(tier)}): {tier_def.description} "
        f"Routed to {response.model_name} to avoid overpaying for a more "
        f"capable model than this request needs."
    )

    return CompletionResponse(
        text=response.text,
        selected_model=response.model_name,
        provider=response.provider,
        tier=int(tier),
        tier_label=tier_def.label,
        reason=reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        was_mocked=response.was_mocked,
    )


# ---------------------------------------------------------------------------
# Step 2: operational endpoints
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    provider: str
    quality_tier: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: int


@app.get("/v1/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """Every model in the registry and what it costs -- lets a caller (or
    a curious human) see the full menu without reading source code."""
    return [
        ModelInfo(
            name=m.name,
            provider=m.provider.value,
            quality_tier=m.quality_tier.value,
            cost_per_input_token=m.cost_per_input_token,
            cost_per_output_token=m.cost_per_output_token,
            avg_latency_ms=m.avg_latency_ms,
        )
        for m in MODEL_REGISTRY.values()
    ]


class StatsResponse(BaseModel):
    total_requests: int
    total_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_pct: float
    escalation_rate: float
    routing_distribution: dict[str, int]


@app.get("/v1/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """The same numbers the Streamlit dashboard shows, as JSON -- so
    another program (a CI check, a Slack bot, a status page) could pull
    them without scraping a dashboard meant for humans."""
    if not Path(DB_PATH).exists():
        return StatsResponse(
            total_requests=0, total_cost_usd=0, baseline_cost_usd=0,
            savings_usd=0, savings_pct=0, escalation_rate=0, routing_distribution={},
        )

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM requests", conn)
    conn.close()

    if df.empty:
        return StatsResponse(
            total_requests=0, total_cost_usd=0, baseline_cost_usd=0,
            savings_usd=0, savings_pct=0, escalation_rate=0, routing_distribution={},
        )

    total_cost = float(df["cost_usd"].sum())
    baseline_cost = float(df["baseline_cost_usd"].sum())
    savings = baseline_cost - total_cost

    return StatsResponse(
        total_requests=len(df),
        total_cost_usd=round(total_cost, 6),
        baseline_cost_usd=round(baseline_cost, 6),
        savings_usd=round(savings, 6),
        savings_pct=round(savings / baseline_cost, 4) if baseline_cost else 0.0,
        escalation_rate=round(float(df["was_escalated"].mean()), 4),
        routing_distribution=df["model_name"].value_counts().to_dict(),
    )


class RoutingConfigUpdate(BaseModel):
    tier_1_simple: str | None = None
    tier_2_moderate: str | None = None
    tier_3_complex: str | None = None


@app.put("/v1/routing-config")
def update_routing_config(update: RoutingConfigUpdate) -> dict:
    """Change which model handles which tier WITHOUT redeploying any code
    -- exactly what the original spec asked for. Only the fields you pass
    get changed; anything left as null keeps its current value. Writes
    back to routing_config.yaml so the change survives a server restart."""
    changes = {k: v for k, v in update.model_dump().items() if v is not None}
    if not changes:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    for tier_key, model_name in changes.items():
        if model_name not in MODEL_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Known models: {list(MODEL_REGISTRY)}",
            )
        router.config["routing"][tier_key] = model_name

    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(router.config, f, default_flow_style=False)

    return {"routing": router.config["routing"]}
