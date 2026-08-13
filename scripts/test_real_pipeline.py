from app.routing.router import load_default_router
from app.client import send_request
from app.db.logger import log_request
from app.models import get_model

router = load_default_router()
baseline_model = get_model("gpt-4o")

prompts = [
    "What is the capital of France?",
    "List the days of the week.",
    "Fix the typo in this sentence: I are going to the store.",
    "Summarize this paragraph in one sentence: The quarterly report showed strong growth across all divisions.",
    "Classify this review as positive or negative: Terrible service, would not recommend.",
    "Extract the email address from this text: Contact us at support@example.com for help.",
]

for p in prompts:
    tier, model = router.route(p)
    response = send_request(p, model)
    row_id = log_request(p, tier, response, baseline_model, was_escalated=False, quality_score=None)
    print(f"[id={row_id}] Tier {tier} -> {model.name} | mocked={response.was_mocked} | cost=${response.cost_usd:.6f} | latency={response.latency_ms}ms")