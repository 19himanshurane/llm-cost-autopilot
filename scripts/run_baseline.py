"""
Phase 1, Step 3: Test every provider.

Sends the same 10 prompts (spanning simple/moderate/complex on purpose —
these double as a preview of the Phase 2 complexity tiers) to every model in
the registry, logs outputs/costs/latencies, and prints a summary table.

Run from the project root:
    python -m scripts.run_baseline

Output:
    data/baseline_results.json  (every raw response)
    data/baseline_summary.csv   (per-model aggregate cost/latency)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.client import send_request
from app.models import MODEL_REGISTRY

BASELINE_PROMPTS = [
    # Tier 1 -- simple
    "Extract the email address from this text: 'Contact Jane at jane.doe@example.com for details.'",
    "Reformat this list into bullet points: apples, bananas, cherries, dates",
    "What is the capital of France?",
    # Tier 2 -- moderate
    "Summarize the following in two sentences: The company reported record "
    "revenue this quarter, driven by strong demand in the enterprise segment, "
    "though margins tightened due to rising cloud infrastructure costs.",
    "Classify the sentiment of this review as positive, negative, or neutral: "
    "'The delivery was late but the product quality exceeded my expectations.'",
    "Compare and contrast REST and GraphQL APIs in a short structured overview.",
    "Analyze this dataset description and list three potential data quality issues: "
    "'User signup table with fields: email, signup_date, country, referral_source. "
    "30% of country values are null.'",
    # Tier 3 -- complex
    "Write a short story (150 words) about an AI that discovers it is being "
    "used to route its own kind to cheaper hardware.",
    "A user says: 'My subscription charged me twice this month but support says "
    "it's a known issue with no ETA.' Draft a nuanced, empathetic response that "
    "also sets realistic expectations.",
    "Design a step-by-step plan for migrating a monolithic Django app to "
    "microservices with minimal downtime, including risk mitigation for each step.",
]


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)

    all_results = []

    for model_name, model_config in MODEL_REGISTRY.items():
        print(f"\n=== {model_name} ({model_config.provider.value}) ===")
        for i, prompt in enumerate(BASELINE_PROMPTS, start=1):
            response = send_request(prompt, model_config)
            mock_tag = " [MOCK]" if response.was_mocked else ""
            print(
                f"  [{i:02d}] {response.latency_ms:5d}ms  "
                f"${response.cost_usd:.6f}  "
                f"{response.input_tokens}in/{response.output_tokens}out{mock_tag}"
            )
            all_results.append(
                {
                    "prompt_index": i,
                    "prompt": prompt,
                    **response.as_dict(),
                }
            )

    results_path = data_dir / "baseline_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {len(all_results)} raw results to {results_path}")

    df = pd.DataFrame(all_results)
    summary = (
        df.groupby("model_name")
        .agg(
            provider=("provider", "first"),
            avg_cost_usd=("cost_usd", "mean"),
            total_cost_usd=("cost_usd", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            was_mocked=("was_mocked", "any"),
        )
        .reset_index()
        .sort_values("total_cost_usd")
    )
    summary_path = data_dir / "baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote per-model summary to {summary_path}\n")
    print(summary.to_string(index=False))

    if summary["was_mocked"].any():
        print(
            "\nNOTE: One or more models ran in MOCK mode (no API key / local "
            "server found). Costs/latencies for those rows are simulated, "
            "not real. Add keys to .env and rerun for real numbers."
        )


if __name__ == "__main__":
    main()
