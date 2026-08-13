"""
Phase 3, Step 4: Feed failures back into the classifier.

Reads every escalation event from data/escalation_log.jsonl (each one
represents a prompt where the classifier's tier choice turned out to be
wrong -- the cheap model's answer diverged too much from the reference
model). For each one:
  - if the prompt isn't in the labeled dataset yet, add it with the
    CORRECTED tier (whatever tier the escalated-to model belongs to).
  - if the prompt IS already in the dataset but labeled too low (e.g. it
    was Tier 1 but actually needed Tier 3), correct its label in place.
Then retrains the classifier on the corrected/enlarged dataset.

In production, this script would run on a schedule (the spec suggests
weekly) via a job scheduler -- that's a deployment/ops concern layered on
top of this logic, not something this script needs to handle itself.

Run from the project root:
    python -m scripts.retrain_from_failures
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routing.tiers import ComplexityTier, TIER_DEFINITIONS
from app.routing.router import Router

ESCALATION_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "escalation_log.jsonl"
LABELED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled_prompts.csv"


def model_name_to_tier(model_name: str, router: Router) -> ComplexityTier:
    """Reverse-lookup: given a model name, which tier does routing_config.yaml
    say it belongs to? Assumes each model is only used for one tier, true
    for our current config."""
    for tier_key, mapped_model in router.config["routing"].items():
        if mapped_model == model_name:
            return ComplexityTier(int(tier_key.split("_")[1]))
    raise ValueError(f"No tier in routing_config.yaml maps to model '{model_name}'")


def main() -> None:
    if not ESCALATION_LOG_PATH.exists():
        print(f"No escalation log found at {ESCALATION_LOG_PATH} -- nothing to learn from yet.")
        return

    router = Router()

    # Load the existing dataset as an ordered list of rows so we can both
    # append new prompts AND correct existing ones in place.
    with LABELED_DATA_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prompt_to_index = {row["prompt"]: i for i, row in enumerate(rows)}

    added, corrected = 0, 0
    with ESCALATION_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            prompt = event["prompt"]
            corrected_tier = model_name_to_tier(event["escalated_model"], router)
            label = TIER_DEFINITIONS[corrected_tier].label

            if prompt in prompt_to_index:
                idx = prompt_to_index[prompt]
                existing_tier = int(rows[idx]["tier"])
                if existing_tier < int(corrected_tier):
                    print(f"  Correcting existing label: '{prompt[:60]}...' "
                          f"{rows[idx]['tier_label']} -> {label}")
                    rows[idx]["tier"] = str(int(corrected_tier))
                    rows[idx]["tier_label"] = label
                    corrected += 1
            else:
                rows.append({"prompt": prompt, "tier": str(int(corrected_tier)), "tier_label": label})
                prompt_to_index[prompt] = len(rows) - 1
                added += 1

    if added == 0 and corrected == 0:
        print("No new failures to learn from -- dataset already reflects everything logged.")
        return

    with LABELED_DATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "tier", "tier_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset updated: {added} new example(s) added, {corrected} existing label(s) corrected.")
    print("Retraining classifier on the corrected dataset...\n")

    import subprocess
    subprocess.run([sys.executable, "-m", "scripts.train_classifier"], check=True)


if __name__ == "__main__":
    main()
