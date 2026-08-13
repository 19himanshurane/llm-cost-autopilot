"""
Complexity tiers for LLM Cost Autopilot's routing classifier.

These three tiers are the entire vocabulary the rest of Phase 2 speaks:
- the labeled dataset (Step 2) uses these tier values as its labels,
- the classifier (Step 3) predicts one of these three values,
- the routing config (Step 4) maps each tier to a specific model.

Getting this definition right matters more than any of the ML code that
comes later -- the classifier can only ever be as good as these category
boundaries and the examples we give it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ComplexityTier(IntEnum):
    """
    Ordered on purpose: 1 = cheapest/simplest, 3 = most expensive/most
    complex. Using IntEnum (instead of a plain string enum) lets us do
    things like "tier >= ComplexityTier.TIER_2_MODERATE" later if we ever
    need threshold logic, since the ordering carries real meaning here.
    """

    TIER_1_SIMPLE = 1
    TIER_2_MODERATE = 2
    TIER_3_COMPLEX = 3


@dataclass(frozen=True)
class TierDefinition:
    """Human-readable documentation for one tier, used by the labeling
    guide and later by the dataset-generation script."""

    tier: ComplexityTier
    label: str
    description: str
    example_tasks: list[str]
    example_prompts: list[str]


TIER_DEFINITIONS: dict[ComplexityTier, TierDefinition] = {
    ComplexityTier.TIER_1_SIMPLE: TierDefinition(
        tier=ComplexityTier.TIER_1_SIMPLE,
        label="Simple",
        description=(
            "Reformatting, extraction, or basic Q&A from context that's "
            "already provided in the prompt. Little to no reasoning "
            "required -- the answer is largely 'in' the prompt already, "
            "the model just has to locate or reshape it."
        ),
        example_tasks=[
            "reformatting",
            "extraction",
            "basic Q&A from provided context",
        ],
        example_prompts=[
            "Extract the email address from this text: 'Contact Jane at "
            "jane.doe@example.com for details.'",
            "Reformat this list into bullet points: apples, bananas, cherries",
            "What is the capital of France?",
        ],
    ),
    ComplexityTier.TIER_2_MODERATE: TierDefinition(
        tier=ComplexityTier.TIER_2_MODERATE,
        label="Moderate",
        description=(
            "Summarization, classification, or structured analysis. "
            "Requires synthesizing multiple pieces of information, but "
            "follows a fairly predictable, well-defined pattern -- there's "
            "usually one clearly 'best' answer."
        ),
        example_tasks=["summarization", "classification", "structured analysis"],
        example_prompts=[
            "Summarize the following in two sentences: The company reported "
            "record revenue this quarter...",
            "Classify the sentiment of this review as positive, negative, "
            "or neutral: 'The delivery was late but quality was great.'",
            "Compare and contrast REST and GraphQL APIs in a short "
            "structured overview.",
        ],
    ),
    ComplexityTier.TIER_3_COMPLEX: TierDefinition(
        tier=ComplexityTier.TIER_3_COMPLEX,
        label="Complex",
        description=(
            "Multi-step reasoning, creative generation, or nuanced "
            "judgment calls. The model has to plan ahead, weigh "
            "trade-offs, or produce genuinely novel content -- there "
            "usually isn't one single 'correct' answer, and getting it "
            "wrong is costly, so we spend more on quality here."
        ),
        example_tasks=[
            "multi-step reasoning",
            "creative generation",
            "nuanced judgment calls",
        ],
        example_prompts=[
            "Design a step-by-step plan for migrating a monolithic Django "
            "app to microservices with minimal downtime.",
            "Write a short story (150 words) about an AI that discovers "
            "it is being used to route its own kind to cheaper hardware.",
            "A user says their subscription charged twice with no ETA on a "
            "fix. Draft a nuanced, empathetic response.",
        ],
    ),
}


def describe(tier: ComplexityTier) -> str:
    """Convenience lookup: given a tier, return its human-readable
    description. Used later when we print labeling instructions."""
    return TIER_DEFINITIONS[tier].description
