"""
Phase 2, Step 2: Build a labeled dataset.

Generates 200+ example prompts labeled by complexity tier (see
app/routing/tiers.py for what each tier means) and writes them to
data/labeled_prompts.csv.

This uses templates + fill-in word pools instead of hand-typing 200+ unique
sentences. IMPORTANT: this is a *draft*. Open the CSV afterward and
spot-check a sample per tier -- if a label looks wrong to you based on the
tier definitions, fix it by hand. The classifier in Step 3 will only ever
be as good as this data.

Run from the project root:
    python -m scripts.generate_labeled_dataset
"""

from __future__ import annotations

import csv
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routing.tiers import ComplexityTier, TIER_DEFINITIONS

random.seed(42)  # fixed seed = same "random" output every time we run this

# ---------------------------------------------------------------------------
# Fill-in word pools. Kept as plain lists so they're easy to extend later --
# just add more entries and rerun the script for a bigger dataset.
# ---------------------------------------------------------------------------

COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "Italy", "Egypt",
             "Peru", "Norway", "Vietnam", "Chile", "Portugal", "Morocco",
             "Thailand", "Poland"]
ARITHMETIC_PAIRS = [(23, 19), (104, 57), (8, 76), (330, 12), (45, 45),
                     (999, 1), (17, 83), (256, 128), (7, 6), (500, 250)]
NAMES = ["Jane Doe", "Raj Patel", "Maria Garcia", "Tom Lee", "Amara Obi", "Wei Zhang"]
EMAILS = ["jane.doe@example.com", "raj.p@work.co", "m.garcia@mail.com",
          "tomlee@corp.io", "amara.obi@biz.net", "wei.zhang@firm.com"]
PHONES = ["555-0132", "555-0198", "555-0147", "555-0111", "555-0176", "555-0123"]
ITEMS_LISTS = [
    "apples, bananas, cherries, dates",
    "Python, JavaScript, Rust, Go",
    "Monday, Wednesday, Friday",
    "invoice #1021, invoice #1022, invoice #1023",
    "red, green, blue, yellow",
    "north, south, east, west",
]
WORDS_TRANSLATE = ["hello", "thank you", "water", "friend", "goodbye", "please", "yes", "no"]
TARGET_LANGS = ["Spanish", "French", "German", "Italian"]
EVENTS_YEARS = ["the first iPhone launch", "the Berlin Wall falling",
                "the first moon landing", "the founding of the UN",
                "the invention of the World Wide Web", "the fall of Rome"]
MONTHS = ["February", "April", "September", "November", "June", "December"]
SIMPLE_WORDS = ["cactus", "child", "goose", "index", "cactus", "mouse"]
QUICK_FACTS = [
    "What is the boiling point of water in Celsius?",
    "What is the chemical symbol for gold?",
    "What is the chemical symbol for oxygen?",
    "What is the freezing point of water in Fahrenheit?",
    "How many days are in a standard week?",
    "What are the three primary colors?",
]

PARAGRAPHS = [
    "The company reported record revenue this quarter, driven by strong demand "
    "in the enterprise segment, though margins tightened due to rising cloud "
    "infrastructure costs.",
    "Researchers found that the new material conducts electricity twice as "
    "efficiently as copper at room temperature, but degrades rapidly above "
    "60 degrees Celsius, limiting near-term commercial use.",
    "The city council approved the new transit line after two years of debate, "
    "citing projected ridership growth, though critics point to a 40% budget "
    "overrun compared to the original proposal.",
    "The startup pivoted from a consumer app to a B2B API product after "
    "discovering enterprise customers were willing to pay ten times more for "
    "the same underlying technology.",
    "A new study suggests that hybrid work arrangements improve reported "
    "employee satisfaction but slightly reduce cross-team collaboration on "
    "unplanned projects.",
    "The airline announced a new route between the two cities, citing rising "
    "business travel demand, but analysts question whether fuel costs make "
    "the route profitable.",
]
REVIEWS = [
    "The delivery was late but the product quality exceeded my expectations.",
    "Terrible customer service, waited three weeks for a response and no refund.",
    "Works exactly as described, arrived on time, would buy again.",
    "Packaging was damaged but the item inside was fine, mixed feelings overall.",
    "Absolutely love it, best purchase I've made this year.",
    "It's okay, does the job but nothing special for the price.",
]
EMAIL_TEXTS = [
    "CONGRATULATIONS!!! You've won a free cruise, click here to claim NOW!!!",
    "Hi team, attaching the Q3 report for review before Friday's meeting.",
    "Your account password was changed. If this wasn't you, contact support.",
    "URGENT: your account will be suspended, verify your details immediately!",
    "Reminder: your dentist appointment is tomorrow at 10am.",
    "You've been selected for a limited time offer, act fast to claim your prize!",
]
THING_PAIRS = [("REST", "GraphQL"), ("SQL databases", "NoSQL databases"),
               ("remote work", "in-office work"), ("Python", "Rust"),
               ("supervised learning", "unsupervised learning"),
               ("microservices", "monolithic architecture"),
               ("renting", "buying a home"), ("electric cars", "hybrid cars")]
DATASET_DESCS = [
    "User signup table with fields: email, signup_date, country, referral_source. "
    "30% of country values are null.",
    "Sales table with columns: order_id, amount, currency, region. Some amounts "
    "are negative with no explanation.",
    "Sensor readings table with timestamp, temperature, humidity. Timestamps are "
    "not in a consistent timezone.",
    "Customer table with name, email, signup_source. Some emails appear "
    "duplicated with slightly different casing.",
]
TICKETS = [
    "I was charged twice for my subscription this month, please help.",
    "The app crashes every time I try to upload a photo.",
    "I can't remember my password and the reset email never arrives.",
    "My order shows delivered but I never received the package.",
    "How do I change the email address on my account?",
    "The export feature is generating corrupted CSV files.",
]
HEADLINES = [
    "Local team wins championship after dramatic overtime finish",
    "Central bank raises interest rates for third straight quarter",
    "New startup unveils AI chip aimed at edge devices",
    "City unveils plan for new public park downtown",
    "Streaming service announces record subscriber growth",
    "Researchers publish breakthrough in battery storage technology",
]
CODE_COMMENTS = [
    "This function doesn't handle the case where the list is empty.",
    "Consider renaming this variable for clarity, but not blocking.",
    "This introduces a SQL injection vulnerability, must fix before merge.",
    "Minor: extra blank line here, feel free to ignore.",
]

STORY_PREMISES = [
    "an AI that discovers it is being used to route its own kind to cheaper hardware",
    "a lighthouse keeper who receives messages from a ship that sank decades ago",
    "a chess player who realizes their opponent is playing a completely different game",
    "a translator who starts hearing a language no one else can understand",
    "a city where everyone forgets one memory each year on their birthday",
    "a gardener who discovers their plants grow faster when they tell them the truth",
]
PROJECT_GOALS = [
    "migrating a monolithic Django app to microservices with minimal downtime",
    "moving a company's on-prem data warehouse to the cloud without a service outage",
    "rolling out a new pricing model without alienating existing customers",
    "launching a mobile app in three new countries within one quarter",
    "consolidating five legacy internal tools into one platform",
    "introducing a four-day work week without reducing team output",
]
COMPLAINTS = [
    "My subscription charged me twice this month but support says it's a known "
    "issue with no ETA.",
    "I was promised a callback within 24 hours and it's been a week with no response.",
    "I've been a customer for five years and this is the third billing error this year.",
    "The feature I rely on most was removed with no warning or migration path.",
]
POSITIONS = [
    "requiring a four-day work week for all knowledge workers",
    "banning AI-generated content from being labeled without disclosure",
    "making all public transit free within major cities",
    "requiring social media platforms to verify user identities",
]
SYSTEM_GOALS = [
    "a real-time chat app supporting 1 million concurrent users",
    "a recommendation engine for a mid-size e-commerce site",
    "a fraud detection pipeline processing 10,000 transactions per second",
    "a document search system for a company with 50 years of internal records",
]
LOGIC_PUZZLES = [
    "Three friends split a bill unevenly based on what they ordered, but one paid "
    "with a card that had a 2% surcharge -- how much does each actually owe?",
    "A train leaves city A at 60mph, another leaves city B (300 miles away) at "
    "40mph two hours later heading toward A -- when do they meet?",
    "A warehouse has three shelves with different restock schedules (every 3, 4, "
    "and 6 days) -- on what day do all three restock on the same day again?",
    "Five people need to cross a bridge at night with one flashlight, max two at "
    "a time, each walks at a different speed -- what's the fastest crossing plan?",
]
STARTUP_DILEMMAS = [
    "running out of runway in four months with two competing feature bets",
    "a key engineer threatening to leave right before a major launch",
    "a large customer asking for a custom feature that conflicts with the roadmap",
    "a competitor copying their core feature and undercutting on price",
]
NEGOTIATION_SCENARIOS = [
    "two engineering teams disagree on which service owns a shared database",
    "a client wants a scope increase but refuses to adjust the budget or timeline",
    "two co-founders disagree on whether to raise venture funding or stay bootstrapped",
    "a manager and a direct report disagree about the report's promotion readiness",
]
CREATIVE_EXPLAIN_TOPICS = [
    "quantum computing", "how the immune system works", "how interest rates "
    "affect the housing market", "how neural networks learn"
]
POEM_THEMES = ["impermanence", "a city at 3am", "the first snow of winter",
               "an old library", "a river that remembers", "a lighthouse"]

# ---------------------------------------------------------------------------
# Tier 1 -- Simple: reformatting, extraction, basic Q&A from provided context
# ---------------------------------------------------------------------------

def build_tier_1() -> list[str]:
    prompts = []
    for c in COUNTRIES:
        prompts.append(f"What is the capital of {c}?")
    for a, b in ARITHMETIC_PAIRS:
        prompts.append(f"What is {a} + {b}?")
    for name, email in zip(NAMES, EMAILS):
        prompts.append(
            f"Extract the email address from this text: 'Contact {name} at "
            f"{email} for details.'"
        )
    for name, phone in zip(NAMES, PHONES):
        prompts.append(
            f"Extract the phone number from this text: 'Call {name} at "
            f"{phone} for support.'"
        )
    for items in ITEMS_LISTS:
        prompts.append(f"Reformat this list into bullet points: {items}")
        prompts.append(f"Convert this comma-separated list into a numbered list: {items}")
    for word, lang in itertools.islice(itertools.product(WORDS_TRANSLATE, TARGET_LANGS), 20):
        prompts.append(f"Translate '{word}' from English to {lang}.")
    for event in EVENTS_YEARS:
        prompts.append(f"What year did {event} happen?")
    for month in MONTHS:
        prompts.append(f"How many days are in {month}?")
    for word in SIMPLE_WORDS:
        prompts.append(f"What is the plural of '{word}'?")
    prompts.extend(QUICK_FACTS)
    for name, phone in zip(NAMES, PHONES):
        prompts.append(
            f"Given this context: '{name} scheduled a call for 3pm on Tuesday, "
            f"reachable at {phone}.' — what time was the call scheduled?"
        )
    return prompts


# ---------------------------------------------------------------------------
# Tier 2 -- Moderate: summarization, classification, structured analysis
# ---------------------------------------------------------------------------

def build_tier_2() -> list[str]:
    prompts = []
    for p in PARAGRAPHS:
        prompts.append(f"Summarize the following in two sentences: {p}")
        prompts.append(f"Identify the main theme of this paragraph: {p}")
    for r in REVIEWS:
        prompts.append(
            f"Classify the sentiment of this review as positive, negative, "
            f"or neutral: '{r}'"
        )
    for e in EMAIL_TEXTS:
        prompts.append(f"Classify this email as spam or not spam: '{e}'")
    for a, b in THING_PAIRS:
        prompts.append(f"Compare and contrast {a} and {b} in a short structured overview.")
        prompts.append(f"Summarize the key differences between {a} and {b}.")
    for d in DATASET_DESCS:
        prompts.append(
            f"Analyze this dataset description and list three potential data "
            f"quality issues: '{d}'"
        )
    for t in TICKETS:
        prompts.append(
            f"Categorize this support ticket into one of: billing, technical, "
            f"account, other: '{t}'"
        )
    for h in HEADLINES:
        prompts.append(
            f"Classify this news headline by topic (politics, sports, tech, "
            f"business, entertainment): '{h}'"
        )
    for c in CODE_COMMENTS:
        prompts.append(f"Classify this code review comment as blocking or non-blocking: '{c}'")
    prompts.append(
        "Structure this unorganized meeting note into an agenda with action "
        "items: 'talked about q3 budget, need someone to follow up with "
        "finance, also discussed new hire start date, marketing wants more "
        "budget for ads'"
    )
    prompts.append(
        "Given these three product reviews, identify the most common "
        "complaint: [1] 'shipping took forever' [2] 'box arrived late but "
        "product is great' [3] 'ordered a week ago still nothing']"
    )
    prompts.append(
        "Structure this rough project update into a status report with "
        "sections for Done, In Progress, and Blocked: 'finished the login "
        "page, still working on payments, waiting on design for the "
        "dashboard, api docs are done'"
    )
    return prompts


# ---------------------------------------------------------------------------
# Tier 3 -- Complex: multi-step reasoning, creative generation, nuanced judgment
# ---------------------------------------------------------------------------

def build_tier_3() -> list[str]:
    prompts = []
    for premise in STORY_PREMISES:
        prompts.append(f"Write a short story (150 words) about {premise}.")
    for goal in PROJECT_GOALS:
        prompts.append(
            f"Design a step-by-step plan for {goal}, including risk "
            f"mitigation for each step."
        )
    for c in COMPLAINTS:
        prompts.append(
            f"A user says: '{c}' Draft a nuanced, empathetic response that "
            f"also sets realistic expectations."
        )
    for pos in POSITIONS:
        prompts.append(
            f"Write a persuasive argument for {pos}, then present the "
            f"strongest counterargument."
        )
    for sg in SYSTEM_GOALS:
        prompts.append(f"Design a system architecture for {sg}, explaining key trade-offs.")
    for puzzle in LOGIC_PUZZLES:
        prompts.append(f"Walk through the multi-step reasoning to solve: {puzzle}")
    for dilemma in STARTUP_DILEMMAS:
        prompts.append(
            f"You are advising a startup facing {dilemma}. What should they "
            f"do and why?"
        )
    for scenario in NEGOTIATION_SCENARIOS:
        prompts.append(
            f"Mediate this disagreement where {scenario}. Propose a fair "
            f"resolution and explain your reasoning."
        )
    for topic in CREATIVE_EXPLAIN_TOPICS:
        prompts.append(f"Explain {topic} using a single extended metaphor a 10-year-old would understand.")
    for theme in POEM_THEMES:
        prompts.append(f"Write a short poem about {theme}.")
    prompts.append(
        "Compare three possible strategies for entering a new international "
        "market, weigh the trade-offs of each, and recommend one with "
        "justification."
    )
    prompts.append(
        "Draft a difficult conversation script for telling a long-tenured "
        "employee their role is being eliminated, balancing honesty and empathy."
    )
    prompts.append(
        "Design an experiment to test whether a new onboarding flow actually "
        "improves 30-day user retention, including how you'd rule out "
        "confounding factors."
    )
    return prompts


def main() -> None:
    rows: list[tuple[str, int, str]] = []
    for tier, builder in [
        (ComplexityTier.TIER_1_SIMPLE, build_tier_1),
        (ComplexityTier.TIER_2_MODERATE, build_tier_2),
        (ComplexityTier.TIER_3_COMPLEX, build_tier_3),
    ]:
        label = TIER_DEFINITIONS[tier].label
        for prompt in builder():
            rows.append((prompt, int(tier), label))

    random.shuffle(rows)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "labeled_prompts.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "tier", "tier_label"])
        writer.writerows(rows)

    counts = {}
    for _, tier, label in rows:
        counts[label] = counts.get(label, 0) + 1

    print(f"Wrote {len(rows)} labeled prompts to {out_path}")
    for label, count in counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
