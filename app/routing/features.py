"""
Phase 2, Step 2 (features) / prep for Step 3 (classifier training).

Converts a raw prompt (a string) into a fixed-length list of numbers --
"features" -- that a classical ML model can actually learn from. Classical
models like LogisticRegression or RandomForestClassifier don't read text;
they only take numeric input, so this file is the bridge between "a prompt
a human wrote" and "something scikit-learn can train on."

Each function below extracts ONE signal. extract_features() combines them
all into a dict, and vectorize() turns that dict into a plain ordered list
(a "feature vector") in a consistent order -- that consistent order matters
a lot, since the classifier has to see features in the same order every
single time, for training AND for predicting later.
"""

from __future__ import annotations

import re

# Words that tend to signal the request needs real reasoning/judgment,
# not just lookup or reformatting -- strong Tier 2/3 signal.
ANALYSIS_VERBS = [
    "analyze", "compare", "contrast", "evaluate", "assess", "recommend",
    "critique", "argue", "weigh", "trade-off", "trade off", "design",
    "strategy", "strategize", "justify", "mediate", "advise",
]

# Words that signal open-ended creative or judgment-heavy generation --
# Tier 3 signal specifically (there's no single "correct" answer).
CREATIVE_JUDGMENT_VERBS = [
    "write a story", "write a poem", "draft", "compose", "empathetic",
    "nuanced", "persuasive", "counterargument", "plan for", "step-by-step",
    "step by step",
]

# Phrases that signal the requested OUTPUT has real structure, which
# usually means the task itself is more involved than a one-line answer.
FORMAT_KEYWORDS = [
    "bullet points", "numbered list", "json", "structured", "step-by-step",
    "step by step", "table", "agenda", "report", "sections", "outline",
]

# Words/punctuation that tend to signal the prompt is layering on multiple
# constraints (do X, but also Y, within Z) -- more constraints usually
# means more for the model to juggle at once.
CONSTRAINT_MARKERS = ["but", "also", "must", "should", "at least", "no more than",
                       "within", "while", "without", "including"]


def _word_count(prompt: str) -> int:
    return len(prompt.split())


def _approx_token_count(prompt: str) -> int:
    # Same rough heuristic used in the mock provider: ~1.3 tokens/word.
    return int(_word_count(prompt) * 1.3)


def _count_keyword_hits(prompt: str, keywords: list[str]) -> int:
    lowered = prompt.lower()
    return sum(1 for kw in keywords if kw in lowered)


def _has_provided_context(prompt: str) -> bool:
    # Heuristic: prompts that hand the model a chunk of context to work
    # from tend to quote it, e.g. "...: 'some text here'".
    return bool(re.search(r"['\"].{10,}['\"]", prompt))


def _constraint_count(prompt: str) -> int:
    lowered = prompt.lower()
    marker_hits = sum(lowered.count(m) for m in CONSTRAINT_MARKERS)
    comma_hits = prompt.count(",")
    return marker_hits + comma_hits


def extract_features(prompt: str) -> dict[str, float]:
    """The main entry point: prompt string in, feature dict out."""
    return {
        "word_count": _word_count(prompt),
        "approx_token_count": _approx_token_count(prompt),
        "char_count": len(prompt),
        "analysis_verb_count": _count_keyword_hits(prompt, ANALYSIS_VERBS),
        "creative_judgment_count": _count_keyword_hits(prompt, CREATIVE_JUDGMENT_VERBS),
        "format_complexity_count": _count_keyword_hits(prompt, FORMAT_KEYWORDS),
        "constraint_count": _constraint_count(prompt),
        "has_context_provided": 1.0 if _has_provided_context(prompt) else 0.0,
        "question_mark_count": prompt.count("?"),
    }


# The fixed order features get turned into a vector -- MUST stay consistent
# between training and prediction, which is why we define it once, here.
FEATURE_NAMES: tuple[str, ...] = (
    "word_count",
    "approx_token_count",
    "char_count",
    "analysis_verb_count",
    "creative_judgment_count",
    "format_complexity_count",
    "constraint_count",
    "has_context_provided",
    "question_mark_count",
)


def vectorize(features: dict[str, float]) -> list[float]:
    """Turn a feature dict into a plain ordered list, using FEATURE_NAMES
    to guarantee the order is always the same."""
    return [features[name] for name in FEATURE_NAMES]


def featurize_prompt(prompt: str) -> list[float]:
    """Convenience one-liner: prompt string straight to a feature vector."""
    return vectorize(extract_features(prompt))
