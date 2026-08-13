"""
Phase 3, Step 1: Quality thresholds + the scoring function they're measured
against.

The core idea: after a cheap model answers a prompt, we secretly ask the
best model the same prompt, then measure how similar the two answers are.
If they're similar enough, we trust the cheap model's answer was fine. If
they diverge too much, that's a signal the cheap model may have gotten it
wrong, and app/eval/verifier.py (Step 2) will escalate.

HONEST SIMPLIFICATION: the original spec describes different quality
checks per task type (exact field match for extraction, a judged 1-5 score
for summarization, label match for classification). Building all three
properly would require classifying task TYPE, not just complexity tier,
which this project doesn't do yet. This V1 uses one general-purpose
similarity score and one threshold everywhere -- good enough to prove the
verify -> escalate -> log loop actually works end to end. Swap in
task-specific scoring later without changing anything that calls this file.
"""

from __future__ import annotations

from difflib import SequenceMatcher

# Below this similarity score (0.0 = completely different, 1.0 = identical),
# we consider the cheap model's answer to have "diverged" from the top-tier
# model's answer, and it becomes a candidate for escalation.
AGREEMENT_THRESHOLD = 0.35


def similarity_score(text_a: str, text_b: str) -> float:
    """
    Returns a 0.0-1.0 similarity score between two pieces of text.

    Uses Python's built-in difflib.SequenceMatcher -- a classic, dependency-
    free way to compare two strings character-by-character-ish (technically
    it finds the longest matching blocks and ratios them). It's not true
    semantic understanding (it won't know two totally different sentences
    "mean" the same thing), but it's transparent, fast, needs no API calls,
    and is a completely reasonable first-pass "custom scoring" function --
    exactly what the original spec calls for before layering in a real
    LLM-as-judge.
    """
    return SequenceMatcher(None, text_a.strip().lower(), text_b.strip().lower()).ratio()


def passes_quality_threshold(text_a: str, text_b: str) -> bool:
    """True if the two texts agree closely enough to trust the cheaper one."""
    return similarity_score(text_a, text_b) >= AGREEMENT_THRESHOLD
