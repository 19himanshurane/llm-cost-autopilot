# LLM Cost Autopilot: Cutting LLM Spend by 57% with Automatic Complexity-Based Routing

## The headline

Across a 633-request load test, routing requests to the cheapest model capable of handling them - instead of sending everything to a single flagship model - cut cost by **57.2%** ($0.3632 actual vs. $0.8493 if every request had gone to GPT-4o), with a 0% rate of escalation-worthy quality divergence detected by an automated verification layer running alongside every request.

## The problem

Most teams calling LLM APIs default to sending every request - a one-line data extraction, a two-sentence summary, a genuinely hard multi-step reasoning task - to the same model, usually the most capable (and most expensive) one available. That's the easy default, and it's also a straightforward way to overspend by 2-20x on requests that never needed that much capability in the first place.

## What I built

LLM Cost Autopilot is a routing layer that sits in front of multiple LLM providers (OpenAI, Anthropic, and Groq's free-tier cloud API) and makes the model choice automatically, per request, based on how complex the request actually is:

**A unified model interface** abstracts away provider-specific API differences behind one `send_request()` call, so the rest of the system never has to know or care which provider is behind a given model.

**A complexity classifier**, trained with scikit-learn on a labeled dataset of prompts spanning three tiers (simple extraction/reformatting, moderate summarization/classification, and complex multi-step reasoning/creative generation), predicts which tier a new prompt belongs to using 9 lightweight text features - word count, presence of analysis verbs like "compare" or "design," requested output format complexity, and more. It currently runs at 86% accuracy on held-out test data.

**A router** maps each tier to a specific model via a YAML config file that can be updated live through the API - no redeploy required.

**An asynchronous quality verification loop** runs after every response is returned to the caller: it quietly re-runs the same prompt through the top-tier reference model in a background thread, compares the two answers, and - if they diverge too much - logs the mismatch and feeds the corrected label back into the training dataset for the next classifier retrain.

**A SQLite-backed cost dashboard** (built with Streamlit) and a **FastAPI service** with a `POST /v1/completions` endpoint, model/stats/routing-config endpoints, and a Docker Compose setup round out the system into something that's actually callable like a real service, not just a collection of scripts.

## Results

| Metric | Value |
|---|---|
| Total requests | 633 |
| Actual cost | $0.3632 |
| Cost if every request used GPT-4o | $0.8493 |
| **Savings** | **$0.4861 (57.2%)** |
| Escalation rate | 0.0% |
| Classifier accuracy (held-out test set) | 86.0% |

Routing distribution came out nearly even across the three tiers (35.7% simple / 34.9% moderate / 29.4% complex), and moving the simple tier off a paid model partway through development visibly increased total savings from an earlier 49.4% to the final 57.2%.

## Real-world verification

The load test above ran in mock mode to build and validate the pipeline without spending anything. Since then, the system has been upgraded to run on genuinely real, free infrastructure for two of its three tiers: **Tier 1** and **Tier 2** now run on real Llama models via **Groq's** free-tier API - verified end to end with real responses, near-zero real cost, and real latency (roughly 150ms-1.8s depending on model size). Only **Tier 3**, routed to GPT-4o, remains mocked, since no free tier exists for OpenAI's frontier models. The 57.2% headline figure still reflects the original full-scale mock-mode load test; a full re-run at that scale using real Tier 1/2 traffic is the natural next step to fully replace "simulated" with "real" throughout this report.

## What I'd build next

Quality verification currently uses one general text-similarity threshold rather than task-type-specific checks (exact field matching for extraction, LLM-judged scoring for summarization, label matching for classification), since that requires classifying task *type*, not just complexity tier. Verification also runs as a background thread inside the API process rather than a separate worker service with a real task queue (Celery/Redis) - sufficient for this project's scale, but the first thing I'd change for a system handling meaningfully more traffic. And wiring in a real OpenAI key and re-running this exact load test would be the natural next step to replace "simulated" with "real" for Tier 3 as well.
