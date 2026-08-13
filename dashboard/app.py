"""
The cost dashboard.

A Streamlit app that reads data/autopilot.db and shows the headline
"money shot" metric (% cost saved vs. sending everything to GPT-4o), cost
per day, routing distribution, requests per complexity tier, quality score
distribution, and escalation rate. Also includes a "Try it live" tab that
sends a real prompt to the running API and shows how it gets routed.

Run from the project root (with the API also running in another terminal):
    uvicorn app.api.main:app --reload
    streamlit run dashboard/app.py

This opens a browser tab automatically -- it's a live local web server,
not a static file, so it keeps running in your terminal until you press
Ctrl+C there.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "autopilot.db"
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")


@st.cache_data(ttl=10)
def load_data() -> pd.DataFrame:
    resp = httpx.get(f"{API_BASE_URL}/v1/requests", timeout=30.0)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    return df


def load_data_safe() -> pd.DataFrame | None:
    """Wraps load_data() so a slow/cold/misconfigured API shows a clear
    message instead of an unhandled traceback filling the page."""
    try:
        return load_data()
    except Exception as exc:
        st.error(
            f"Couldn't reach the API at {API_BASE_URL} to load dashboard data ({exc}). "
            "If it's been idle a while, Render's free tier can take up to a minute to "
            "wake up -- try refreshing in a moment."
        )
        return None


st.title("LLM Cost Autopilot - Cost Dashboard")

tab_dashboard, tab_try_it = st.tabs(["Cost Dashboard", "Try it live"])

with tab_dashboard:
    df = load_data_safe()

    if df is None:
        st.stop()

    if df.empty:
        st.warning("No requests logged yet. Run `python -m scripts.populate_demo_data` first.")
        st.stop()

    total_cost = df["cost_usd"].sum()
    baseline_cost = df["baseline_cost_usd"].sum()
    savings_usd = baseline_cost - total_cost
    savings_pct = (savings_usd / baseline_cost) if baseline_cost else 0

    # ---- Headline metric ----
    st.subheader("The money shot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual cost (with routing)", f"${total_cost:.4f}")
    col2.metric("Cost if everything used GPT-4o", f"${baseline_cost:.4f}")
    col3.metric("Saved", f"${savings_usd:.4f}", f"{savings_pct:.1%}")

    st.divider()

    # ---- Cost per day ----
    st.subheader("Cost per day")
    by_day = df.groupby("date")[["cost_usd", "baseline_cost_usd"]].sum()
    by_day.columns = ["Actual cost", "Baseline (all GPT-4o) cost"]
    st.bar_chart(by_day)

    # ---- Routing distribution ----
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Routing distribution")
        st.caption("Which model handled what share of requests")
        st.bar_chart(df["model_name"].value_counts())
    with col_b:
        st.subheader("Requests by complexity tier")
        tier_labels = {1: "Tier 1 (Simple)", 2: "Tier 2 (Moderate)", 3: "Tier 3 (Complex)"}
        st.bar_chart(df["tier"].map(tier_labels).value_counts())

    st.divider()

    # ---- Quality + escalation ----
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Quality score distribution")
        quality_scores = df["quality_score"].dropna()
        if not quality_scores.empty:
            binned = pd.cut(quality_scores, bins=10)
            counts = binned.value_counts().sort_index()
            counts.index = counts.index.astype(str)  # clean labels, not raw Interval objects
            st.bar_chart(counts)
        else:
            st.write("No quality scores recorded yet.")
    with col_d:
        st.subheader("Escalation rate")
        escalation_rate = df["was_escalated"].mean()
        st.metric(
            "Escalated requests",
            f"{escalation_rate:.1%}",
            f"{int(df['was_escalated'].sum())} of {len(df)}",
        )
        st.caption(
            "Not meaningful yet: escalation compares the cheap model's answer "
            "against the Tier 3 reference model, which is currently mocked "
            "(no paid OpenAI key configured). A real answer will rarely "
            "text-match a canned mock string, so this rate reads high by "
            "construction, not because the cheap models are performing badly."
        )

    st.divider()
    st.caption(
        "NOTE: figures include a mix of real and mock responses. Tier 1 and Tier 2 requests logged since a Groq API key was configured are real (see was_mocked in the database); Tier 3 (GPT-4o) remains mocked, since no paid key is configured for it. Because verification compares against that mocked Tier 3 response, the escalation rate above is not a reliable quality signal in this configuration."
    )

with tab_try_it:
    st.subheader("Send a real prompt through the live router")
    st.caption(
        f"This calls your running API at {API_BASE_URL}/v1/completions - make sure "
        "`uvicorn app.api.main:app --reload` is running in another terminal."
    )

    example_prompts = {
        "Simple (Tier 1 expected)": "What is the capital of Japan?",
        "Moderate (Tier 2 expected)": "Classify this review as positive or negative: The battery life is amazing but customer service was slow.",
        "Complex (Tier 3 expected)": "Design a fault-tolerant architecture for a real-time payment system handling 1 million transactions per day.",
    }
    choice = st.selectbox("Try an example, or write your own below:", ["(write my own)"] + list(example_prompts.keys()))
    default_text = "" if choice == "(write my own)" else example_prompts[choice]

    prompt = st.text_area("Prompt", value=default_text, height=100)
    send = st.button("Route & send", type="primary")

    if send and prompt.strip():
        with st.spinner("Classifying complexity, routing, and calling the model..."):
            try:
                resp = httpx.post(f"{API_BASE_URL}/v1/completions", json={"prompt": prompt}, timeout=60.0)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                st.error(f"Couldn't reach the API at {API_BASE_URL}. Is `uvicorn app.api.main:app --reload` running? ({exc})")
            else:
                tier_names = {1: "Tier 1 - Simple", 2: "Tier 2 - Moderate", 3: "Tier 3 - Complex"}
                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity tier", tier_names.get(result["tier"], result["tier"]))
                c2.metric("Model selected", result["selected_model"])
                c3.metric("Real or mocked?", "Real" if not result["was_mocked"] else "Mocked")

                st.markdown(f"**Why this model:** {result.get('reason', 'n/a')}")
                st.markdown("**Response:**")
                st.info(result["text"])

                c4, c5, c6 = st.columns(3)
                c4.metric("Cost", f"${result['cost_usd']:.6f}")
                c5.metric("Latency", f"{result['latency_ms']} ms")
                c6.metric("Tokens (in/out)", f"{result['input_tokens']} / {result['output_tokens']}")
    elif send:
        st.warning("Type a prompt first.")
