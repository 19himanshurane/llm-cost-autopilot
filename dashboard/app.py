"""
Phase 4, Step 2: The cost dashboard.

A Streamlit app that reads data/autopilot.db and shows the headline
"money shot" metric (% cost saved vs. sending everything to GPT-4o), cost
per day, routing distribution, requests per complexity tier, quality score
distribution, and escalation rate.

Run from the project root:
    streamlit run dashboard/app.py

This opens a browser tab automatically -- it's a live local web server,
not a static file, so it keeps running in your terminal until you press
Ctrl+C there.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "autopilot.db"

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")


@st.cache_data(ttl=10)
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM requests", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    return df


df = load_data()

st.title("LLM Cost Autopilot — Cost Dashboard")

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

st.divider()
st.caption(
    "NOTE: figures reflect MOCK responses unless real API keys are configured "
    "in .env — costs/latencies are simulated, not real, until then."
)
