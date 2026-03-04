"""Streamlit dashboard – interactive UI for the Model Evaluation Arena."""

from __future__ import annotations

import json
import time

import pandas as pd
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

TASK_TYPES = ["summarization", "qa", "coding", "reasoning"]

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Model Evaluation Arena",
    page_icon="🏟️",
    layout="wide",
)

st.title("🏟️ Open Source Model Evaluation Arena")

# ── Top-level navigation ─────────────────────────────────────────────────────

mode = st.radio(
    "Choose mode:",
    ["🏆 Model Benchmark", "🧠 Intelligent Workflow Builder"],
    horizontal=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Classic Model Benchmark
# ══════════════════════════════════════════════════════════════════════════════

if mode == "🏆 Model Benchmark":
    st.markdown(
        "Benchmark HuggingFace models on summarization, QA, coding, and reasoning tasks."
    )

    # ── Sidebar: experiment configuration ─────────────────────────────────

    with st.sidebar:
        st.header("⚙️ Experiment Configuration")

        task_type = st.selectbox("Task Type", TASK_TYPES, index=0)

        selected_models = st.multiselect(
            "Models to evaluate",
            DEFAULT_MODELS,
            default=DEFAULT_MODELS[:3],
        )

        custom_model = st.text_input("Add custom model ID (HuggingFace)")
        if custom_model and custom_model not in selected_models:
            selected_models.append(custom_model)

        dataset_name = st.text_input("Dataset (leave blank for default)", value="")
        max_samples = st.slider("Max samples", min_value=1, max_value=200, value=10)
        split = st.selectbox("Dataset split", ["test", "validation", "train"], index=0)

        run_btn = st.button("🚀 Run Experiment", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────────────────

    tabs = st.tabs(["🏆 Leaderboard", "📊 Detailed Results", "📝 Raw JSON"])

    def _run_experiment() -> dict | None:
        """POST to the sync experiment endpoint and return the report."""
        payload = {
            "task_type": task_type,
            "model_ids": selected_models,
            "dataset_name": dataset_name or None,
            "split": split,
            "max_samples": max_samples,
        }
        try:
            with st.spinner("Running evaluation pipeline… this may take a few minutes."):
                resp = requests.post(f"{API_BASE}/experiments/sync", json=payload, timeout=600)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"API error {resp.status_code}: {resp.text[:500]}")
                return None
        except requests.ConnectionError:
            st.error(
                "Cannot reach the API server. Make sure it is running:\n\n"
                "```\nuvicorn arena.api.main:app --reload\n```"
            )
            return None

    if run_btn:
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            report = _run_experiment()
            if report:
                st.session_state["report"] = report

    # ── Render results ────────────────────────────────────────────────────

    report = st.session_state.get("report")

    if report:
        with tabs[0]:
            st.subheader("🏆 Leaderboard")
            leaderboard = report.get("leaderboard", [])
            if leaderboard:
                df = pd.DataFrame(leaderboard)
                df = df.sort_values("rank")

                medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                df[""] = df["rank"].map(lambda r: medals.get(r, ""))

                display_cols = [
                    "", "rank", "model_id", "avg_quality_score",
                    "avg_latency_seconds", "total_cost_usd", "num_tasks",
                ]
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

                st.subheader("Quality Score Comparison")
                chart_df = df.set_index("model_id")[["avg_quality_score"]]
                st.bar_chart(chart_df)

                st.subheader("Average Latency (seconds)")
                lat_df = df.set_index("model_id")[["avg_latency_seconds"]]
                st.bar_chart(lat_df)

                if any(e.get("metric_breakdown") for e in leaderboard):
                    st.subheader("Metric Breakdown")
                    breakdown_rows = []
                    for e in leaderboard:
                        row = {"model_id": e["model_id"]}
                        row.update(e.get("metric_breakdown", {}))
                        breakdown_rows.append(row)
                    bd_df = pd.DataFrame(breakdown_rows)
                    st.dataframe(bd_df, use_container_width=True, hide_index=True)
            else:
                st.info("No leaderboard data available.")

        with tabs[1]:
            st.subheader("📊 Detailed Results")
            details = report.get("detailed_results", [])
            if details:
                det_df = pd.DataFrame(details)
                st.dataframe(det_df, use_container_width=True, hide_index=True)
            else:
                st.info("No detailed results.")

        with tabs[2]:
            st.subheader("📝 Raw JSON Response")
            st.json(report)
    else:
        st.info("Configure an experiment in the sidebar and click **Run Experiment** to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Intelligent Workflow Builder
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown(
        "Describe what you want to build in plain English. The system will:\n"
        "1. **Decompose** your request into pipeline steps using an LLM\n"
        "2. **Identify** capability categories for each step (text, voice, image, code…)\n"
        "3. **Benchmark** candidate models for each step\n"
        "4. **Recommend** the best model for every step"
    )

    # ── Sidebar: workflow configuration ───────────────────────────────────

    with st.sidebar:
        st.header("🧠 Workflow Configuration")

        decomposer_model = st.selectbox(
            "Decomposer Model",
            ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
            index=0,
            help="The LLM used to analyse and decompose your request.",
        )

        st.subheader("Scoring Weights")
        quality_w = st.slider("Quality weight", 0.0, 1.0, 0.60, 0.05)
        latency_w = st.slider("Latency weight", 0.0, 1.0, 0.25, 0.05)
        cost_w = st.slider("Cost weight", 0.0, 1.0, 0.15, 0.05)

    # ── User request input ────────────────────────────────────────────────

    st.subheader("💡 Describe Your Project")

    example_requests = [
        "I want to build a podcast summariser that takes audio, transcribes it, summarises the text, and generates a short audio summary.",
        "Build me a chatbot that can translate between English and Spanish, answer questions about documents, and generate code snippets.",
        "I need an app that takes product images, classifies them, generates a description, and detects objects in the image.",
        "Create a math tutoring assistant that can solve equations step by step, generate practice problems, and explain concepts clearly.",
    ]

    user_request = st.text_area(
        "What do you want to build?",
        height=100,
        placeholder="e.g., I want to build a voice assistant that translates languages and summarises documents…",
    )

    with st.expander("💡 Example requests (click to copy)"):
        for i, ex in enumerate(example_requests):
            if st.button(f"Use example {i+1}", key=f"ex_{i}"):
                st.session_state["wf_user_request"] = ex
                st.rerun()

    # If an example was selected, use it
    if "wf_user_request" in st.session_state and not user_request:
        user_request = st.session_state.pop("wf_user_request", "")

    build_btn = st.button("🔨 Build Workflow", type="primary", use_container_width=True)

    # ── Run workflow ──────────────────────────────────────────────────────

    def _run_workflow(request_text: str) -> dict | None:
        payload = {
            "user_request": request_text,
            "decomposer_model": decomposer_model,
            "quality_weight": quality_w,
            "latency_weight": latency_w,
            "cost_weight": cost_w,
        }
        try:
            with st.spinner("🧠 Decomposing task and benchmarking models… this may take a few minutes."):
                resp = requests.post(f"{API_BASE}/workflow", json=payload, timeout=900)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"API error {resp.status_code}: {resp.text[:500]}")
                return None
        except requests.ConnectionError:
            st.error(
                "Cannot reach the API server. Make sure it is running:\n\n"
                "```\nuvicorn arena.api.main:app --reload\n```"
            )
            return None

    if build_btn:
        if not user_request or len(user_request.strip()) < 5:
            st.warning("Please describe what you want to build (at least 5 characters).")
        else:
            wf = _run_workflow(user_request.strip())
            if wf:
                st.session_state["workflow"] = wf

    # ── Render workflow ───────────────────────────────────────────────────

    wf = st.session_state.get("workflow")

    if wf:
        wf_tabs = st.tabs([
            "📋 Pipeline Overview",
            "🏆 Per-Step Benchmarks",
            "🗺️ Workflow Diagram",
            "📝 Raw JSON",
        ])

        # ── Tab 1: Pipeline overview ──────────────────────────────────────

        with wf_tabs[0]:
            st.subheader("📋 Recommended Pipeline")

            # Task analysis
            st.info(f"**Task Analysis:** {wf.get('task_analysis', 'N/A')}")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Steps", len(wf.get("steps", [])))
            col2.metric("Est. Latency", f"{wf.get('total_estimated_latency', 0):.2f}s")
            col3.metric("Est. Cost", f"${wf.get('total_estimated_cost_per_run', 0):.6f}")
            col4.metric("Status", wf.get("status", "N/A"))

            st.divider()

            # Step cards
            for step in wf.get("steps", []):
                with st.container():
                    st.markdown(f"### Step {step['step_number']}: {step['title']}")
                    cols = st.columns([2, 1, 1, 1])
                    cols[0].markdown(f"**Description:** {step['description']}")
                    cols[1].markdown(f"**Capability:** `{step['capability']}`")
                    cols[2].markdown(f"**Recommended Model:**\n`{step['recommended_model']}`")
                    cols[3].markdown(
                        f"**Quality:** {step['avg_quality_score']:.3f}\n\n"
                        f"**Latency:** {step['avg_latency_seconds']:.2f}s"
                    )

                    if step.get("alternatives"):
                        st.markdown(
                            "**Alternatives:** " + ", ".join(f"`{a}`" for a in step["alternatives"])
                        )

                    if step.get("input_description") or step.get("output_description"):
                        ic, oc = st.columns(2)
                        ic.markdown(f"**Input:** {step.get('input_description', '')}")
                        oc.markdown(f"**Output:** {step.get('output_description', '')}")

                    st.divider()

        # ── Tab 2: Per-step benchmarks ────────────────────────────────────

        with wf_tabs[1]:
            st.subheader("🏆 Per-Step Benchmark Results")

            for bench in wf.get("step_benchmarks", []):
                st.markdown(
                    f"### Step {bench['step_number']}: {bench['step_title']} "
                    f"(`{bench['capability']}`)"
                )
                st.markdown(f"**Recommendation:** {bench.get('recommendation_reason', 'N/A')}")

                rankings = bench.get("rankings", [])
                if rankings:
                    rank_data = []
                    for r in rankings:
                        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                        rank_data.append({
                            "": medals.get(r["rank"], ""),
                            "Rank": r["rank"],
                            "Model": r["model_id"],
                            "Quality": round(r["avg_quality_score"], 4),
                            "Latency (s)": round(r["avg_latency_seconds"], 3),
                            "Cost ($)": round(r["estimated_cost_usd"], 6),
                            "Error": r.get("error") or "",
                        })
                    rank_df = pd.DataFrame(rank_data)
                    st.dataframe(rank_df, use_container_width=True, hide_index=True)

                    # Show output samples in expander
                    for r in rankings:
                        sample = r.get("output_sample", "").strip()
                        if sample:
                            with st.expander(f"Output sample: {r['model_id']}"):
                                st.text(sample)

                    # Bar chart
                    valid = [r for r in rankings if not r.get("error")]
                    if valid:
                        chart_data = pd.DataFrame([
                            {"Model": r["model_id"], "Quality": r["avg_quality_score"]}
                            for r in valid
                        ]).set_index("Model")
                        st.bar_chart(chart_data)
                else:
                    st.info("No benchmark data for this step.")
                st.divider()

        # ── Tab 3: Workflow diagram ───────────────────────────────────────

        with wf_tabs[2]:
            st.subheader("🗺️ Workflow Diagram")

            steps = wf.get("steps", [])
            if steps:
                # Build a text-based pipeline diagram
                diagram_lines = []
                for i, step in enumerate(steps):
                    model_short = step["recommended_model"].split("/")[-1]
                    box = (
                        f"┌─{'─' * 50}─┐\n"
                        f"│ Step {step['step_number']}: {step['title']:<42} │\n"
                        f"│ Capability: {step['capability']:<37} │\n"
                        f"│ Model: {model_short:<42} │\n"
                        f"└─{'─' * 50}─┘"
                    )
                    diagram_lines.append(box)
                    if i < len(steps) - 1:
                        diagram_lines.append(f"{'':>25}▼")

                st.code("\n".join(diagram_lines), language=None)

                # Also show as a clean table
                st.subheader("Pipeline Summary Table")
                summary_data = [{
                    "Step": s["step_number"],
                    "Title": s["title"],
                    "Capability": s["capability"],
                    "Model": s["recommended_model"],
                    "Quality": round(s["avg_quality_score"], 4),
                    "Latency (s)": round(s["avg_latency_seconds"], 3),
                } for s in steps]
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            else:
                st.info("No steps to display.")

        # ── Tab 4: Raw JSON ───────────────────────────────────────────────

        with wf_tabs[3]:
            st.subheader("📝 Raw JSON Response")
            st.json(wf)

    else:
        st.info("Describe your project above and click **Build Workflow** to get started.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Open Source Model Evaluation Arena • "
    "Powered by HuggingFace Inference API, FastAPI, and Streamlit"
)
