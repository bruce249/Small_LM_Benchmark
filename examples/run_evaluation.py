"""Example: run a summarization evaluation comparing 3 HuggingFace models.

Usage
-----
    # Set your HuggingFace API token first:
    export HF_API_TOKEN="hf_..."

    # Run the example:
    python -m examples.run_evaluation

This script demonstrates the end-to-end pipeline:
  1. TaskGeneratorAgent loads CNN/DailyMail and creates prompted tasks.
  2. ModelRunnerAgent calls the HF Inference API for each model in parallel.
  3. EvaluatorAgent scores outputs with ROUGE and BLEU.
  4. ReportAgent aggregates results and prints a leaderboard.
"""

from __future__ import annotations

import asyncio
import json
import sys

from arena.agents.report_agent import ReportAgent
from arena.experiments.experiment_manager import ExperimentManager
from arena.logging_config import setup_logging
from arena.schemas import TaskType


async def main() -> None:
    setup_logging()

    # ── Configure the experiment ──────────────────────────────────────
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]

    manager = ExperimentManager()
    report = await manager.run(
        task_type=TaskType.SUMMARIZATION,
        model_ids=models,
        max_samples=5,      # Small for quick demo; increase for real evals
        split="test",
    )

    # ── Print human-readable leaderboard ──────────────────────────────
    print("\n" + "=" * 72)
    print("  MODEL EVALUATION ARENA – LEADERBOARD")
    print("=" * 72)

    for entry in report.leaderboard:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry.rank, "  ")
        print(
            f"\n  {medal} #{entry.rank}  {entry.model_id}\n"
            f"       Quality : {entry.avg_quality_score:.4f}\n"
            f"       Latency : {entry.avg_latency_seconds:.2f}s\n"
            f"       Cost    : ${entry.total_cost_usd:.6f}\n"
            f"       Tasks   : {entry.num_tasks}"
        )
        if entry.metric_breakdown:
            for k, v in entry.metric_breakdown.items():
                print(f"         {k:12s}: {v:.4f}")

    print("\n" + "=" * 72)

    # ── Also dump the full JSON report ────────────────────────────────
    report_dict = ReportAgent.report_to_dict(report)
    print("\n── Full JSON Report ──")
    print(json.dumps(report_dict, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
