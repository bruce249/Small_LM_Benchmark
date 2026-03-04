"""Example: run the intelligent workflow builder end-to-end.

This script demonstrates the full pipeline:
1. User describes what they want to build (natural language).
2. TaskDecomposerAgent uses an LLM to break it into capability-tagged steps.
3. WorkflowOptimizerAgent benchmarks candidate models per step.
4. A complete pipeline recommendation is printed.

Usage
-----
    $env:HF_API_TOKEN="hf_..."
    python -m examples.run_workflow
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.agents.task_decomposer import TaskDecomposerAgent
from arena.agents.workflow_optimizer import WorkflowOptimizerAgent
from arena.workflow_schemas import workflow_to_dict


async def main() -> None:
    # ── 1. Define the user request ────────────────────────────────────────
    user_request = (
        "I want to build a podcast summariser that takes audio, "
        "transcribes it, summarises the text, and generates a short audio summary."
    )

    print("=" * 72)
    print("🧠  Intelligent Workflow Builder")
    print("=" * 72)
    print(f"\n📝 User Request:\n   {user_request}\n")

    # ── 2. Decompose the request ──────────────────────────────────────────
    print("─" * 72)
    print("Step 1: Decomposing request into pipeline steps…")
    print("─" * 72)

    decomposer = TaskDecomposerAgent(
        decomposer_model="Qwen/Qwen2.5-7B-Instruct",
    )
    analysis, steps = await decomposer.decompose(user_request)

    print(f"\n📊 Analysis: {analysis}\n")
    print(f"🔧 Decomposed into {len(steps)} steps:")
    for s in steps:
        print(
            f"   Step {s.step_number}: {s.title} "
            f"[{s.capability.value}]"
        )
        print(f"      Description: {s.description}")
        print(f"      Test prompt: {s.test_prompt[:100]}…" if len(s.test_prompt) > 100 else f"      Test prompt: {s.test_prompt}")
        print()

    # ── 3. Benchmark and optimise ─────────────────────────────────────────
    print("─" * 72)
    print("Step 2: Benchmarking candidate models per step…")
    print("─" * 72)

    optimizer = WorkflowOptimizerAgent(
        quality_weight=0.60,
        latency_weight=0.25,
        cost_weight=0.15,
    )

    recommendation = await optimizer.optimize(
        user_request=user_request,
        task_analysis=analysis,
        steps=steps,
    )

    # ── 4. Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("🏆  WORKFLOW RECOMMENDATION")
    print("=" * 72)

    print(f"\n📊 Task Analysis: {recommendation.task_analysis}")
    print(f"📋 Total Steps: {len(recommendation.steps)}")
    print(f"⏱️  Est. Total Latency: {recommendation.total_estimated_latency:.2f}s")
    print(f"💰 Est. Total Cost: ${recommendation.total_estimated_cost_per_run:.6f}\n")

    for ws in recommendation.steps:
        print(f"  ┌─ Step {ws.step_number}: {ws.title}")
        print(f"  │  Capability:  {ws.capability.value}")
        print(f"  │  Model:       {ws.recommended_model} ({ws.model_display_name})")
        print(f"  │  Quality:     {ws.avg_quality_score:.4f}")
        print(f"  │  Latency:     {ws.avg_latency_seconds:.3f}s")
        print(f"  │  Cost:        ${ws.estimated_cost_usd:.6f}")
        if ws.alternatives:
            print(f"  │  Alternatives: {', '.join(ws.alternatives)}")
        print(f"  │  Input:       {ws.input_description}")
        print(f"  │  Output:      {ws.output_description}")
        print(f"  └{'─' * 50}")
        print()

    # ── 5. Per-step benchmark details ─────────────────────────────────────
    print("─" * 72)
    print("🔬  Per-Step Benchmark Details")
    print("─" * 72)

    for bench in recommendation.step_benchmarks:
        print(f"\n  Step {bench.step_number}: {bench.step_title} [{bench.capability.value}]")
        print(f"  Candidates tested: {bench.candidates_tested}")
        print(f"  Recommended: {bench.recommended_model}")
        print(f"  Reason: {bench.recommendation_reason}")

        for r in bench.rankings:
            status = "❌ " + r.error if r.error else "✅"
            print(
                f"    #{r.rank} {r.model_id}  "
                f"quality={r.avg_quality_score:.4f}  "
                f"latency={r.avg_latency_seconds:.3f}s  "
                f"{status}"
            )

    # ── 6. Save full JSON ─────────────────────────────────────────────────
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "workflow_output.json",
    )
    with open(output_path, "w") as f:
        json.dump(workflow_to_dict(recommendation), f, indent=2)

    print(f"\n💾 Full JSON saved to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
