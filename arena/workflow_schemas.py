"""Workflow-specific schemas for intelligent task decomposition and model recommendation.

This module extends the base arena schemas with data types for:
  - Capability categories (text, voice, image, math, code, etc.)
  - Pipeline steps produced by the TaskDecomposerAgent
  - Per-step benchmark results and model recommendations
  - The final WorkflowRecommendation output
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── Capability categories ────────────────────────────────────────────────────


class Capability(str, Enum):
    """Model capability categories used for routing sub-tasks to models."""

    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    MATH_REASONING = "math_reasoning"
    TRANSLATION = "translation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE_GENERATION = "image_generation"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_TO_TEXT = "image_to_text"
    OBJECT_DETECTION = "object_detection"
    EMBEDDING = "embedding"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DATA_EXTRACTION = "data_extraction"


# ── Pipeline step from decomposition ─────────────────────────────────────────


@dataclass
class PipelineStep:
    """A single step in a decomposed user task.

    Produced by the TaskDecomposerAgent after analysing the user's request.
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    title: str = ""
    description: str = ""
    capability: Capability = Capability.TEXT_GENERATION
    input_description: str = ""
    output_description: str = ""
    test_prompt: str = ""          # A concrete prompt to benchmark candidate models
    reference_output: str = ""     # Expected output for scoring (may be empty)
    depends_on: list[int] = field(default_factory=list)  # step_numbers this depends on
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Candidate model for a step ───────────────────────────────────────────────


@dataclass
class CandidateModel:
    """A model considered for a specific pipeline step."""

    model_id: str = ""
    display_name: str = ""
    capability: Capability = Capability.TEXT_GENERATION
    is_chat_model: bool = True
    cost_per_1k_input: float = 0.0001
    cost_per_1k_output: float = 0.0001
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Per-step benchmark result ─────────────────────────────────────────────────


@dataclass
class StepBenchmarkResult:
    """Benchmark results for one pipeline step across multiple candidate models."""

    step_number: int = 0
    step_title: str = ""
    capability: Capability = Capability.TEXT_GENERATION
    candidates_tested: int = 0
    rankings: list[StepModelRanking] = field(default_factory=list)
    recommended_model: str = ""
    recommendation_reason: str = ""


@dataclass
class StepModelRanking:
    """How a single model performed on a specific pipeline step."""

    model_id: str = ""
    rank: int = 0
    avg_quality_score: float = 0.0
    avg_latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    output_sample: str = ""  # Example output for the user to inspect
    error: str | None = None


# ── Final workflow recommendation ─────────────────────────────────────────────


@dataclass
class WorkflowStep:
    """A single step in the recommended workflow with the best model chosen."""

    step_number: int = 0
    title: str = ""
    description: str = ""
    capability: Capability = Capability.TEXT_GENERATION
    recommended_model: str = ""
    model_display_name: str = ""
    avg_quality_score: float = 0.0
    avg_latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    alternatives: list[str] = field(default_factory=list)  # runner-up model IDs
    input_description: str = ""
    output_description: str = ""


@dataclass
class WorkflowRecommendation:
    """The complete output: a recommended pipeline of models for the user's task."""

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_request: str = ""
    task_analysis: str = ""        # LLM's analysis of what the user wants
    steps: list[WorkflowStep] = field(default_factory=list)
    step_benchmarks: list[StepBenchmarkResult] = field(default_factory=list)
    total_estimated_cost_per_run: float = 0.0
    total_estimated_latency: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "completed"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Serialisation ─────────────────────────────────────────────────────────────


def workflow_to_dict(wf: WorkflowRecommendation) -> dict[str, Any]:
    """Convert a WorkflowRecommendation to a JSON-safe dict."""
    return {
        "workflow_id": wf.workflow_id,
        "user_request": wf.user_request,
        "task_analysis": wf.task_analysis,
        "status": wf.status,
        "created_at": wf.created_at.isoformat(),
        "total_estimated_cost_per_run": wf.total_estimated_cost_per_run,
        "total_estimated_latency": wf.total_estimated_latency,
        "error": wf.error,
        "steps": [
            {
                "step_number": s.step_number,
                "title": s.title,
                "description": s.description,
                "capability": s.capability.value,
                "recommended_model": s.recommended_model,
                "model_display_name": s.model_display_name,
                "avg_quality_score": s.avg_quality_score,
                "avg_latency_seconds": s.avg_latency_seconds,
                "estimated_cost_usd": s.estimated_cost_usd,
                "alternatives": s.alternatives,
                "input_description": s.input_description,
                "output_description": s.output_description,
            }
            for s in wf.steps
        ],
        "step_benchmarks": [
            {
                "step_number": sb.step_number,
                "step_title": sb.step_title,
                "capability": sb.capability.value,
                "candidates_tested": sb.candidates_tested,
                "recommended_model": sb.recommended_model,
                "recommendation_reason": sb.recommendation_reason,
                "rankings": [
                    {
                        "model_id": r.model_id,
                        "rank": r.rank,
                        "avg_quality_score": r.avg_quality_score,
                        "avg_latency_seconds": r.avg_latency_seconds,
                        "estimated_cost_usd": r.estimated_cost_usd,
                        "scores": r.scores,
                        "output_sample": r.output_sample[:500],
                        "error": r.error,
                    }
                    for r in sb.rankings
                ],
            }
            for sb in wf.step_benchmarks
        ],
    }
