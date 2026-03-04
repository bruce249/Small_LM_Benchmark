"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    """Body of POST /experiments."""

    task_type: str = Field(
        default="summarization",
        description="One of: summarization, qa, coding, reasoning",
    )
    model_ids: list[str] = Field(
        default_factory=list,
        description="HuggingFace model IDs to benchmark. Empty = use all registered models.",
    )
    dataset_name: str | None = Field(
        default=None,
        description="HuggingFace dataset identifier. None = use default for task type.",
    )
    dataset_config: str | None = Field(default=None)
    split: str = Field(default="test")
    max_samples: int = Field(default=50, ge=1, le=1000)


class ExperimentResponse(BaseModel):
    """Returned by POST /experiments (async mode)."""

    experiment_id: str
    status: str
    message: str = "Experiment queued"


class LeaderboardEntrySchema(BaseModel):
    rank: int
    model_id: str
    avg_quality_score: float
    avg_latency_seconds: float
    total_cost_usd: float
    metric_breakdown: dict[str, float] = {}
    num_tasks: int = 0


class ExperimentReportSchema(BaseModel):
    experiment_id: str
    task_type: str
    dataset_name: str
    models: list[str]
    status: str
    created_at: str
    completed_at: str | None = None
    leaderboard: list[LeaderboardEntrySchema] = []
    error: str | None = None


class ModelInfoSchema(BaseModel):
    model_id: str
    display_name: str
    provider: str
    supported_tasks: list[str]
    max_input_tokens: int
    max_output_tokens: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


# ── Workflow schemas ──────────────────────────────────────────────────────────


class WorkflowRequest(BaseModel):
    """Body of POST /workflow."""

    user_request: str = Field(
        ...,
        description="Natural language description of what the user wants to build.",
        min_length=5,
        max_length=2000,
    )
    decomposer_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Model used to decompose the request into steps.",
    )
    quality_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    latency_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    cost_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class StepModelRankingSchema(BaseModel):
    model_id: str
    rank: int = 0
    avg_quality_score: float = 0.0
    avg_latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    scores: dict[str, float] = {}
    output_sample: str = ""
    error: str | None = None


class StepBenchmarkSchema(BaseModel):
    step_number: int
    step_title: str
    capability: str
    candidates_tested: int = 0
    recommended_model: str = ""
    recommendation_reason: str = ""
    rankings: list[StepModelRankingSchema] = []


class WorkflowStepSchema(BaseModel):
    step_number: int
    title: str
    description: str
    capability: str
    recommended_model: str
    model_display_name: str = ""
    avg_quality_score: float = 0.0
    avg_latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    alternatives: list[str] = []
    input_description: str = ""
    output_description: str = ""


class WorkflowResponse(BaseModel):
    """Returned by POST /workflow."""

    workflow_id: str
    user_request: str
    task_analysis: str
    status: str
    created_at: str
    total_estimated_cost_per_run: float = 0.0
    total_estimated_latency: float = 0.0
    steps: list[WorkflowStepSchema] = []
    step_benchmarks: list[StepBenchmarkSchema] = []
    error: str | None = None
