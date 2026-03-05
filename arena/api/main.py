"""FastAPI application – REST endpoints for the Evaluation Arena."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from arena import __version__
from arena.api.schemas import (
    ExperimentReportSchema,
    ExperimentRequest,
    ExperimentResponse,
    HealthResponse,
    LeaderboardEntrySchema,
    ModelInfoSchema,
    StepBenchmarkSchema,
    StepModelRankingSchema,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowStepSchema,
)
from arena.agents.report_agent import ReportAgent
from arena.agents.task_decomposer import TaskDecomposerAgent
from arena.agents.workflow_optimizer import WorkflowOptimizerAgent
from arena.config import get_settings
from arena.db.session import get_db
from arena.experiments.experiment_manager import ExperimentManager
from arena.logging_config import get_logger, setup_logging
from arena.schemas import ExperimentReport, TaskType
from arena.services.model_registry import ModelRegistry
from arena.workflow_schemas import WorkflowRecommendation, workflow_to_dict

logger = get_logger("api")

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

# ── Singleton services ────────────────────────────────────────────────────────
_registry = ModelRegistry()
_manager = ExperimentManager(model_registry=_registry)

# In-memory cache of completed reports (production would use DB/Redis)
_reports: dict[str, ExperimentReport] = {}
_workflows: dict[str, WorkflowRecommendation] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle hook."""
    setup_logging()
    logger.info("Arena API v%s starting …", __version__)
    yield
    logger.info("Arena API shutting down")


app = FastAPI(
    title="Open Source Model Evaluation Arena",
    description="Benchmark HuggingFace models on summarization, QA, coding, and reasoning tasks.",
    version=__version__,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Token extraction helper ───────────────────────────────────────────────────


def _extract_token(authorization: str | None = Header(None)) -> str | None:
    """Extract the HF token from the Authorization header.

    Accepts ``Bearer <token>`` or a bare token string.
    Falls back to the environment variable when no header is sent.
    """
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization
    # Fallback: env var
    settings = get_settings()
    return settings.hf_api_token or None


# ── Token validation endpoint ─────────────────────────────────────────────────


@app.post("/auth/validate")
async def validate_token(request: Request):
    """Validate a HuggingFace API token.

    Strategy:
    1. Try ``/api/whoami`` – works for classic (read/write) tokens.
    2. If whoami returns 401, try a lightweight Inference API call.
    3. If inference returns 200/402/403/429 → token is valid (402 = out of credits).
    4. If everything returns 401, accept the token anyway if the format looks
       correct (``hf_`` prefix, 30+ chars) – fine-grained tokens may lack
       permissions for both endpoints.  Errors will surface during actual usage.
    """
    body = await request.json()
    token = body.get("token", "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    # Basic format check
    if not token.startswith("hf_") or len(token) < 10:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "detail": "Token must start with 'hf_' and be at least 10 characters"},
        )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # ── Attempt 1: whoami (classic tokens) ────────────────────
            resp = await client.get(
                "https://huggingface.co/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "valid": True,
                    "username": data.get("name", data.get("fullname", "user")),
                    "email": data.get("email", ""),
                    "verified_via": "whoami",
                }

            # ── Attempt 2: Inference API (fine-grained tokens) ────────
            try:
                inf_resp = await client.post(
                    "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-7B-Instruct/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=20,
                )
                if inf_resp.status_code in (200, 402, 403, 429):
                    # 200=works, 402=out of credits, 403=gated, 429=rate limit
                    # All prove the token IS valid, just maybe out of credits
                    return {
                        "valid": True,
                        "username": "HF User",
                        "email": "",
                        "verified_via": "inference",
                    }
            except Exception:
                pass  # Inference check failed, continue to fallback

            # ── Fallback: accept well-formed tokens ──────────────────
            # Fine-grained tokens with limited scopes may fail both whoami
            # and inference.  Accept them based on format; errors will be
            # shown during actual workflow/benchmark usage.
            if len(token) >= 30:
                return {
                    "valid": True,
                    "username": "HF User",
                    "email": "",
                    "verified_via": "format",
                }

            # Token too short to be real
            return JSONResponse(
                status_code=401,
                content={
                    "valid": False,
                    "detail": "Token rejected by HuggingFace. Please create a new token at huggingface.co/settings/tokens",
                },
            )
    except httpx.TimeoutException:
        # On timeout, still accept well-formed tokens
        if len(token) >= 30:
            return {"valid": True, "username": "HF User", "email": "", "verified_via": "format"}
        raise HTTPException(status_code=504, detail="HuggingFace API timeout")
    except Exception as exc:
        # On any error, accept well-formed tokens rather than blocking
        if len(token) >= 30:
            return {"valid": True, "username": "HF User", "email": "", "verified_via": "format"}
        raise HTTPException(status_code=500, detail=f"Validation error: {exc}")


# ── Static files ──────────────────────────────────────────────────────────────

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend SPA."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Arena API is running. Frontend not found at /frontend/index.html."}


# ── Helper ────────────────────────────────────────────────────────────────────


def _resolve_task_type(raw: str) -> TaskType:
    try:
        return TaskType(raw.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type '{raw}'. Choose from: {[t.value for t in TaskType]}",
        )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.get("/models", response_model=list[ModelInfoSchema])
async def list_models(task_type: str | None = None):
    """List all registered models, optionally filtered by task type."""
    tt = _resolve_task_type(task_type) if task_type else None
    return [
        ModelInfoSchema(
            model_id=m.model_id,
            display_name=m.display_name,
            provider=m.provider,
            supported_tasks=[t.value for t in m.supported_tasks],
            max_input_tokens=m.max_input_tokens,
            max_output_tokens=m.max_output_tokens,
        )
        for m in _registry.list_models(tt)
    ]


@app.post("/experiments", response_model=ExperimentResponse, status_code=202)
async def create_experiment(
    body: ExperimentRequest,
    background_tasks: BackgroundTasks,
    token: str | None = Depends(_extract_token),
):
    """Launch an evaluation experiment (async via background task)."""
    task_type = _resolve_task_type(body.task_type)
    exp_id = str(uuid.uuid4())

    async def _run() -> None:
        try:
            mgr = ExperimentManager(model_registry=_registry, hf_token=token)
            report = await mgr.run(
                task_type=task_type,
                model_ids=body.model_ids or None,
                dataset_name=body.dataset_name,
                dataset_config=body.dataset_config,
                split=body.split,
                max_samples=body.max_samples,
                experiment_id=exp_id,
            )
            _reports[exp_id] = report
        except Exception:
            logger.exception("Background experiment %s failed", exp_id)

    background_tasks.add_task(_run)

    return ExperimentResponse(
        experiment_id=exp_id,
        status="pending",
        message="Experiment queued – poll GET /experiments/{id} for results",
    )


@app.post("/experiments/sync", response_model=ExperimentReportSchema)
async def create_experiment_sync(
    body: ExperimentRequest,
    token: str | None = Depends(_extract_token),
):
    """Run an experiment synchronously and return the full report."""
    task_type = _resolve_task_type(body.task_type)
    mgr = ExperimentManager(model_registry=_registry, hf_token=token)

    report = await mgr.run(
        task_type=task_type,
        model_ids=body.model_ids or None,
        dataset_name=body.dataset_name,
        dataset_config=body.dataset_config,
        split=body.split,
        max_samples=body.max_samples,
    )
    _reports[report.experiment_id] = report
    return _report_to_schema(report)


@app.get("/experiments/{experiment_id}", response_model=ExperimentReportSchema)
async def get_experiment(experiment_id: str):
    """Retrieve the result of a previously launched experiment."""
    report = _reports.get(experiment_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Experiment not found or still running")
    return _report_to_schema(report)


@app.get("/leaderboard/{experiment_id}", response_model=list[LeaderboardEntrySchema])
async def get_leaderboard(experiment_id: str):
    """Get just the leaderboard for an experiment."""
    report = _reports.get(experiment_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return [
        LeaderboardEntrySchema(
            rank=e.rank,
            model_id=e.model_id,
            avg_quality_score=e.avg_quality_score,
            avg_latency_seconds=e.avg_latency_seconds,
            total_cost_usd=e.total_cost_usd,
            metric_breakdown=e.metric_breakdown,
            num_tasks=e.num_tasks,
        )
        for e in report.leaderboard
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _report_to_schema(report: ExperimentReport) -> ExperimentReportSchema:
    return ExperimentReportSchema(
        experiment_id=report.experiment_id,
        task_type=report.task_type.value,
        dataset_name=report.dataset_name,
        models=report.models,
        status=report.status.value,
        created_at=report.created_at.isoformat(),
        completed_at=report.completed_at.isoformat() if report.completed_at else None,
        leaderboard=[
            LeaderboardEntrySchema(
                rank=e.rank,
                model_id=e.model_id,
                avg_quality_score=e.avg_quality_score,
                avg_latency_seconds=e.avg_latency_seconds,
                total_cost_usd=e.total_cost_usd,
                metric_breakdown=e.metric_breakdown,
                num_tasks=e.num_tasks,
            )
            for e in report.leaderboard
        ],
        error=report.error,
    )


# ── Workflow endpoints ────────────────────────────────────────────────────────


@app.post("/workflow", response_model=WorkflowResponse)
async def create_workflow(
    body: WorkflowRequest,
    token: str | None = Depends(_extract_token),
):
    """Decompose a user request into steps, benchmark candidates, and recommend a pipeline.

    This is the main "intelligent workflow builder" endpoint. It:
    1. Uses an LLM to decompose the user's request into capability-tagged steps.
    2. For each step, discovers candidate models and benchmarks them.
    3. Returns a complete pipeline recommendation with the best model per step.
    """
    decomposer = TaskDecomposerAgent(decomposer_model=body.decomposer_model, hf_token=token)
    optimizer = WorkflowOptimizerAgent(
        quality_weight=body.quality_weight,
        latency_weight=body.latency_weight,
        cost_weight=body.cost_weight,
        hf_token=token,
    )

    try:
        # Step 1: Decompose the user request
        analysis, steps = await decomposer.decompose(body.user_request)

        # Step 2: Benchmark and optimise
        recommendation = await optimizer.optimize(
            user_request=body.user_request,
            task_analysis=analysis,
            steps=steps,
        )
        _workflows[recommendation.workflow_id] = recommendation
        return _workflow_to_schema(recommendation)

    except Exception as exc:
        logger.exception("Workflow creation failed")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}")


@app.post("/workflow/async", response_model=ExperimentResponse, status_code=202)
async def create_workflow_async(
    body: WorkflowRequest,
    background_tasks: BackgroundTasks,
    token: str | None = Depends(_extract_token),
):
    """Launch a workflow build asynchronously."""
    wf_id = str(uuid.uuid4())

    async def _run() -> None:
        try:
            decomposer = TaskDecomposerAgent(decomposer_model=body.decomposer_model, hf_token=token)
            optimizer = WorkflowOptimizerAgent(
                quality_weight=body.quality_weight,
                latency_weight=body.latency_weight,
                cost_weight=body.cost_weight,
                hf_token=token,
            )
            analysis, steps = await decomposer.decompose(body.user_request)
            recommendation = await optimizer.optimize(
                user_request=body.user_request,
                task_analysis=analysis,
                steps=steps,
            )
            recommendation.workflow_id = wf_id
            _workflows[wf_id] = recommendation
        except Exception:
            logger.exception("Background workflow %s failed", wf_id)

    background_tasks.add_task(_run)
    return ExperimentResponse(
        experiment_id=wf_id,
        status="pending",
        message="Workflow build queued – poll GET /workflow/{id} for results",
    )


@app.get("/workflow/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Retrieve a completed workflow recommendation."""
    wf = _workflows.get(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found or still running")
    return _workflow_to_schema(wf)


def _workflow_to_schema(wf: WorkflowRecommendation) -> WorkflowResponse:
    return WorkflowResponse(
        workflow_id=wf.workflow_id,
        user_request=wf.user_request,
        task_analysis=wf.task_analysis,
        status=wf.status,
        created_at=wf.created_at.isoformat(),
        total_estimated_cost_per_run=wf.total_estimated_cost_per_run,
        total_estimated_latency=wf.total_estimated_latency,
        error=wf.error,
        steps=[
            WorkflowStepSchema(
                step_number=s.step_number,
                title=s.title,
                description=s.description,
                capability=s.capability.value,
                recommended_model=s.recommended_model,
                model_display_name=s.model_display_name,
                avg_quality_score=s.avg_quality_score,
                avg_latency_seconds=s.avg_latency_seconds,
                estimated_cost_usd=s.estimated_cost_usd,
                alternatives=s.alternatives,
                input_description=s.input_description,
                output_description=s.output_description,
            )
            for s in wf.steps
        ],
        step_benchmarks=[
            StepBenchmarkSchema(
                step_number=sb.step_number,
                step_title=sb.step_title,
                capability=sb.capability.value,
                candidates_tested=sb.candidates_tested,
                recommended_model=sb.recommended_model,
                recommendation_reason=sb.recommendation_reason,
                rankings=[
                    StepModelRankingSchema(
                        model_id=r.model_id,
                        rank=r.rank,
                        avg_quality_score=r.avg_quality_score,
                        avg_latency_seconds=r.avg_latency_seconds,
                        estimated_cost_usd=r.estimated_cost_usd,
                        scores=r.scores,
                        output_sample=r.output_sample[:500],
                        error=r.error,
                    )
                    for r in sb.rankings
                ],
            )
            for sb in wf.step_benchmarks
        ],
    )
