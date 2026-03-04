"""Celery tasks – wrappers around the ExperimentManager pipeline.

Each task runs synchronously inside a Celery worker process but
internally uses ``asyncio.run()`` to drive the async pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from arena.agents.report_agent import ReportAgent
from arena.experiments.experiment_manager import ExperimentManager
from arena.logging_config import get_logger, setup_logging
from arena.schemas import TaskType
from arena.worker import celery_app

logger = get_logger("worker.tasks")

setup_logging()


@celery_app.task(
    name="arena.run_experiment",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def run_experiment_task(
    self,
    task_type: str,
    model_ids: list[str],
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "test",
    max_samples: int = 50,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Celery entry-point: run a full evaluation experiment.

    Parameters match those of :meth:`ExperimentManager.run` but are
    JSON-serialisable primitives.

    Returns
    -------
    dict
        JSON-safe report dict.
    """
    logger.info(
        "Celery task %s: running experiment %s",
        self.request.id,
        experiment_id or "(auto)",
    )

    try:
        tt = TaskType(task_type)
    except ValueError:
        return {"error": f"Unknown task_type: {task_type}"}

    manager = ExperimentManager()

    try:
        report = asyncio.run(
            manager.run(
                task_type=tt,
                model_ids=model_ids or None,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_samples=max_samples,
                experiment_id=experiment_id,
            )
        )
        return ReportAgent.report_to_dict(report)
    except Exception as exc:
        logger.error("Celery task %s failed: %s", self.request.id, exc, exc_info=True)
        self.retry(exc=exc)
        return {"error": str(exc)}  # unreachable but keeps mypy happy
