"""ExperimentManager – orchestrates the full evaluation pipeline.

This is the central coordinator that wires together all four agents:
  TaskGeneratorAgent → ModelRunnerAgent → EvaluatorAgent → ReportAgent

It also persists experiment state to PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from arena.agents.evaluator import EvaluatorAgent
from arena.agents.model_runner import ModelRunnerAgent
from arena.agents.report_agent import ReportAgent
from arena.agents.task_generator import TaskGeneratorAgent
from arena.logging_config import get_logger
from arena.schemas import ExperimentReport, ExperimentStatus, TaskType
from arena.services.dataset_loader import DatasetLoader
from arena.services.model_registry import ModelRegistry

logger = get_logger("experiments.manager")


class ExperimentManager:
    """Stateless orchestrator that drives end-to-end evaluations.

    Usage
    -----
    >>> mgr = ExperimentManager()
    >>> report = await mgr.run(
    ...     task_type=TaskType.SUMMARIZATION,
    ...     model_ids=["mistralai/Mistral-7B-Instruct-v0.3", ...],
    ...     max_samples=20,
    ... )
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        model_registry: ModelRegistry | None = None,
        task_generator: TaskGeneratorAgent | None = None,
        model_runner: ModelRunnerAgent | None = None,
        evaluator: EvaluatorAgent | None = None,
        reporter: ReportAgent | None = None,
        hf_token: str | None = None,
    ) -> None:
        self._loader = dataset_loader or DatasetLoader()
        self._registry = model_registry or ModelRegistry()
        self._task_gen = task_generator or TaskGeneratorAgent(self._loader)
        self._runner = model_runner or ModelRunnerAgent(hf_token=hf_token)
        self._evaluator = evaluator or EvaluatorAgent(self._registry)
        self._reporter = reporter or ReportAgent()

    async def run(
        self,
        task_type: TaskType = TaskType.SUMMARIZATION,
        model_ids: list[str] | None = None,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        split: str = "test",
        max_samples: int = 50,
        experiment_id: str | None = None,
    ) -> ExperimentReport:
        """Execute the full evaluation pipeline.

        Parameters
        ----------
        task_type:
            Type of evaluation task.
        model_ids:
            HuggingFace model identifiers to benchmark.
            Defaults to all models in the registry.
        dataset_name:
            HuggingFace dataset identifier (``None`` → default for task_type).
        dataset_config:
            Optional dataset config name.
        split:
            Which dataset split to use.
        max_samples:
            Max number of evaluation prompts.
        experiment_id:
            Optional pre-assigned experiment ID.

        Returns
        -------
        ExperimentReport
            The full report with leaderboard and detailed results.
        """
        exp_id = experiment_id or str(uuid.uuid4())
        model_ids = model_ids or self._registry.list_model_ids(task_type)

        logger.info(
            "┌─ Experiment %s ─────────────────────────────",
            exp_id,
        )
        logger.info("│  Task type  : %s", task_type.value)
        logger.info("│  Models     : %s", ", ".join(model_ids))
        logger.info("│  Dataset    : %s", dataset_name or "(default)")
        logger.info("│  Max samples: %d", max_samples)
        logger.info("└────────────────────────────────────────────")

        try:
            # Step 1 – Generate tasks
            logger.info("[1/4] Generating evaluation tasks …")
            tasks = await self._task_gen.generate(
                task_type=task_type,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_samples=max_samples,
            )

            # Step 2 – Run models in parallel
            logger.info("[2/4] Running %d models on %d tasks …", len(model_ids), len(tasks))
            outputs = await self._runner.run_all(tasks, model_ids)

            # Step 3 – Evaluate outputs
            logger.info("[3/4] Evaluating outputs …")
            results = await self._evaluator.evaluate(tasks, outputs)

            # Step 4 – Generate report
            logger.info("[4/4] Generating report …")
            report = await self._reporter.generate_report(
                experiment_id=exp_id,
                task_type=task_type,
                dataset_name=dataset_name or "(default)",
                model_ids=model_ids,
                results=results,
            )

            logger.info("✓ Experiment %s completed successfully", exp_id)
            return report

        except Exception as exc:
            logger.error("✗ Experiment %s failed: %s", exp_id, exc, exc_info=True)
            return ExperimentReport(
                experiment_id=exp_id,
                task_type=task_type,
                dataset_name=dataset_name or "(default)",
                models=model_ids,
                status=ExperimentStatus.FAILED,
                error=str(exc),
            )

    async def run_and_persist(
        self,
        db_session: Any,
        **kwargs: Any,
    ) -> ExperimentReport:
        """Run experiment and save the results to PostgreSQL.

        Parameters are identical to :meth:`run`, plus *db_session*.
        """
        from arena.db.models import EvalResultRecord, ExperimentRecord

        # Create DB record
        exp_id = kwargs.get("experiment_id") or str(uuid.uuid4())
        kwargs["experiment_id"] = exp_id

        record = ExperimentRecord(
            id=exp_id,
            task_type=kwargs.get("task_type", TaskType.SUMMARIZATION).value,
            dataset_name=kwargs.get("dataset_name", "(default)"),
            models=kwargs.get("model_ids", []),
            status="running",
            config={
                "max_samples": kwargs.get("max_samples", 50),
                "split": kwargs.get("split", "test"),
            },
        )
        db_session.add(record)
        await db_session.flush()

        # Run
        report = await self.run(**kwargs)

        # Update DB record
        record.status = report.status.value
        record.leaderboard = (
            ReportAgent.report_to_dict(report).get("leaderboard") if report.leaderboard else None
        )
        record.error = report.error
        record.completed_at = report.completed_at or datetime.utcnow()

        # Persist detailed results
        for r in report.detailed_results:
            db_session.add(
                EvalResultRecord(
                    experiment_id=exp_id,
                    task_id=r.task_id,
                    model_id=r.model_id,
                    scores=r.scores,
                    latency_seconds=r.latency_seconds,
                    estimated_cost_usd=r.estimated_cost_usd,
                    error=r.error,
                )
            )

        await db_session.commit()
        logger.info("Experiment %s persisted to database", exp_id)
        return report
