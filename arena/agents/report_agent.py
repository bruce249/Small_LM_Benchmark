"""ReportAgent – aggregates evaluation results into a leaderboard."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

from arena.logging_config import get_logger
from arena.schemas import (
    EvalResult,
    ExperimentReport,
    ExperimentStatus,
    LeaderboardEntry,
    TaskType,
)

logger = get_logger("agents.report_agent")


class ReportAgent:
    """Produces the final :class:`ExperimentReport` with a ranked leaderboard.

    Responsibilities
    ----------------
    * Group results by model.
    * Compute aggregate quality, latency, and cost metrics.
    * Rank models by a composite score.
    * Return a structured, JSON-serialisable report.
    """

    def __init__(self, quality_weight: float = 0.6, latency_weight: float = 0.25, cost_weight: float = 0.15) -> None:
        self._quality_w = quality_weight
        self._latency_w = latency_weight
        self._cost_w = cost_weight

    async def generate_report(
        self,
        experiment_id: str,
        task_type: TaskType,
        dataset_name: str,
        model_ids: list[str],
        results: list[EvalResult],
    ) -> ExperimentReport:
        """Aggregate evaluation results and produce a ranked leaderboard.

        Parameters
        ----------
        experiment_id:
            Unique identifier for the experiment.
        task_type:
            The evaluation task category.
        dataset_name:
            Name of the dataset used.
        model_ids:
            List of model identifiers.
        results:
            Scored evaluation results from the :class:`EvaluatorAgent`.

        Returns
        -------
        ExperimentReport
        """
        logger.info("Generating report for experiment %s", experiment_id)

        # ── Group by model ────────────────────────────────────────────
        by_model: dict[str, list[EvalResult]] = defaultdict(list)
        for r in results:
            by_model[r.model_id].append(r)

        # ── Build leaderboard entries ─────────────────────────────────
        entries: list[LeaderboardEntry] = []
        for model_id in model_ids:
            model_results = by_model.get(model_id, [])
            successful = [r for r in model_results if r.error is None]

            if not successful:
                entries.append(
                    LeaderboardEntry(
                        model_id=model_id,
                        avg_quality_score=0.0,
                        avg_latency_seconds=0.0,
                        total_cost_usd=0.0,
                        num_tasks=0,
                    )
                )
                continue

            # Average every metric across tasks
            all_metric_keys: set[str] = set()
            for r in successful:
                all_metric_keys.update(r.scores.keys())

            metric_means: dict[str, float] = {}
            for key in sorted(all_metric_keys):
                values = [r.scores[key] for r in successful if key in r.scores]
                metric_means[key] = round(mean(values), 4) if values else 0.0

            # Composite quality = mean of all metric values
            avg_quality = round(mean(metric_means.values()), 4) if metric_means else 0.0

            entries.append(
                LeaderboardEntry(
                    model_id=model_id,
                    avg_quality_score=avg_quality,
                    avg_latency_seconds=round(
                        mean(r.latency_seconds for r in successful), 4
                    ),
                    total_cost_usd=round(sum(r.estimated_cost_usd for r in successful), 6),
                    metric_breakdown=metric_means,
                    num_tasks=len(successful),
                )
            )

        # ── Rank by composite score ───────────────────────────────────
        entries = self._rank(entries)

        report = ExperimentReport(
            experiment_id=experiment_id,
            task_type=task_type,
            dataset_name=dataset_name,
            models=model_ids,
            leaderboard=entries,
            detailed_results=results,
            status=ExperimentStatus.COMPLETED,
            completed_at=datetime.utcnow(),
        )

        logger.info(
            "Report complete – #1: %s (quality=%.4f)",
            entries[0].model_id if entries else "N/A",
            entries[0].avg_quality_score if entries else 0.0,
        )
        return report

    # ── Ranking ───────────────────────────────────────────────────────

    def _rank(self, entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
        """Rank entries using a weighted composite of quality, latency, cost.

        Higher quality is better; lower latency and cost are better.
        We normalise each dimension to [0, 1] before weighting.
        """
        if not entries:
            return entries

        # Normalisation helpers
        max_quality = max(e.avg_quality_score for e in entries) or 1.0
        max_latency = max(e.avg_latency_seconds for e in entries) or 1.0
        max_cost = max(e.total_cost_usd for e in entries) or 1.0

        def composite(e: LeaderboardEntry) -> float:
            q = e.avg_quality_score / max_quality
            l = 1.0 - (e.avg_latency_seconds / max_latency)  # lower is better
            c = 1.0 - (e.total_cost_usd / max_cost)          # lower is better
            return self._quality_w * q + self._latency_w * l + self._cost_w * c

        ranked = sorted(entries, key=composite, reverse=True)
        for idx, entry in enumerate(ranked, start=1):
            entry.rank = idx

        return ranked

    # ── Serialisation helpers ─────────────────────────────────────────

    @staticmethod
    def report_to_dict(report: ExperimentReport) -> dict[str, Any]:
        """Convert a report to a plain dict for JSON serialisation."""
        return {
            "experiment_id": report.experiment_id,
            "task_type": report.task_type.value,
            "dataset_name": report.dataset_name,
            "models": report.models,
            "status": report.status.value,
            "created_at": report.created_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None,
            "leaderboard": [
                {
                    "rank": e.rank,
                    "model_id": e.model_id,
                    "avg_quality_score": e.avg_quality_score,
                    "avg_latency_seconds": e.avg_latency_seconds,
                    "total_cost_usd": e.total_cost_usd,
                    "metric_breakdown": e.metric_breakdown,
                    "num_tasks": e.num_tasks,
                }
                for e in report.leaderboard
            ],
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "scores": r.scores,
                    "latency_seconds": r.latency_seconds,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "error": r.error,
                }
                for r in report.detailed_results
            ],
        }
