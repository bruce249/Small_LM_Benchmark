"""EvaluatorAgent – scores model outputs using automated metrics."""

from __future__ import annotations

import re
from typing import Any

from arena.config import get_settings
from arena.logging_config import get_logger
from arena.schemas import EvalResult, EvalTask, ModelOutput, TaskType
from arena.services.model_registry import ModelRegistry

logger = get_logger("agents.evaluator")

# Lazy-loaded metric modules to avoid import overhead at startup
_rouge_scorer = None
_bleu_scorer = None


def _get_rouge():
    """Lazy-load rouge-score."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs  # type: ignore[import-untyped]
        _rouge_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU using nltk."""
    try:
        from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
            SmoothingFunction,
            sentence_bleu,
        )
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 0.0


def _extract_numeric_answer(text: str) -> str | None:
    """Extract the last number from a reasoning answer."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return numbers[-1] if numbers else None


class EvaluatorAgent:
    """Scores model outputs and computes latency / cost metrics.

    Responsibilities
    ----------------
    * Evaluate summarization with ROUGE.
    * Evaluate QA with exact-match accuracy.
    * Evaluate reasoning with numeric answer accuracy.
    * Compute estimated cost using token counts and model pricing.
    * Return :class:`EvalResult` per (task, model) pair.
    """

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self._registry = model_registry or ModelRegistry()
        self._settings = get_settings()

    async def evaluate(
        self,
        tasks: list[EvalTask],
        outputs: list[ModelOutput],
    ) -> list[EvalResult]:
        """Evaluate every model output against its corresponding task.

        Parameters
        ----------
        tasks:
            The original tasks (used for reference answers and task_type).
        outputs:
            The raw model outputs to score.

        Returns
        -------
        list[EvalResult]
        """
        task_map: dict[str, EvalTask] = {t.task_id: t for t in tasks}
        results: list[EvalResult] = []

        for output in outputs:
            task = task_map.get(output.task_id)
            if task is None:
                logger.warning("No matching task for output task_id=%s", output.task_id)
                continue

            if output.error:
                results.append(
                    EvalResult(
                        task_id=output.task_id,
                        model_id=output.model_id,
                        scores={},
                        latency_seconds=output.latency_seconds,
                        estimated_cost_usd=0.0,
                        error=output.error,
                    )
                )
                continue

            scores = self._score(task, output)
            cost = self._estimate_cost(output)

            results.append(
                EvalResult(
                    task_id=output.task_id,
                    model_id=output.model_id,
                    scores=scores,
                    latency_seconds=output.latency_seconds,
                    estimated_cost_usd=cost,
                )
            )

        logger.info("Evaluated %d outputs → %d results", len(outputs), len(results))
        return results

    # ── Scoring by task type ──────────────────────────────────────────

    def _score(self, task: EvalTask, output: ModelOutput) -> dict[str, float]:
        """Dispatch scoring to the appropriate metric suite."""
        task_type = task.task_type
        reference = task.reference.strip()
        hypothesis = output.output_text.strip()

        if task_type == TaskType.SUMMARIZATION:
            return self._score_summarization(reference, hypothesis)
        elif task_type == TaskType.QA:
            return self._score_qa(reference, hypothesis)
        elif task_type == TaskType.REASONING:
            return self._score_reasoning(reference, hypothesis)
        elif task_type == TaskType.CODING:
            return self._score_coding(reference, hypothesis)
        else:
            return self._score_summarization(reference, hypothesis)

    @staticmethod
    def _score_summarization(reference: str, hypothesis: str) -> dict[str, float]:
        """ROUGE-based scoring for summarization."""
        if not reference or not hypothesis:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

        scorer = _get_rouge()
        rouge_scores = scorer.score(reference, hypothesis)
        bleu = _compute_bleu(reference, hypothesis)

        return {
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4),
            "bleu": round(bleu, 4),
        }

    @staticmethod
    def _score_qa(reference: str, hypothesis: str) -> dict[str, float]:
        """Exact-match and token-overlap F1 for QA."""
        ref_norm = reference.lower().strip()
        hyp_norm = hypothesis.lower().strip()

        exact_match = 1.0 if ref_norm in hyp_norm or hyp_norm in ref_norm else 0.0

        ref_tokens = set(ref_norm.split())
        hyp_tokens = set(hyp_norm.split())
        if not ref_tokens or not hyp_tokens:
            f1 = 0.0
        else:
            common = ref_tokens & hyp_tokens
            precision = len(common) / len(hyp_tokens) if hyp_tokens else 0.0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return {
            "exact_match": exact_match,
            "f1": round(f1, 4),
        }

    @staticmethod
    def _score_reasoning(reference: str, hypothesis: str) -> dict[str, float]:
        """Numeric answer accuracy for reasoning tasks."""
        ref_num = _extract_numeric_answer(reference)
        hyp_num = _extract_numeric_answer(hypothesis)

        if ref_num is None:
            return {"accuracy": 0.0}

        accuracy = 1.0 if ref_num == hyp_num else 0.0
        return {"accuracy": accuracy}

    @staticmethod
    def _score_coding(reference: str, hypothesis: str) -> dict[str, float]:
        """Simple heuristic for code similarity (token overlap + BLEU)."""
        bleu = _compute_bleu(reference, hypothesis)

        ref_tokens = set(reference.split())
        hyp_tokens = set(hypothesis.split())
        if not ref_tokens:
            overlap = 0.0
        else:
            overlap = len(ref_tokens & hyp_tokens) / len(ref_tokens)

        return {
            "bleu": round(bleu, 4),
            "token_overlap": round(overlap, 4),
        }

    # ── Cost estimation ───────────────────────────────────────────────

    def _estimate_cost(self, output: ModelOutput) -> float:
        """Estimate USD cost based on token counts and model pricing."""
        model_info = self._registry.get(output.model_id)
        if model_info:
            input_rate = model_info.cost_per_1k_input
            output_rate = model_info.cost_per_1k_output
        else:
            input_rate = self._settings.cost_per_1k_input_tokens
            output_rate = self._settings.cost_per_1k_output_tokens

        cost = (
            (output.input_tokens / 1000) * input_rate
            + (output.output_tokens / 1000) * output_rate
        )
        return round(cost, 6)
