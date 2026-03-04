"""Dataset loading service – wraps HuggingFace Datasets."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from arena.logging_config import get_logger
from arena.schemas import TaskType

logger = get_logger("services.dataset_loader")

# ── Default dataset mappings per task type ────────────────────────────────────

DEFAULT_DATASETS: dict[TaskType, dict[str, str]] = {
    TaskType.SUMMARIZATION: {
        "name": "cnn_dailymail",
        "config": "3.0.0",
        "input_col": "article",
        "target_col": "highlights",
    },
    TaskType.QA: {
        "name": "squad",
        "config": None,  # type: ignore[dict-item]
        "input_col": "question",
        "target_col": "answers",
        "context_col": "context",
    },
    TaskType.CODING: {
        "name": "openai_humaneval",
        "config": None,  # type: ignore[dict-item]
        "input_col": "prompt",
        "target_col": "canonical_solution",
    },
    TaskType.REASONING: {
        "name": "gsm8k",
        "config": "main",
        "input_col": "question",
        "target_col": "answer",
    },
}


class DatasetLoader:
    """Loads and caches HuggingFace datasets, returning prompt/reference pairs."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def load(
        self,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str = "test",
        task_type: TaskType = TaskType.SUMMARIZATION,
        max_samples: int = 50,
    ) -> list[dict[str, str]]:
        """Return a list of ``{"input": ..., "reference": ...}`` dicts.

        Parameters
        ----------
        dataset_name:
            HuggingFace dataset identifier.  Falls back to the default
            dataset for the given *task_type* if ``None``.
        config:
            Optional dataset configuration (e.g. ``"3.0.0"``).
        split:
            Which split to load (``"test"``, ``"validation"``, …).
        task_type:
            Used to resolve defaults and column mappings.
        max_samples:
            Cap on the number of samples to return.
        """
        defaults = DEFAULT_DATASETS.get(task_type, DEFAULT_DATASETS[TaskType.SUMMARIZATION])
        dataset_name = dataset_name or defaults["name"]
        config = config or defaults.get("config")
        input_col = defaults["input_col"]
        target_col = defaults["target_col"]
        context_col = defaults.get("context_col")

        cache_key = f"{dataset_name}:{config}:{split}"
        logger.info("Loading dataset %s [split=%s, max=%d]", cache_key, split, max_samples)

        if cache_key not in self._cache:
            try:
                ds = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
            except Exception:
                # Some datasets only have 'train' or 'validation'
                logger.warning("Split '%s' not found, falling back to 'train'", split)
                ds = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
            self._cache[cache_key] = ds

        ds = self._cache[cache_key]
        samples: list[dict[str, str]] = []

        for idx, row in enumerate(ds):
            if idx >= max_samples:
                break

            input_text = str(row.get(input_col, ""))
            if context_col and context_col in row:
                input_text = f"Context: {row[context_col]}\n\nQuestion: {input_text}"

            reference = row.get(target_col, "")
            if isinstance(reference, dict):
                # SQuAD-style answers: {"text": [...], "answer_start": [...]}
                reference = reference.get("text", [""])[0] if reference.get("text") else ""
            elif isinstance(reference, list):
                reference = reference[0] if reference else ""

            samples.append({"input": input_text, "reference": str(reference)})

        logger.info("Loaded %d samples from %s", len(samples), dataset_name)
        return samples
