"""TaskGeneratorAgent – loads datasets and creates evaluation tasks."""

from __future__ import annotations

from arena.logging_config import get_logger
from arena.schemas import EvalTask, TaskType
from arena.services.dataset_loader import DatasetLoader
from arena.services.prompt_templates import build_prompt

logger = get_logger("agents.task_generator")


class TaskGeneratorAgent:
    """Generates a list of :class:`EvalTask` objects from a HuggingFace dataset.

    Responsibilities
    ----------------
    * Load the dataset (via :class:`DatasetLoader`).
    * Apply the appropriate prompt template for the task type.
    * Return a batch of ready-to-run :class:`EvalTask` instances.
    """

    def __init__(self, dataset_loader: DatasetLoader | None = None) -> None:
        self._loader = dataset_loader or DatasetLoader()

    async def generate(
        self,
        task_type: TaskType,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        split: str = "test",
        max_samples: int = 50,
    ) -> list[EvalTask]:
        """Load data and produce prompted evaluation tasks.

        Parameters
        ----------
        task_type:
            Category of evaluation (summarization, QA, …).
        dataset_name:
            HuggingFace dataset ID.  ``None`` → use the default for *task_type*.
        dataset_config:
            Optional dataset configuration name.
        split:
            Dataset split to load.
        max_samples:
            Maximum number of evaluation tasks to produce.

        Returns
        -------
        list[EvalTask]
            A list of tasks ready for :class:`ModelRunnerAgent`.
        """
        logger.info(
            "Generating tasks: type=%s dataset=%s split=%s max=%d",
            task_type.value,
            dataset_name or "(default)",
            split,
            max_samples,
        )

        samples = self._loader.load(
            dataset_name=dataset_name,
            config=dataset_config,
            split=split,
            task_type=task_type,
            max_samples=max_samples,
        )

        tasks: list[EvalTask] = []
        for idx, sample in enumerate(samples):
            prompt = build_prompt(task_type, sample["input"])
            task = EvalTask(
                task_type=task_type,
                prompt=prompt,
                reference=sample["reference"],
                dataset_name=dataset_name or "(default)",
                dataset_split=split,
                sample_index=idx,
            )
            tasks.append(task)

        logger.info("Generated %d evaluation tasks", len(tasks))
        return tasks
