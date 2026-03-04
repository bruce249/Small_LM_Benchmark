"""Model registry – keeps track of available models and their metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arena.logging_config import get_logger
from arena.schemas import TaskType

logger = get_logger("services.model_registry")


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    model_id: str
    display_name: str = ""
    provider: str = "huggingface"
    supported_tasks: list[TaskType] = field(default_factory=lambda: list(TaskType))
    max_input_tokens: int = 4096
    max_output_tokens: int = 1024
    cost_per_1k_input: float = 0.0015
    cost_per_1k_output: float = 0.002
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.model_id.split("/")[-1]


# ── Pre-populated registry of popular open-source models ─────────────────────

_BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5-7B-Instruct",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0001,
    ),
    ModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B-Instruct",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0001,
    ),
    ModelInfo(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama-3.2-3B-Instruct",
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00005,
    ),
    ModelInfo(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        display_name="Qwen2.5-Coder-32B-Instruct",
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.0002,
    ),
]


class ModelRegistry:
    """Thread-safe in-memory registry of model metadata."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        for m in _BUILTIN_MODELS:
            self.register(m)

    # ── Public API ────────────────────────────────────────────────────

    def register(self, model: ModelInfo) -> None:
        """Add or update a model in the registry."""
        self._models[model.model_id] = model
        logger.debug("Registered model: %s", model.model_id)

    def get(self, model_id: str) -> ModelInfo | None:
        """Look up a model by its HuggingFace identifier."""
        return self._models.get(model_id)

    def get_or_default(self, model_id: str) -> ModelInfo:
        """Return known model info or construct a default entry."""
        existing = self.get(model_id)
        if existing:
            return existing
        default = ModelInfo(model_id=model_id)
        self.register(default)
        return default

    def list_models(
        self,
        task_type: TaskType | None = None,
    ) -> list[ModelInfo]:
        """Return all registered models, optionally filtered by task."""
        models = list(self._models.values())
        if task_type is not None:
            models = [m for m in models if task_type in m.supported_tasks]
        return models

    def list_model_ids(self, task_type: TaskType | None = None) -> list[str]:
        """Convenience: return just the model IDs."""
        return [m.model_id for m in self.list_models(task_type)]
