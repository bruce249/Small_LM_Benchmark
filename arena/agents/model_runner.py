"""ModelRunnerAgent – executes prompts against HuggingFace models in parallel.

Uses the official ``huggingface_hub.InferenceClient`` with the chat-completions
API, which routes through the current HuggingFace inference infrastructure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from huggingface_hub import InferenceClient  # type: ignore[import-untyped]

from arena.config import get_settings
from arena.logging_config import get_logger
from arena.schemas import EvalTask, ModelOutput

logger = get_logger("agents.model_runner")


class ModelRunnerAgent:
    """Sends evaluation prompts to HuggingFace Inference API endpoints.

    Responsibilities
    ----------------
    * Call the HF Inference API (chat completions) for each ``(model, task)`` pair.
    * Measure wall-clock latency per request.
    * Capture token usage when the API returns it.
    * Execute multiple models in parallel via :mod:`asyncio`.
    """

    def __init__(self, max_concurrent: int | None = None, hf_token: str | None = None) -> None:
        settings = get_settings()
        self._token = hf_token or settings.hf_api_token
        self._timeout = settings.default_timeout_seconds
        self._semaphore = asyncio.Semaphore(
            max_concurrent or settings.max_concurrent_models
        )

    # ── Public API ────────────────────────────────────────────────────

    async def run_all(
        self,
        tasks: list[EvalTask],
        model_ids: list[str],
    ) -> list[ModelOutput]:
        """Run every task against every model, returning flattened outputs.

        Execution is bounded by a semaphore so we do not overwhelm the
        upstream API with too many concurrent requests.
        """
        logger.info(
            "Running %d tasks × %d models (%d total calls)",
            len(tasks),
            len(model_ids),
            len(tasks) * len(model_ids),
        )

        coros = [
            self._run_single(task, model_id)
            for model_id in model_ids
            for task in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Convert any unexpected exceptions into error ModelOutputs
        final: list[ModelOutput] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Unexpected task exception: %s", result)
                final.append(ModelOutput(error=str(result)))
            else:
                final.append(result)

        logger.info("Completed %d model calls", len(final))
        return final

    # ── Internal ──────────────────────────────────────────────────────

    async def _run_single(
        self,
        task: EvalTask,
        model_id: str,
    ) -> ModelOutput:
        """Call the HF Inference API for a single (task, model) pair."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._call_model_sync, task, model_id
            )

    def _call_model_sync(self, task: EvalTask, model_id: str) -> ModelOutput:
        """Synchronous HF API call (runs in executor thread)."""
        client = InferenceClient(token=self._token if self._token else None)

        start = time.perf_counter()
        try:
            response = client.chat_completion(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Follow instructions precisely and be concise."
                        ),
                    },
                    {"role": "user", "content": task.prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            elapsed = time.perf_counter() - start

            output_text = response.choices[0].message.content or ""
            input_tokens = (
                getattr(response.usage, "prompt_tokens", 0)
                if response.usage else 0
            )
            output_tokens = (
                getattr(response.usage, "completion_tokens", 0)
                if response.usage else 0
            )

            # Fallback token estimation if API doesn't return usage
            if not input_tokens:
                input_tokens = max(len(task.prompt) // 4, 1)
            if not output_tokens:
                output_tokens = max(len(output_text) // 4, 1)

            logger.debug(
                "Model %s responded in %.2fs (%d tokens)",
                model_id, elapsed, output_tokens,
            )

            return ModelOutput(
                task_id=task.task_id,
                model_id=model_id,
                output_text=output_text,
                latency_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "Error calling model %s: %s", model_id, exc, exc_info=True
            )
            return ModelOutput(
                task_id=task.task_id,
                model_id=model_id,
                latency_seconds=elapsed,
                error=str(exc),
            )
