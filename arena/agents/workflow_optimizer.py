"""WorkflowOptimizerAgent – benchmarks candidates per pipeline step and recommends the best model.

This is the orchestrator that ties the full workflow together:

1. Receives decomposed PipelineSteps from TaskDecomposerAgent.
2. For each step, queries ModelDiscoveryService for candidate models.
3. Benchmarks each candidate using ModelRunnerAgent + EvaluatorAgent.
4. Ranks candidates per step and selects the best.
5. Assembles the final WorkflowRecommendation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from statistics import mean
from typing import Any

from huggingface_hub import InferenceClient  # type: ignore[import-untyped]

from arena.config import get_settings
from arena.logging_config import get_logger
from arena.schemas import EvalResult, EvalTask, ModelOutput, TaskType
from arena.services.model_discovery import ModelDiscoveryService
from arena.workflow_schemas import (
    Capability,
    CandidateModel,
    PipelineStep,
    StepBenchmarkResult,
    StepModelRanking,
    WorkflowRecommendation,
    WorkflowStep,
)

logger = get_logger("agents.workflow_optimizer")

# Mapping from Capability to the TaskType the evaluator understands.
# Non-text capabilities are evaluated with generic text metrics.
_CAPABILITY_TO_TASKTYPE: dict[Capability, TaskType] = {
    Capability.TEXT_GENERATION: TaskType.SUMMARIZATION,
    Capability.SUMMARIZATION: TaskType.SUMMARIZATION,
    Capability.QA: TaskType.QA,
    Capability.CHAT: TaskType.SUMMARIZATION,
    Capability.CODE_GENERATION: TaskType.CODING,
    Capability.MATH_REASONING: TaskType.REASONING,
    Capability.TRANSLATION: TaskType.SUMMARIZATION,
    Capability.SENTIMENT_ANALYSIS: TaskType.QA,
    Capability.DATA_EXTRACTION: TaskType.QA,
    # Non-text capabilities – benchmarked differently
    Capability.SPEECH_TO_TEXT: TaskType.SUMMARIZATION,
    Capability.TEXT_TO_SPEECH: TaskType.SUMMARIZATION,
    Capability.IMAGE_GENERATION: TaskType.SUMMARIZATION,
    Capability.IMAGE_CLASSIFICATION: TaskType.SUMMARIZATION,
    Capability.IMAGE_TO_TEXT: TaskType.SUMMARIZATION,
    Capability.OBJECT_DETECTION: TaskType.SUMMARIZATION,
    Capability.EMBEDDING: TaskType.SUMMARIZATION,
}

# Capabilities that use chatCompletion API (text-based models)
_CHAT_CAPABILITIES = {
    Capability.TEXT_GENERATION,
    Capability.SUMMARIZATION,
    Capability.QA,
    Capability.CHAT,
    Capability.CODE_GENERATION,
    Capability.MATH_REASONING,
    Capability.TRANSLATION,
    Capability.SENTIMENT_ANALYSIS,
    Capability.DATA_EXTRACTION,
}


class WorkflowOptimizerAgent:
    """Benchmarks candidate models per step and recommends the optimal pipeline.

    Parameters
    ----------
    quality_weight:
        Weight for output quality in composite scoring (0-1).
    latency_weight:
        Weight for latency in composite scoring (0-1).
    cost_weight:
        Weight for estimated cost in composite scoring (0-1).
    max_concurrent:
        Maximum parallel API calls during benchmarking.
    """

    def __init__(
        self,
        quality_weight: float = 0.60,
        latency_weight: float = 0.25,
        cost_weight: float = 0.15,
        max_concurrent: int = 3,
        hf_token: str | None = None,
    ) -> None:
        self._quality_w = quality_weight
        self._latency_w = latency_weight
        self._cost_w = cost_weight

        settings = get_settings()
        self._token = hf_token or settings.hf_api_token
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._discovery = ModelDiscoveryService()

    # ── Public API ────────────────────────────────────────────────────

    async def optimize(
        self,
        user_request: str,
        task_analysis: str,
        steps: list[PipelineStep],
    ) -> WorkflowRecommendation:
        """Benchmark all candidates per step and build the workflow recommendation.

        Parameters
        ----------
        user_request:
            The original user request text.
        task_analysis:
            The analysis text from the decomposer.
        steps:
            The decomposed pipeline steps.

        Returns
        -------
        WorkflowRecommendation
        """
        logger.info(
            "Optimizing workflow with %d steps for: %.80s…",
            len(steps),
            user_request,
        )

        step_benchmarks: list[StepBenchmarkResult] = []
        workflow_steps: list[WorkflowStep] = []

        for step in steps:
            logger.info(
                "Step %d: %s (capability=%s)",
                step.step_number,
                step.title,
                step.capability.value,
            )

            candidates = self._discovery.get_candidates(step.capability)

            if step.capability in _CHAT_CAPABILITIES:
                # Text-based: benchmark all candidates via chat completion
                benchmark = await self._benchmark_chat_step(step, candidates)
            else:
                # Non-text: benchmark using modality-specific HF inference endpoints
                benchmark = await self._benchmark_non_chat_step(step, candidates)

            step_benchmarks.append(benchmark)

            # Build the workflow step with the recommended model
            top = benchmark.rankings[0] if benchmark.rankings else None
            alternatives = [
                r.model_id for r in benchmark.rankings[1:3]
            ]

            wf_step = WorkflowStep(
                step_number=step.step_number,
                title=step.title,
                description=step.description,
                capability=step.capability,
                recommended_model=benchmark.recommended_model,
                model_display_name=self._display_name(
                    benchmark.recommended_model, candidates
                ),
                avg_quality_score=top.avg_quality_score if top else 0.0,
                avg_latency_seconds=top.avg_latency_seconds if top else 0.0,
                estimated_cost_usd=top.estimated_cost_usd if top else 0.0,
                alternatives=alternatives,
                input_description=step.input_description,
                output_description=step.output_description,
            )
            workflow_steps.append(wf_step)

        total_cost = sum(s.estimated_cost_usd for s in workflow_steps)
        total_latency = sum(s.avg_latency_seconds for s in workflow_steps)

        recommendation = WorkflowRecommendation(
            user_request=user_request,
            task_analysis=task_analysis,
            steps=workflow_steps,
            step_benchmarks=step_benchmarks,
            total_estimated_cost_per_run=round(total_cost, 6),
            total_estimated_latency=round(total_latency, 3),
        )

        logger.info(
            "Workflow recommendation ready: %d steps, est. latency=%.2fs, cost=$%.6f",
            len(workflow_steps),
            total_latency,
            total_cost,
        )
        return recommendation

    # ── Chat-based benchmarking ───────────────────────────────────────

    async def _benchmark_chat_step(
        self,
        step: PipelineStep,
        candidates: list[CandidateModel],
    ) -> StepBenchmarkResult:
        """Benchmark chat-capable models by running the step's test prompt."""
        # Build EvalTask from the step's test prompt
        task = EvalTask(
            task_id=f"wf-{step.step_id}",
            task_type=_CAPABILITY_TO_TASKTYPE.get(
                step.capability, TaskType.SUMMARIZATION
            ),
            prompt=step.test_prompt or f"Complete this task: {step.description}",
            reference=step.reference_output,
            dataset_name="workflow_benchmark",
        )

        # Run all candidates in parallel
        coros = [
            self._run_chat_model(task, c) for c in candidates if c.is_chat_model
        ]
        raw_results: list[tuple[str, ModelOutput]] = await asyncio.gather(
            *coros, return_exceptions=True  # type: ignore[arg-type]
        )

        # Process results
        rankings: list[StepModelRanking] = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error("Benchmark exception: %s", result)
                continue

            model_id, output = result
            if output.error:
                rankings.append(
                    StepModelRanking(
                        model_id=model_id,
                        error=output.error,
                    )
                )
                continue

            # Score the output
            scores = self._score_output(task, output)
            quality = mean(scores.values()) if scores else 0.0

            rankings.append(
                StepModelRanking(
                    model_id=model_id,
                    avg_quality_score=round(quality, 4),
                    avg_latency_seconds=round(output.latency_seconds, 3),
                    estimated_cost_usd=self._estimate_cost(output),
                    scores=scores,
                    output_sample=output.output_text[:500],
                )
            )

        # Sort by composite score (higher is better)
        rankings = self._rank(rankings)

        recommended = rankings[0].model_id if rankings else "unknown"
        reason = self._build_reason(rankings, step)

        return StepBenchmarkResult(
            step_number=step.step_number,
            step_title=step.title,
            capability=step.capability,
            candidates_tested=len(rankings),
            rankings=rankings,
            recommended_model=recommended,
            recommendation_reason=reason,
        )

    async def _run_chat_model(
        self, task: EvalTask, candidate: CandidateModel
    ) -> tuple[str, ModelOutput]:
        """Run a single chat model and return (model_id, output)."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None, self._call_chat_sync, task, candidate.model_id
            )
            return candidate.model_id, output

    def _call_chat_sync(self, task: EvalTask, model_id: str) -> ModelOutput:
        """Synchronous chat completion call."""
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
            text = response.choices[0].message.content or ""

            input_tokens = (
                getattr(response.usage, "prompt_tokens", 0)
                if response.usage
                else max(len(task.prompt) // 4, 1)
            )
            output_tokens = (
                getattr(response.usage, "completion_tokens", 0)
                if response.usage
                else max(len(text) // 4, 1)
            )

            if not input_tokens:
                input_tokens = max(len(task.prompt) // 4, 1)
            if not output_tokens:
                output_tokens = max(len(text) // 4, 1)

            return ModelOutput(
                task_id=task.task_id,
                model_id=model_id,
                output_text=text,
                latency_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error("Chat model %s failed: %s", model_id, exc)
            return ModelOutput(
                task_id=task.task_id,
                model_id=model_id,
                latency_seconds=elapsed,
                error=str(exc),
            )

    # ── Non-chat step benchmarking (audio / image / embedding) ──────

    async def _benchmark_non_chat_step(
        self,
        step: PipelineStep,
        candidates: list[CandidateModel],
    ) -> StepBenchmarkResult:
        """Benchmark non-chat models using their native HF Inference endpoints.

        Supports: text_to_speech, automatic_speech_recognition, text_to_image,
        image_classification, image_to_text, object_detection,
        feature_extraction (embedding), sentence_similarity.
        """
        coros = [
            self._run_non_chat_model(step, c) for c in candidates
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        rankings: list[StepModelRanking] = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error("Non-chat benchmark exception: %s", result)
                continue
            rankings.append(result)

        # Sort and rank
        rankings = self._rank(rankings)
        recommended = rankings[0].model_id if rankings else (
            candidates[0].model_id if candidates else "unknown"
        )
        reason = self._build_reason(rankings, step)

        return StepBenchmarkResult(
            step_number=step.step_number,
            step_title=step.title,
            capability=step.capability,
            candidates_tested=len(rankings),
            rankings=rankings,
            recommended_model=recommended,
            recommendation_reason=reason,
        )

    async def _run_non_chat_model(
        self,
        step: PipelineStep,
        candidate: CandidateModel,
    ) -> StepModelRanking:
        """Run a single non-chat model via its native HF inference endpoint."""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._call_non_chat_sync, step, candidate
            )

    def _call_non_chat_sync(
        self,
        step: PipelineStep,
        candidate: CandidateModel,
    ) -> StepModelRanking:
        """Synchronous call to a non-chat HF inference endpoint via httpx."""
        model_id = candidate.model_id
        capability = step.capability
        start = time.perf_counter()

        try:
            result_info = self._dispatch_non_chat(model_id, capability, step)
            elapsed = time.perf_counter() - start

            logger.info(
                "Non-chat model %s responded in %.2fs – %s",
                model_id, elapsed, result_info.get("summary", "OK"),
            )

            quality = result_info.get("quality", 0.7)

            return StepModelRanking(
                model_id=model_id,
                avg_quality_score=round(quality, 4),
                avg_latency_seconds=round(elapsed, 3),
                estimated_cost_usd=round(elapsed * 0.0001, 6),
                scores=result_info.get("scores", {}),
                output_sample=result_info.get("summary", "Model responded successfully."),
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error("Non-chat model %s failed: %s", model_id, exc)
            return StepModelRanking(
                model_id=model_id,
                avg_quality_score=0.0,
                avg_latency_seconds=round(elapsed, 3),
                estimated_cost_usd=0.0,
                error=str(exc),
            )

    def _hf_inference_post(self, model_id: str, payload: dict | bytes, is_binary_input: bool = False, content_type: str = "audio/wav") -> httpx.Response:
        """Make a raw POST to the HF Inference API for a model."""
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        if is_binary_input:
            headers["Content-Type"] = content_type
            return httpx.post(url, content=payload, headers=headers, timeout=60)
        else:
            return httpx.post(url, json=payload, headers=headers, timeout=60)

    def _dispatch_non_chat(
        self,
        model_id: str,
        capability: Capability,
        step: PipelineStep,
    ) -> dict:
        """Route to the correct HF Inference API endpoint based on capability."""

        if capability == Capability.TEXT_TO_SPEECH:
            test_text = step.test_prompt or "Hello, this is a benchmark test for text-to-speech synthesis quality."
            resp = self._hf_inference_post(model_id, {"inputs": test_text})
            resp.raise_for_status()
            audio_bytes = resp.content
            size = len(audio_bytes) if audio_bytes else 0
            quality = min(1.0, size / 5000)
            return {
                "summary": f"Generated {size:,} bytes of audio",
                "quality": max(0.3, quality),
                "scores": {"audio_size_bytes": float(size), "output_valid": 1.0},
            }

        elif capability == Capability.SPEECH_TO_TEXT:
            # Generate test audio via TTS, then transcribe
            test_text = "The quick brown fox jumps over the lazy dog."
            try:
                tts_resp = self._hf_inference_post("facebook/mms-tts-eng", {"inputs": test_text})
                tts_resp.raise_for_status()
                tts_audio = tts_resp.content
            except Exception:
                tts_audio = self._minimal_wav()

            resp = self._hf_inference_post(model_id, tts_audio, is_binary_input=True)
            resp.raise_for_status()
            result = resp.json()
            transcript_text = ""
            if isinstance(result, dict):
                transcript_text = result.get("text", "")
            elif isinstance(result, list) and result:
                transcript_text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            else:
                transcript_text = str(result)

            expected_words = set(test_text.lower().split())
            got_words = set(transcript_text.lower().split())
            overlap = len(expected_words & got_words) / len(expected_words) if expected_words else 0
            quality = max(0.1, overlap)

            return {
                "summary": f'Transcribed: "{transcript_text[:120]}"',
                "quality": round(quality, 4),
                "scores": {"word_overlap": round(overlap, 4), "transcript_length": float(len(transcript_text))},
            }

        elif capability == Capability.TEXT_TO_IMAGE:
            prompt = step.test_prompt or "A beautiful sunset over mountains, digital art"
            resp = self._hf_inference_post(model_id, {"inputs": prompt})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            size = len(resp.content)
            is_image = "image" in content_type or size > 1000
            quality = 0.8 if is_image and size > 5000 else 0.4
            return {
                "summary": f"Generated image: {size:,} bytes ({content_type})",
                "quality": quality,
                "scores": {"image_size_bytes": float(size), "output_valid": 1.0 if is_image else 0.0},
            }

        elif capability == Capability.IMAGE_CLASSIFICATION:
            test_image = self._test_image()
            resp = self._hf_inference_post(model_id, test_image, is_binary_input=True, content_type="image/png")
            resp.raise_for_status()
            results = resp.json()
            top_label = ""
            top_score = 0.0
            if results and isinstance(results, list) and len(results) > 0:
                top_label = results[0].get("label", "")
                top_score = results[0].get("score", 0.0)
            quality = min(1.0, top_score) if top_score > 0 else 0.5
            return {
                "summary": f'Top label: "{top_label}" (score={top_score:.3f})',
                "quality": round(quality, 4),
                "scores": {"top_score": round(top_score, 4), "num_labels": float(len(results) if results else 0)},
            }

        elif capability == Capability.IMAGE_TO_TEXT:
            test_image = self._test_image()
            resp = self._hf_inference_post(model_id, test_image, is_binary_input=True, content_type="image/png")
            resp.raise_for_status()
            result = resp.json()
            text = ""
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
            elif isinstance(result, dict):
                text = result.get("generated_text", "")
            else:
                text = str(result)
            quality = 0.7 if len(text) > 5 else 0.2
            return {
                "summary": f'Caption: "{text[:150]}"',
                "quality": quality,
                "scores": {"caption_length": float(len(text)), "output_valid": 1.0 if text else 0.0},
            }

        elif capability == Capability.OBJECT_DETECTION:
            test_image = self._test_image()
            resp = self._hf_inference_post(model_id, test_image, is_binary_input=True, content_type="image/png")
            resp.raise_for_status()
            results = resp.json()
            num_objects = len(results) if isinstance(results, list) else 0
            quality = 0.7 if num_objects > 0 else 0.3
            return {
                "summary": f"Detected {num_objects} objects",
                "quality": quality,
                "scores": {"num_objects": float(num_objects), "output_valid": 1.0},
            }

        elif capability == Capability.EMBEDDING:
            test_text = step.test_prompt or "This is a test sentence for embedding."
            resp = self._hf_inference_post(model_id, {"inputs": test_text})
            resp.raise_for_status()
            result = resp.json()
            dim = 0
            if isinstance(result, list):
                if result and isinstance(result[0], list):
                    dim = len(result[0])
                else:
                    dim = len(result)
            quality = 0.8 if dim > 0 else 0.1
            return {
                "summary": f"Embedding dimension: {dim}",
                "quality": quality,
                "scores": {"embedding_dim": float(dim), "output_valid": 1.0 if dim > 0 else 0.0},
            }

        else:
            # Unknown non-chat capability — attempt a basic check
            return {
                "summary": f"Unknown capability {capability.value} — skipping live test",
                "quality": 0.0,
                "scores": {},
            }

    @staticmethod
    def _minimal_wav() -> bytes:
        """Return a minimal valid WAV file (~1 second of silence) for STT testing."""
        import struct
        sample_rate = 16000
        num_samples = sample_rate  # 1 second
        data_size = num_samples * 2  # 16-bit mono
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
            b"data", data_size,
        )
        return header + b"\x00" * data_size

    @staticmethod
    def _test_image() -> bytes:
        """Generate a small test PNG (32x32 gradient) for image model testing."""
        import io
        import struct
        import zlib

        width, height = 32, 32
        # Build raw pixel data (RGB)
        raw = b""
        for y in range(height):
            raw += b"\x00"  # filter byte
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = 128
                raw += struct.pack("BBB", r, g, b)

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        png += _chunk(b"IDAT", zlib.compress(raw))
        png += _chunk(b"IEND", b"")
        return png

    # ── Scoring ───────────────────────────────────────────────────────

    def _score_output(self, task: EvalTask, output: ModelOutput) -> dict[str, float]:
        """Score a model output using the appropriate metric suite."""
        reference = task.reference.strip()
        hypothesis = output.output_text.strip()

        if not reference:
            # No reference available — use heuristic quality indicators
            return self._heuristic_score(hypothesis, task)

        task_type = task.task_type
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
    def _heuristic_score(text: str, task: EvalTask) -> dict[str, float]:
        """When no reference is available, score based on output quality heuristics."""
        if not text:
            return {"completeness": 0.0, "coherence": 0.0}

        # Length-based completeness (reasonable output should be 20-500 chars)
        length = len(text)
        if length < 10:
            completeness = 0.1
        elif length < 50:
            completeness = 0.4
        elif length < 500:
            completeness = 0.8
        else:
            completeness = 1.0

        # Simple coherence: ratio of alphabetic words to total tokens
        tokens = text.split()
        if not tokens:
            coherence = 0.0
        else:
            alpha_tokens = sum(1 for t in tokens if t[0].isalpha())
            coherence = alpha_tokens / len(tokens)

        # Relevance: keyword overlap with the prompt
        prompt_keywords = set(task.prompt.lower().split())
        output_keywords = set(text.lower().split())
        if prompt_keywords:
            relevance = len(prompt_keywords & output_keywords) / len(prompt_keywords)
        else:
            relevance = 0.5

        return {
            "completeness": round(completeness, 4),
            "coherence": round(coherence, 4),
            "relevance": round(relevance, 4),
        }

    @staticmethod
    def _score_summarization(reference: str, hypothesis: str) -> dict[str, float]:
        """ROUGE-based scoring."""
        try:
            from rouge_score import rouge_scorer as rs  # type: ignore[import-untyped]

            scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
            }
        except Exception:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    @staticmethod
    def _score_qa(reference: str, hypothesis: str) -> dict[str, float]:
        ref_norm = reference.lower().strip()
        hyp_norm = hypothesis.lower().strip()
        exact = 1.0 if ref_norm in hyp_norm or hyp_norm in ref_norm else 0.0
        ref_tokens = set(ref_norm.split())
        hyp_tokens = set(hyp_norm.split())
        common = ref_tokens & hyp_tokens
        precision = len(common) / len(hyp_tokens) if hyp_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"exact_match": exact, "f1": round(f1, 4)}

    @staticmethod
    def _score_reasoning(reference: str, hypothesis: str) -> dict[str, float]:
        import re as _re

        ref_nums = _re.findall(r"[-+]?\d*\.?\d+", reference.replace(",", ""))
        hyp_nums = _re.findall(r"[-+]?\d*\.?\d+", hypothesis.replace(",", ""))
        ref_num = ref_nums[-1] if ref_nums else None
        hyp_num = hyp_nums[-1] if hyp_nums else None
        return {"accuracy": 1.0 if ref_num and ref_num == hyp_num else 0.0}

    @staticmethod
    def _score_coding(reference: str, hypothesis: str) -> dict[str, float]:
        try:
            from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
                SmoothingFunction,
                sentence_bleu,
            )

            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            bleu = float(
                sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    smoothing_function=SmoothingFunction().method1,
                )
            ) if ref_tokens and hyp_tokens else 0.0
        except Exception:
            bleu = 0.0

        ref_set = set(reference.split())
        hyp_set = set(hypothesis.split())
        overlap = len(ref_set & hyp_set) / len(ref_set) if ref_set else 0.0
        return {"bleu": round(bleu, 4), "token_overlap": round(overlap, 4)}

    # ── Ranking ───────────────────────────────────────────────────────

    def _rank(self, rankings: list[StepModelRanking]) -> list[StepModelRanking]:
        """Sort rankings by composite score (quality, latency, cost)."""
        # Normalise latency (lower is better)
        max_latency = max((r.avg_latency_seconds for r in rankings if not r.error), default=1.0) or 1.0
        max_cost = max((r.estimated_cost_usd for r in rankings if not r.error), default=0.0001) or 0.0001

        def composite(r: StepModelRanking) -> float:
            if r.error:
                return -1.0
            latency_norm = 1.0 - (r.avg_latency_seconds / max_latency)
            cost_norm = 1.0 - (r.estimated_cost_usd / max_cost) if max_cost > 0 else 1.0
            return (
                self._quality_w * r.avg_quality_score
                + self._latency_w * latency_norm
                + self._cost_w * cost_norm
            )

        scored = sorted(rankings, key=composite, reverse=True)
        for i, r in enumerate(scored):
            r.rank = i + 1
        return scored

    def _build_reason(
        self, rankings: list[StepModelRanking], step: PipelineStep
    ) -> str:
        """Build a human-readable recommendation reason."""
        if not rankings:
            return "No candidates were successfully benchmarked."

        top = rankings[0]
        if top.error:
            return f"All candidates failed. Top error: {top.error}"

        parts = [
            f"Recommended {top.model_id} for '{step.title}'.",
            f"Quality: {top.avg_quality_score:.3f}",
            f"Latency: {top.avg_latency_seconds:.2f}s",
        ]
        if len(rankings) > 1 and not rankings[1].error:
            runner = rankings[1]
            parts.append(
                f"Runner-up: {runner.model_id} "
                f"(quality={runner.avg_quality_score:.3f}, "
                f"latency={runner.avg_latency_seconds:.2f}s)"
            )

        return " | ".join(parts)

    @staticmethod
    def _estimate_cost(output: ModelOutput) -> float:
        """Estimate USD cost from token counts."""
        settings = get_settings()
        cost = (
            (output.input_tokens / 1000) * settings.cost_per_1k_input_tokens
            + (output.output_tokens / 1000) * settings.cost_per_1k_output_tokens
        )
        return round(cost, 6)

    @staticmethod
    def _display_name(model_id: str, candidates: list[CandidateModel]) -> str:
        for c in candidates:
            if c.model_id == model_id:
                return c.display_name
        return model_id.split("/")[-1]
