"""TaskDecomposerAgent – uses an LLM to break a user request into pipeline steps.

This is the "brain" that understands what the user wants to build and decomposes
it into ordered steps, each tagged with a capability type so the right class of
models can be benchmarked.

Example
-------
User: "I want to build a podcast summariser that takes audio, transcribes it,
       summarises the text, and generates a short audio summary."

Decomposed into:
  Step 1: Speech-to-Text     (capability: speech_to_text)
  Step 2: Text Summarization (capability: summarization)
  Step 3: Text-to-Speech     (capability: text_to_speech)
"""

from __future__ import annotations

import json
import re
from typing import Any

from huggingface_hub import InferenceClient  # type: ignore[import-untyped]

from arena.config import get_settings
from arena.logging_config import get_logger
from arena.workflow_schemas import Capability, PipelineStep

logger = get_logger("agents.task_decomposer")

# All valid capability values for the LLM to choose from
_CAPABILITY_VALUES = [c.value for c in Capability]

_DECOMPOSITION_SYSTEM_PROMPT = """\
You are an expert AI systems architect. Your job is to analyse a user's project \
request and decompose it into a sequence of discrete pipeline steps.

For each step you MUST assign exactly ONE capability from this list:
{capabilities}

RULES:
1. Output ONLY valid JSON — no markdown fences, no commentary.
2. The JSON must be an object with two keys:
   - "analysis": A 2-3 sentence summary of what the user wants to build.
   - "steps": An array of step objects.
3. Each step object has these keys:
   - "step_number": integer starting at 1
   - "title": short title (3-6 words)
   - "description": one sentence explaining what this step does
   - "capability": one of the capability values listed above
   - "input_description": what this step receives
   - "output_description": what this step produces
   - "test_prompt": a concrete example prompt that could be sent to a model \
to test this capability (be specific and realistic)
   - "depends_on": list of step_numbers this step depends on (empty if first)
4. Order steps logically (data flows from earlier to later).
5. Keep it to 2-7 steps. Don't over-decompose.
6. For non-text modalities (voice, image), still create a test_prompt that \
describes what the model should do, even if the actual API call would differ.
"""

_DECOMPOSITION_USER_PROMPT = """\
User request: {user_request}

Decompose this into pipeline steps with capabilities. Return ONLY the JSON.
"""


class TaskDecomposerAgent:
    """Analyses a user request and produces a list of :class:`PipelineStep`.

    Uses a base HuggingFace chat model to understand the user's intent and
    map it to a structured pipeline with capability tags.
    """

    def __init__(
        self,
        decomposer_model: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_token: str | None = None,
    ) -> None:
        self._model = decomposer_model
        settings = get_settings()
        self._token = hf_token or settings.hf_api_token

    async def decompose(self, user_request: str) -> tuple[str, list[PipelineStep]]:
        """Break a user request into pipeline steps.

        Parameters
        ----------
        user_request:
            The natural-language description of what the user wants to build.

        Returns
        -------
        tuple[str, list[PipelineStep]]
            A tuple of (task_analysis_text, list_of_steps).
        """
        logger.info("Decomposing user request: %.100s…", user_request)

        system_prompt = _DECOMPOSITION_SYSTEM_PROMPT.format(
            capabilities=", ".join(_CAPABILITY_VALUES)
        )
        user_prompt = _DECOMPOSITION_USER_PROMPT.format(user_request=user_request)

        raw_response = self._call_llm(system_prompt, user_prompt)
        logger.debug("Raw decomposition response: %s", raw_response[:500])

        analysis, steps = self._parse_response(raw_response, user_request)
        logger.info(
            "Decomposed into %d steps: %s",
            len(steps),
            [(s.step_number, s.title, s.capability.value) for s in steps],
        )
        return analysis, steps

    # ── LLM call ──────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous call to the decomposition model."""
        client = InferenceClient(token=self._token if self._token else None)
        try:
            response = client.chat_completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Decomposer LLM call failed: %s", exc)
            raise RuntimeError(f"Failed to decompose task: {exc}") from exc

    # ── Response parsing ──────────────────────────────────────────────

    def _parse_response(
        self, raw: str, user_request: str
    ) -> tuple[str, list[PipelineStep]]:
        """Parse the LLM JSON response into PipelineStep objects."""
        # Strip markdown code fences if the LLM wrapped the JSON
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON, attempting extraction: %s", exc)
            # Try to find JSON object in the response
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse decomposition response: {raw[:300]}")

        analysis = data.get("analysis", f"Analysis of: {user_request}")
        raw_steps = data.get("steps", [])

        steps: list[PipelineStep] = []
        for s in raw_steps:
            capability = self._resolve_capability(s.get("capability", "text_generation"))
            step = PipelineStep(
                step_number=s.get("step_number", len(steps) + 1),
                title=s.get("title", "Untitled Step"),
                description=s.get("description", ""),
                capability=capability,
                input_description=s.get("input_description", ""),
                output_description=s.get("output_description", ""),
                test_prompt=s.get("test_prompt", ""),
                depends_on=s.get("depends_on", []),
            )
            steps.append(step)

        if not steps:
            # Fallback: single text_generation step
            steps = [
                PipelineStep(
                    step_number=1,
                    title="Process Request",
                    description=f"Handle: {user_request[:100]}",
                    capability=Capability.TEXT_GENERATION,
                    test_prompt=user_request,
                )
            ]

        return analysis, steps

    @staticmethod
    def _resolve_capability(raw: str) -> Capability:
        """Map a raw capability string to the enum, with fuzzy matching."""
        raw_lower = raw.lower().strip()

        # Direct match
        try:
            return Capability(raw_lower)
        except ValueError:
            pass

        # Fuzzy keyword matching
        keyword_map: dict[str, Capability] = {
            "text": Capability.TEXT_GENERATION,
            "chat": Capability.CHAT,
            "summar": Capability.SUMMARIZATION,
            "qa": Capability.QA,
            "question": Capability.QA,
            "code": Capability.CODE_GENERATION,
            "program": Capability.CODE_GENERATION,
            "math": Capability.MATH_REASONING,
            "reason": Capability.MATH_REASONING,
            "translat": Capability.TRANSLATION,
            "speech_to_text": Capability.SPEECH_TO_TEXT,
            "stt": Capability.SPEECH_TO_TEXT,
            "asr": Capability.SPEECH_TO_TEXT,
            "transcrib": Capability.SPEECH_TO_TEXT,
            "text_to_speech": Capability.TEXT_TO_SPEECH,
            "tts": Capability.TEXT_TO_SPEECH,
            "voice_gen": Capability.TEXT_TO_SPEECH,
            "image_gen": Capability.IMAGE_GENERATION,
            "text_to_image": Capability.IMAGE_GENERATION,
            "image_class": Capability.IMAGE_CLASSIFICATION,
            "image_to_text": Capability.IMAGE_TO_TEXT,
            "caption": Capability.IMAGE_TO_TEXT,
            "object_detect": Capability.OBJECT_DETECTION,
            "detect": Capability.OBJECT_DETECTION,
            "embed": Capability.EMBEDDING,
            "sentiment": Capability.SENTIMENT_ANALYSIS,
            "extract": Capability.DATA_EXTRACTION,
        }

        for keyword, cap in keyword_map.items():
            if keyword in raw_lower:
                return cap

        logger.warning("Unknown capability '%s', defaulting to text_generation", raw)
        return Capability.TEXT_GENERATION
