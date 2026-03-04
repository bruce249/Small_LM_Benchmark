"""ModelDiscoveryService – maps capabilities to candidate HuggingFace models.

For each capability type (text, voice, image, math, code, etc.), this service
maintains a curated registry of known-good HuggingFace models that can be
benchmarked by the WorkflowOptimizer.

Candidate lists are intentionally split by capability so that we only test
models that are actually designed for a particular modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arena.logging_config import get_logger
from arena.workflow_schemas import Capability, CandidateModel

logger = get_logger("services.model_discovery")


# ── Curated candidate pools per capability ────────────────────────────────────
# These are models confirmed to work on the HuggingFace Inference API (chat
# completions or dedicated pipeline endpoints).

_CANDIDATE_POOL: dict[Capability, list[CandidateModel]] = {
    # ── Text / Chat ───────────────────────────────────────────────────
    Capability.TEXT_GENERATION: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.TEXT_GENERATION),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.TEXT_GENERATION),
        CandidateModel(model_id="meta-llama/Llama-3.2-3B-Instruct", display_name="Llama-3.2-3B", capability=Capability.TEXT_GENERATION),
    ],
    Capability.CHAT: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.CHAT),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.CHAT),
        CandidateModel(model_id="meta-llama/Llama-3.2-3B-Instruct", display_name="Llama-3.2-3B", capability=Capability.CHAT),
    ],
    Capability.SUMMARIZATION: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.SUMMARIZATION),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.SUMMARIZATION),
        CandidateModel(model_id="meta-llama/Llama-3.2-3B-Instruct", display_name="Llama-3.2-3B", capability=Capability.SUMMARIZATION),
    ],
    Capability.QA: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.QA),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.QA),
        CandidateModel(model_id="meta-llama/Llama-3.2-3B-Instruct", display_name="Llama-3.2-3B", capability=Capability.QA),
    ],

    # ── Code ──────────────────────────────────────────────────────────
    Capability.CODE_GENERATION: [
        CandidateModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", display_name="Qwen2.5-Coder-32B", capability=Capability.CODE_GENERATION),
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.CODE_GENERATION),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.CODE_GENERATION),
    ],

    # ── Math / Reasoning ──────────────────────────────────────────────
    Capability.MATH_REASONING: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.MATH_REASONING),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.MATH_REASONING),
        CandidateModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", display_name="Qwen2.5-Coder-32B", capability=Capability.MATH_REASONING),
    ],

    # ── Translation ───────────────────────────────────────────────────
    Capability.TRANSLATION: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.TRANSLATION),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.TRANSLATION),
    ],

    # ── Voice / Audio ─────────────────────────────────────────────────
    Capability.SPEECH_TO_TEXT: [
        CandidateModel(model_id="openai/whisper-large-v3", display_name="Whisper-Large-v3", capability=Capability.SPEECH_TO_TEXT, is_chat_model=False),
        CandidateModel(model_id="openai/whisper-large-v3-turbo", display_name="Whisper-Large-v3-Turbo", capability=Capability.SPEECH_TO_TEXT, is_chat_model=False),
    ],
    Capability.TEXT_TO_SPEECH: [
        CandidateModel(model_id="facebook/mms-tts-eng", display_name="MMS-TTS-English", capability=Capability.TEXT_TO_SPEECH, is_chat_model=False),
        CandidateModel(model_id="espnet/kan-bayashi_ljspeech_vits", display_name="ESPnet-VITS", capability=Capability.TEXT_TO_SPEECH, is_chat_model=False),
    ],

    # ── Image ─────────────────────────────────────────────────────────
    Capability.IMAGE_GENERATION: [
        CandidateModel(model_id="black-forest-labs/FLUX.1-schnell", display_name="FLUX.1-Schnell", capability=Capability.IMAGE_GENERATION, is_chat_model=False),
        CandidateModel(model_id="stabilityai/stable-diffusion-xl-base-1.0", display_name="SDXL-Base", capability=Capability.IMAGE_GENERATION, is_chat_model=False),
    ],
    Capability.IMAGE_CLASSIFICATION: [
        CandidateModel(model_id="google/vit-base-patch16-224", display_name="ViT-Base", capability=Capability.IMAGE_CLASSIFICATION, is_chat_model=False),
        CandidateModel(model_id="microsoft/resnet-50", display_name="ResNet-50", capability=Capability.IMAGE_CLASSIFICATION, is_chat_model=False),
    ],
    Capability.IMAGE_TO_TEXT: [
        CandidateModel(model_id="Salesforce/blip-image-captioning-large", display_name="BLIP-Caption-Large", capability=Capability.IMAGE_TO_TEXT, is_chat_model=False),
        CandidateModel(model_id="nlpconnect/vit-gpt2-image-captioning", display_name="ViT-GPT2-Caption", capability=Capability.IMAGE_TO_TEXT, is_chat_model=False),
    ],
    Capability.OBJECT_DETECTION: [
        CandidateModel(model_id="facebook/detr-resnet-50", display_name="DETR-ResNet-50", capability=Capability.OBJECT_DETECTION, is_chat_model=False),
    ],

    # ── Embedding / Analysis ──────────────────────────────────────────
    Capability.EMBEDDING: [
        CandidateModel(model_id="sentence-transformers/all-MiniLM-L6-v2", display_name="MiniLM-L6-v2", capability=Capability.EMBEDDING, is_chat_model=False),
        CandidateModel(model_id="BAAI/bge-small-en-v1.5", display_name="BGE-Small-EN", capability=Capability.EMBEDDING, is_chat_model=False),
    ],
    Capability.SENTIMENT_ANALYSIS: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.SENTIMENT_ANALYSIS),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.SENTIMENT_ANALYSIS),
    ],
    Capability.DATA_EXTRACTION: [
        CandidateModel(model_id="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B", capability=Capability.DATA_EXTRACTION),
        CandidateModel(model_id="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B", capability=Capability.DATA_EXTRACTION),
    ],
}


class ModelDiscoveryService:
    """Returns candidate models for a given capability type.

    The service maintains a curated pool of models known to work with the
    HuggingFace Inference API.  Users can also register custom candidates.
    """

    def __init__(self) -> None:
        # Deep copy so mutations don't affect the module-level pool
        self._pool: dict[Capability, list[CandidateModel]] = {
            k: list(v) for k, v in _CANDIDATE_POOL.items()
        }

    def get_candidates(self, capability: Capability) -> list[CandidateModel]:
        """Return candidate models for a capability, falling back to TEXT_GENERATION."""
        candidates = self._pool.get(capability)
        if not candidates:
            logger.warning(
                "No candidates for capability %s – falling back to text_generation",
                capability.value,
            )
            candidates = self._pool.get(Capability.TEXT_GENERATION, [])
        logger.info(
            "Found %d candidates for %s: %s",
            len(candidates),
            capability.value,
            [c.model_id for c in candidates],
        )
        return candidates

    def get_candidate_ids(self, capability: Capability) -> list[str]:
        """Return just the model IDs for a capability."""
        return [c.model_id for c in self.get_candidates(capability)]

    def add_candidate(self, capability: Capability, candidate: CandidateModel) -> None:
        """Register an additional candidate model for a capability."""
        if capability not in self._pool:
            self._pool[capability] = []
        self._pool[capability].append(candidate)
        logger.info("Added candidate %s for %s", candidate.model_id, capability.value)

    def list_capabilities(self) -> list[Capability]:
        """Return all capabilities that have at least one candidate."""
        return [cap for cap, models in self._pool.items() if models]

    def list_all_candidates(self) -> dict[str, list[str]]:
        """Return a full map of capability → model IDs."""
        return {
            cap.value: [c.model_id for c in models]
            for cap, models in self._pool.items()
            if models
        }
