"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Application-wide settings with sensible defaults."""

    # ── Database ──────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://arena:arena@localhost:5432/arena",
        )
    )
    database_echo: bool = field(
        default_factory=lambda: os.getenv("DATABASE_ECHO", "false").lower() == "true"
    )

    # ── Redis / Celery ────────────────────────────────────────────────
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    celery_broker_url: str = field(
        default_factory=lambda: os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    )
    celery_result_backend: str = field(
        default_factory=lambda: os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    )

    # ── HuggingFace ───────────────────────────────────────────────────
    hf_api_token: str = field(
        default_factory=lambda: os.getenv("HF_API_TOKEN", "")
    )
    hf_inference_base_url: str = field(
        default_factory=lambda: os.getenv(
            "HF_INFERENCE_BASE_URL",
            "https://router.huggingface.co/hf-inference/models",
        )
    )

    # ── Evaluation defaults ───────────────────────────────────────────
    default_max_samples: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_MAX_SAMPLES", "50"))
    )
    default_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TIMEOUT_SECONDS", "120"))
    )
    max_concurrent_models: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_MODELS", "5"))
    )

    # ── Cost estimation (USD per 1K tokens) ───────────────────────────
    cost_per_1k_input_tokens: float = field(
        default_factory=lambda: float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0.0015"))
    )
    cost_per_1k_output_tokens: float = field(
        default_factory=lambda: float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0.002"))
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
