"""Celery application for distributing evaluation experiments."""

from __future__ import annotations

from celery import Celery

from arena.config import get_settings

settings = get_settings()

celery_app = Celery(
    "arena",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 h
)
