"""SQLAlchemy ORM models for experiment persistence."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class ExperimentRecord(Base):
    """Persistent record of a complete evaluation experiment."""

    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(50), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False)
    models = Column(JSONB, nullable=False, default=list)
    status = Column(
        Enum("pending", "running", "completed", "failed", name="experiment_status"),
        nullable=False,
        default="pending",
        index=True,
    )
    config = Column(JSONB, nullable=False, default=dict)
    leaderboard = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    results = relationship(
        "EvalResultRecord",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Experiment {self.id} [{self.status}]>"


class EvalResultRecord(Base):
    """Stores individual evaluation results for each (model, task) pair."""

    __tablename__ = "eval_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    prompt = Column(Text, nullable=True)
    reference = Column(Text, nullable=True)
    output_text = Column(Text, nullable=True)
    scores = Column(JSONB, nullable=False, default=dict)
    latency_seconds = Column(Float, nullable=False, default=0.0)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    experiment = relationship("ExperimentRecord", back_populates="results")

    def __repr__(self) -> str:
        return f"<EvalResult {self.model_id} on {self.task_id}>"
