"""
Job scheduling models for managing scheduled tasks and job execution.
Supports cron jobs, one-time jobs, and interval-based jobs.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base

class Job(Base):
    """
    Job model representing scheduled tasks in the system.
    Supports cron expressions, intervals, and one-time execution.
    """
    __tablename__ = "core_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'one_time', 'cron', 'interval'
    
    # Scheduling configuration
    schedule_spec: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Cron expression or interval
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Execution tracking
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")  # queued, running, completed, failed, paused
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Worker assignment
    worker_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("worker_instances.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Job configuration
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeout_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    executions: Mapped[List["JobExecution"]] = relationship(
        "JobExecution", 
        back_populates="job",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, name='{self.name}', job_type='{self.job_type}', status='{self.status}')>"

class JobExecution(Base):
    """
    Job Execution model tracking individual runs of scheduled jobs.
    Useful for monitoring job history and performance.
    """
    __tablename__ = "job_executions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to job
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("core_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Execution details
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # running, completed, failed
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Performance tracking
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Resource usage (for monitoring)
    cpu_usage: Mapped[Optional[float]] = mapped_column(JSON, nullable=True)
    memory_usage: Mapped[Optional[float]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="executions")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self) -> str:
        return f"<JobExecution(id={self.id}, job_id={self.job_id}, status='{self.status}')>"

# Event models for the event bus (Phase 2)
class Event(Base):
    """
    Event model for the event bus system.
    Stores all events flowing through the system.
    """
    __tablename__ = "flex_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Multi-tenancy support
    tenant_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    
    # Processing tracking
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<Event(id={self.id}, event_type='{self.event_type}', source='{self.source}')>"