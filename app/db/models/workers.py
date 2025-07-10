"""
Worker management models for tracking worker instances and their tasks.
These models support the distributed worker architecture.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base

class WorkerInstance(Base):
    """
    Worker Instance model representing individual worker agents.
    Each worker can handle specific types of tasks based on capabilities.
    """
    __tablename__ = "worker_instances"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Worker identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)  # gRPC endpoint
    
    # Worker capabilities and metadata
    capabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    meta_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Worker status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")  # active, inactive, unhealthy
    
    # Health monitoring
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    tasks: Mapped[List["WorkerTask"]] = relationship(
        "WorkerTask", 
        back_populates="worker",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<WorkerInstance(id={self.id}, name='{self.name}', status='{self.status}')>"

class WorkerTask(Base):
    """
    Worker Task model representing individual tasks assigned to workers.
    Tracks task lifecycle from creation to completion.
    """
    __tablename__ = "worker_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to worker
    worker_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("worker_instances.id", ondelete="SET NULL"),
        nullable=True,  # Can be null if worker is deleted
        index=True
    )
    
    # Task details
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Task status tracking
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps for performance tracking
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Priority and retry logic
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)  # 1=highest, 10=lowest
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    
    # Relationships
    worker: Mapped[Optional["WorkerInstance"]] = relationship("WorkerInstance", back_populates="tasks")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self) -> str:
        return f"<WorkerTask(id={self.id}, task_type='{self.task_type}', status='{self.status}')>"