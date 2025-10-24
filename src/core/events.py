"""
Event models and types for the event bus system.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EventType(str, Enum):
    """Event types for the event bus."""
    USER_PROMPT = "user_prompt"
    UPEE_COMPLETE = "upee_complete"
    WORKER_TASK_CREATED = "worker_task_created"
    WORKER_TASK_COMPLETED = "worker_task_completed"
    WORKER_TASK_FAILED = "worker_task_failed"
    JOB_SCHEDULED = "job_scheduled"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    PLUGIN_INSTALLED = "plugin_installed"
    PLUGIN_UNINSTALLED = "plugin_uninstalled"
    SYSTEM_HEALTH_CHECK = "system_health_check"


class BaseEvent(BaseModel):
    """Base event model for all events."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: EventType
    payload: Dict[str, Any]
    source: Optional[str] = None
    tenant_id: Optional[uuid.UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class UserPromptEvent(BaseEvent):
    """Event triggered when user sends a prompt."""
    event_type: EventType = EventType.USER_PROMPT
    
    def __init__(self, user_id: uuid.UUID, message: str, conversation_id: uuid.UUID, **kwargs):
        super().__init__(
            payload={
                "user_id": str(user_id),
                "message": message,
                "conversation_id": str(conversation_id)
            },
            **kwargs
        )


class UPEECompleteEvent(BaseEvent):
    """Event triggered when UPEE processing is complete."""
    event_type: EventType = EventType.UPEE_COMPLETE
    
    def __init__(self, conversation_id: uuid.UUID, request_id: str, quality_score: float, **kwargs):
        super().__init__(
            payload={
                "conversation_id": str(conversation_id),
                "request_id": request_id,
                "quality_score": quality_score
            },
            **kwargs
        )


class WorkerTaskEvent(BaseEvent):
    """Base event for worker task events."""
    
    def __init__(self, task_id: uuid.UUID, worker_id: uuid.UUID, task_type: str, **kwargs):
        super().__init__(
            payload={
                "task_id": str(task_id),
                "worker_id": str(worker_id),
                "task_type": task_type
            },
            **kwargs
        )


class WorkerTaskCreatedEvent(WorkerTaskEvent):
    """Event triggered when a worker task is created."""
    event_type: EventType = EventType.WORKER_TASK_CREATED


class WorkerTaskCompletedEvent(WorkerTaskEvent):
    """Event triggered when a worker task is completed."""
    event_type: EventType = EventType.WORKER_TASK_COMPLETED
    
    def __init__(self, task_id: uuid.UUID, worker_id: uuid.UUID, task_type: str, result: Dict[str, Any], **kwargs):
        super().__init__(task_id, worker_id, task_type, **kwargs)
        self.payload["result"] = result


class WorkerTaskFailedEvent(WorkerTaskEvent):
    """Event triggered when a worker task fails."""
    event_type: EventType = EventType.WORKER_TASK_FAILED
    
    def __init__(self, task_id: uuid.UUID, worker_id: uuid.UUID, task_type: str, error: str, **kwargs):
        super().__init__(task_id, worker_id, task_type, **kwargs)
        self.payload["error"] = error


class JobEvent(BaseEvent):
    """Base event for job events."""
    
    def __init__(self, job_id: uuid.UUID, job_name: str, job_type: str, **kwargs):
        super().__init__(
            payload={
                "job_id": str(job_id),
                "job_name": job_name,
                "job_type": job_type
            },
            **kwargs
        )


class JobScheduledEvent(JobEvent):
    """Event triggered when a job is scheduled."""
    event_type: EventType = EventType.JOB_SCHEDULED


class JobCompletedEvent(JobEvent):
    """Event triggered when a job is completed."""
    event_type: EventType = EventType.JOB_COMPLETED
    
    def __init__(self, job_id: uuid.UUID, job_name: str, job_type: str, result: Dict[str, Any], **kwargs):
        super().__init__(job_id, job_name, job_type, **kwargs)
        self.payload["result"] = result


class JobFailedEvent(JobEvent):
    """Event triggered when a job fails."""
    event_type: EventType = EventType.JOB_FAILED
    
    def __init__(self, job_id: uuid.UUID, job_name: str, job_type: str, error: str, **kwargs):
        super().__init__(job_id, job_name, job_type, **kwargs)
        self.payload["error"] = error


class SystemHealthCheckEvent(BaseEvent):
    """Event triggered for system health checks."""
    event_type: EventType = EventType.SYSTEM_HEALTH_CHECK
    
    def __init__(self, status: str, services: Dict[str, Any], **kwargs):
        super().__init__(
            payload={
                "status": status,
                "services": services
            },
            **kwargs
        )