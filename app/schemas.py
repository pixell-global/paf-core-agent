"""Pydantic schemas for request/response models."""

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """SSE event types."""
    THINKING = "thinking"
    CONTENT = "content"
    COMPLETE = "complete"
    ERROR = "error"
    DONE = "done"


class FileContext(BaseModel):
    """File context for chat requests."""
    path: str = Field(description="File path")
    content: str = Field(description="File content")
    line_start: Optional[int] = Field(default=None, description="Start line number")
    line_end: Optional[int] = Field(default=None, description="End line number")


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(description="User message")
    show_thinking: bool = Field(default=False, description="Show thinking events")
    files: Optional[List[FileContext]] = Field(default=None, description="File contexts")
    model: Optional[str] = Field(default=None, description="Specific model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SSEEvent(BaseModel):
    """Server-Sent Event model."""
    event: EventType = Field(description="Event type")
    data: str = Field(description="Event data")
    id: Optional[str] = Field(default=None, description="Event ID")


class ThinkingEvent(BaseModel):
    """Thinking event data."""
    phase: str = Field(description="UPEE phase name")
    content: str = Field(description="Thinking content")
    timestamp: float = Field(description="Event timestamp")


class ContentEvent(BaseModel):
    """Content event data."""
    content: str = Field(description="Content chunk")
    timestamp: float = Field(description="Event timestamp")


class CompleteEvent(BaseModel):
    """Completion event data."""
    total_tokens: int = Field(description="Total tokens used")
    duration: float = Field(description="Request duration in seconds")
    model: str = Field(description="Model used")
    timestamp: float = Field(description="Event timestamp")


class ErrorEvent(BaseModel):
    """Error event data."""
    error: str = Field(description="Error message")
    error_type: str = Field(description="Error type")
    timestamp: float = Field(description="Event timestamp")


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(description="Overall health status")
    version: str = Field(description="Application version")
    timestamp: float = Field(description="Health check timestamp")
    services: Dict[str, Dict[str, Any]] = Field(description="Service statuses")


class ServiceHealth(BaseModel):
    """Individual service health."""
    status: str = Field(description="Service status")
    latency_ms: Optional[float] = Field(default=None, description="Service latency")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")
    last_check: float = Field(description="Last health check timestamp")


class UPEEPhase(str, Enum):
    """UPEE loop phases."""
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    EVALUATE = "evaluate"


class UPEEResult(BaseModel):
    """UPEE processing result."""
    phase: UPEEPhase = Field(description="Current phase")
    content: str = Field(description="Phase output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Phase metadata")
    completed: bool = Field(default=False, description="Whether phase is completed")
    error: Optional[str] = Field(default=None, description="Error if phase failed") 