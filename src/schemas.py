"""Pydantic schemas for request/response models."""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, validator


class EventType(str, Enum):
    """SSE event types."""
    THINKING = "thinking"
    CONTENT = "content"
    COMPLETE = "complete"
    ERROR = "error"
    DONE = "done"


class FileContent(BaseModel):
    """File content for chat requests."""
    file_name: str = Field(description="Name of the file")
    file_path: Optional[str] = Field(default=None, description="Full path to the file")
    content: Optional[str] = Field(default=None, description="File content (for small files)")
    signed_url: Optional[str] = Field(default=None, description="Signed URL for large files")
    file_type: str = Field(description="MIME type or file extension")
    file_size: int = Field(description="File size in bytes")
    line_start: Optional[int] = Field(default=None, description="Start line number for partial content")
    line_end: Optional[int] = Field(default=None, description="End line number for partial content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional file metadata")
    
    @validator('content', 'signed_url')
    def content_or_url_required(cls, v, values):
        """Ensure either content or signed_url is provided."""
        if not v and not values.get('signed_url') and not values.get('content'):
            raise ValueError('Either content or signed_url must be provided')
        return v


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")
    files: Optional[List[FileContent]] = Field(default=None, description="Files associated with this message")


class FileContext(BaseModel):
    """Legacy file context for backwards compatibility."""
    path: str = Field(description="File path")
    content: str = Field(description="File content")
    line_start: Optional[int] = Field(default=None, description="Start line number")
    line_end: Optional[int] = Field(default=None, description="End line number")


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(description="User message")
    show_thinking: bool = Field(default=False, description="Show thinking events")
    files: Optional[List[Union[FileContext, FileContent]]] = Field(default=None, description="File contexts (legacy and new format)")
    history: Optional[List[ConversationMessage]] = Field(default=None, description="Conversation history for short-term memory")
    model: Optional[str] = Field(default=None, description="Specific model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    context_window_size: Optional[int] = Field(default=None, description="Override default context window size")
    memory_limit: Optional[int] = Field(default=10, description="Maximum number of history messages to keep")


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


# Job management schemas
class JobCreate(BaseModel):
    """Job creation request."""
    name: str = Field(description="Job name")
    job_type: str = Field(description="Job type")
    payload: Dict[str, Any] = Field(description="Job payload")
    schedule_spec: Optional[str] = Field(default=None, description="Schedule specification")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: Optional[int] = Field(default=None, description="Job timeout in seconds")


class JobResponse(BaseModel):
    """Job response model."""
    id: uuid.UUID = Field(description="Job ID")
    name: str = Field(description="Job name")
    job_type: str = Field(description="Job type")
    schedule_spec: Optional[str] = Field(default=None, description="Schedule specification")
    status: str = Field(description="Job status")
    payload: Dict[str, Any] = Field(description="Job payload")
    created_at: datetime = Field(description="Creation timestamp")
    scheduled_at: Optional[datetime] = Field(default=None, description="Scheduled execution time")
    max_retries: int = Field(description="Maximum retry attempts")

    class Config:
        json_encoders = {
            uuid.UUID: str,
            datetime: lambda v: v.isoformat()
        } 