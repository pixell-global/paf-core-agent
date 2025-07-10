"""
A2A communication protocol definitions.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone


class MessageType(str, Enum):
    """Types of A2A messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response" 
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    COORDINATION = "coordination"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class A2AMessage(BaseModel):
    """A2A communication message."""
    id: str
    type: MessageType
    source_agent_id: str
    target_agent_id: Optional[str] = None  # None for broadcast
    conversation_id: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    ttl_seconds: Optional[int] = None
    requires_ack: bool = False
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def create(
        cls,
        message_type: MessageType,
        source_agent_id: str,
        payload: Dict[str, Any],
        target_agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        requires_ack: bool = False,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> "A2AMessage":
        """Create a new A2A message."""
        return cls(
            id=str(uuid.uuid4()),
            type=message_type,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            conversation_id=conversation_id,
            priority=priority,
            payload=payload,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds,
            requires_ack=requires_ack,
            correlation_id=correlation_id,
            reply_to=reply_to
        )


class TaskRequest(BaseModel):
    """Task request payload."""
    task_type: str
    task_data: Dict[str, Any]
    timeout_seconds: Optional[int] = None
    requirements: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Task response payload."""
    task_id: str
    status: str  # success, error, timeout
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    worker_agent_id: Optional[str] = None


class StatusUpdate(BaseModel):
    """Status update payload."""
    agent_id: str
    status: str  # online, offline, busy, idle
    capabilities: List[str]
    current_load: Optional[int] = None
    max_capacity: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ResourceRequest(BaseModel):
    """Resource request payload."""
    resource_type: str
    resource_spec: Dict[str, Any]
    allocation_id: Optional[str] = None
    duration_seconds: Optional[int] = None


class ResourceResponse(BaseModel):
    """Resource response payload."""
    request_id: str
    status: str  # granted, denied, pending
    resource_details: Optional[Dict[str, Any]] = None
    allocation_id: Optional[str] = None
    expires_at: Optional[datetime] = None


class CoordinationMessage(BaseModel):
    """Coordination message payload."""
    action: str  # elect_leader, sync_state, distribute_work
    data: Dict[str, Any]
    epoch: Optional[int] = None


class A2AProtocol:
    """A2A protocol utilities and validation."""
    
    @staticmethod
    def validate_message(message: A2AMessage) -> bool:
        """Validate A2A message structure."""
        try:
            # Basic validation
            if not message.id or not message.source_agent_id:
                return False
            
            # Type-specific validation
            if message.type == MessageType.TASK_REQUEST:
                TaskRequest.model_validate(message.payload)
            elif message.type == MessageType.TASK_RESPONSE:
                TaskResponse.model_validate(message.payload)
            elif message.type == MessageType.STATUS_UPDATE:
                StatusUpdate.model_validate(message.payload)
            elif message.type == MessageType.RESOURCE_REQUEST:
                ResourceRequest.model_validate(message.payload)
            elif message.type == MessageType.RESOURCE_RESPONSE:
                ResourceResponse.model_validate(message.payload)
            elif message.type == MessageType.COORDINATION:
                CoordinationMessage.model_validate(message.payload)
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def is_expired(message: A2AMessage) -> bool:
        """Check if message has expired."""
        if not message.ttl_seconds:
            return False
        
        age = (datetime.now(timezone.utc) - message.created_at).total_seconds()
        return age > message.ttl_seconds
    
    @staticmethod
    def create_ack_message(original_message: A2AMessage, source_agent_id: str) -> A2AMessage:
        """Create acknowledgment message."""
        return A2AMessage.create(
            message_type=MessageType.STATUS_UPDATE,
            source_agent_id=source_agent_id,
            target_agent_id=original_message.source_agent_id,
            payload={"ack": True, "original_message_id": original_message.id},
            correlation_id=original_message.id
        )
    
    @staticmethod
    def create_error_message(
        source_agent_id: str,
        target_agent_id: str,
        error_message: str,
        correlation_id: Optional[str] = None
    ) -> A2AMessage:
        """Create error message."""
        return A2AMessage.create(
            message_type=MessageType.ERROR,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            payload={"error": error_message},
            correlation_id=correlation_id
        )