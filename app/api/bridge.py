"""
A2A Bridge API endpoints.
"""

import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.core.bridge.a2a_bridge import A2ABridge
from app.core.bridge.protocol import MessageType, MessagePriority

router = APIRouter()

# Global bridge instance - in production would be managed differently
bridge_instance: Optional[A2ABridge] = None


class SendMessageRequest(BaseModel):
    message_type: MessageType
    target_agent_id: Optional[str] = None
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    conversation_id: Optional[str] = None
    ttl_seconds: Optional[int] = None
    requires_ack: bool = False


class TaskRequestRequest(BaseModel):
    target_agent_id: str
    task_type: str
    task_data: Dict[str, Any]
    conversation_id: Optional[str] = None
    timeout_seconds: int = 300
    requirements: Optional[Dict[str, Any]] = None


class StatusUpdateRequest(BaseModel):
    status: str
    capabilities: List[str]
    current_load: Optional[int] = None
    max_capacity: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class UiEventRequest(BaseModel):
    event: Dict[str, Any]
    target_agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    correlation_id: Optional[str] = None


def get_bridge() -> A2ABridge:
    """Get the bridge instance."""
    global bridge_instance
    if not bridge_instance:
        # Initialize bridge with default agent ID
        bridge_instance = A2ABridge("paf-core-agent")
    return bridge_instance


@router.post("/start")
async def start_bridge(db: AsyncSession = Depends(get_db)):
    """Start the A2A bridge."""
    try:
        bridge = get_bridge()
        await bridge.start()
        
        return {
            "message": "A2A bridge started successfully",
            "agent_id": bridge.agent_id,
            "status": bridge.get_bridge_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bridge: {str(e)}")


@router.post("/stop")
async def stop_bridge(db: AsyncSession = Depends(get_db)):
    """Stop the A2A bridge."""
    try:
        bridge = get_bridge()
        await bridge.stop()
        
        return {"message": "A2A bridge stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop bridge: {str(e)}")


@router.get("/status")
async def get_bridge_status(db: AsyncSession = Depends(get_db)):
    """Get bridge status."""
    try:
        bridge = get_bridge()
        return bridge.get_bridge_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/agents")
async def get_connected_agents(db: AsyncSession = Depends(get_db)):
    """Get list of connected agents."""
    try:
        bridge = get_bridge()
        return bridge.get_connected_agents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")


@router.post("/message")
async def send_message(
    request: SendMessageRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send an A2A message."""
    try:
        bridge = get_bridge()
        
        message = await bridge.send_message(
            message_type=request.message_type,
            target_agent_id=request.target_agent_id,
            payload=request.payload,
            priority=request.priority,
            conversation_id=request.conversation_id,
            ttl_seconds=request.ttl_seconds,
            requires_ack=request.requires_ack
        )
        
        return {
            "message_id": message.id,
            "status": "sent",
            "message": message.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send message: {str(e)}")


@router.post("/task/request")
async def send_task_request(
    request: TaskRequestRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send a task request to another agent."""
    try:
        bridge = get_bridge()
        
        message = await bridge.send_task_request(
            target_agent_id=request.target_agent_id,
            task_type=request.task_type,
            task_data=request.task_data,
            conversation_id=request.conversation_id,
            timeout_seconds=request.timeout_seconds,
            requirements=request.requirements
        )
        
        return {
            "task_request_id": message.id,
            "status": "sent",
            "message": message.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send task request: {str(e)}")


@router.post("/status/broadcast")
async def broadcast_status(
    request: StatusUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Broadcast status update to all connected agents."""
    try:
        bridge = get_bridge()
        
        message = await bridge.broadcast_status_update(
            status=request.status,
            capabilities=request.capabilities,
            current_load=request.current_load,
            max_capacity=request.max_capacity,
            metadata=request.metadata
        )
        
        return {
            "broadcast_id": message.id,
            "status": "sent",
            "message": message.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to broadcast status: {str(e)}")


@router.get("/router/status")
async def get_router_status(db: AsyncSession = Depends(get_db)):
    """Get message router status."""
    try:
        bridge = get_bridge()
        return bridge.router.get_queue_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get router status: {str(e)}")


@router.post("/cleanup/expired")
async def cleanup_expired_requests(db: AsyncSession = Depends(get_db)):
    """Clean up expired pending requests."""
    try:
        bridge = get_bridge()
        expired_count = await bridge.cleanup_expired_requests()
        
        return {
            "message": "Cleanup completed",
            "expired_requests": expired_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {str(e)}")


@router.post("/test/echo")
async def test_echo_message(
    message: str = "Hello from A2A bridge",
    target_agent_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Send a test echo message."""
    try:
        bridge = get_bridge()
        
        message_obj = await bridge.send_message(
            message_type=MessageType.HEARTBEAT,
            target_agent_id=target_agent_id,
            payload={"echo": message, "timestamp": "now"},
            priority=MessagePriority.LOW
        )
        
        return {
            "test_message_id": message_obj.id,
            "echo_content": message,
            "status": "sent"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send test message: {str(e)}")


@router.post("/ui/event")
async def forward_ui_event(request: UiEventRequest, db: AsyncSession = Depends(get_db)):
    """Accept a ui.event envelope from client and forward to target agent.

    Returns action.result from the agent if available.
    """
    try:
        bridge = get_bridge()
        if not isinstance(request.event, dict) or request.event.get("type") != "ui.event":
            raise HTTPException(status_code=400, detail="Invalid ui.event envelope")

        payload = request.event
        message = await bridge.send_message(
            message_type=MessageType.COORDINATION,
            target_agent_id=request.target_agent_id,
            payload=payload,
            priority=MessagePriority.NORMAL,
            conversation_id=request.conversation_id,
        )

        return {
            "status": "sent",
            "message_id": message.id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to forward ui.event: {str(e)}")