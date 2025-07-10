"""
Main A2A Bridge implementation for agent-to-agent communication.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.db.database import AsyncSessionLocal
from app.db.models.jobs import Event
from app.core.event_bus import event_bus
from app.core.events import BaseEvent, EventType
from .protocol import A2AMessage, MessageType, MessagePriority, A2AProtocol
from .message_router import MessageRouter

logger = logging.getLogger(__name__)


class A2ABridge:
    """Agent-to-Agent communication bridge."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.router = MessageRouter()
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.connected_agents: Dict[str, Dict[str, Any]] = {}
        self.running = False
        
        # Message statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "conversations_started": 0,
            "tasks_completed": 0,
            "errors": 0
        }
    
    async def start(self) -> None:
        """Start the A2A bridge."""
        if self.running:
            logger.warning(f"A2A bridge for agent {self.agent_id} is already running")
            return
        
        self.running = True
        
        # Start message router
        await self.router.start()
        
        # Register self as message handler
        self.router.register_agent_handler(self.agent_id, self._handle_incoming_message)
        
        # Subscribe to relevant events
        event_bus.subscribe(EventType.USER_PROMPT, self._handle_user_prompt_event)
        
        logger.info(f"A2A bridge started for agent {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the A2A bridge."""
        if not self.running:
            return
        
        self.running = False
        
        # Send offline status to connected agents
        await self._broadcast_status_update("offline")
        
        # Stop message router
        await self.router.stop()
        
        logger.info(f"A2A bridge stopped for agent {self.agent_id}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable[[A2AMessage], Any]) -> None:
        """Register a handler for specific message types."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for message type {message_type.value}")
    
    async def send_message(
        self,
        message_type: MessageType,
        target_agent_id: Optional[str],
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        requires_ack: bool = False,
        correlation_id: Optional[str] = None
    ) -> A2AMessage:
        """Send an A2A message."""
        
        message = A2AMessage.create(
            message_type=message_type,
            source_agent_id=self.agent_id,
            target_agent_id=target_agent_id,
            payload=payload,
            priority=priority,
            conversation_id=conversation_id,
            ttl_seconds=ttl_seconds,
            requires_ack=requires_ack,
            correlation_id=correlation_id
        )
        
        # Validate message
        if not A2AProtocol.validate_message(message):
            raise ValueError(f"Invalid message structure: {message}")
        
        # Store message in database for persistence
        await self._store_message(message)
        
        # Route message
        success = await self.router.route_message(message)
        
        if success:
            self.stats["messages_sent"] += 1
            logger.info(f"Sent message {message.id} from {self.agent_id} to {target_agent_id or 'broadcast'}")
            
            # Track pending request if requires ack
            if requires_ack:
                self.pending_requests[message.id] = {
                    "message": message,
                    "sent_at": datetime.now(timezone.utc),
                    "timeout_seconds": ttl_seconds or 300
                }
        else:
            self.stats["errors"] += 1
            logger.error(f"Failed to send message {message.id}")
        
        return message
    
    async def send_task_request(
        self,
        target_agent_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        timeout_seconds: int = 300,
        requirements: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """Send a task request to another agent."""
        
        payload = {
            "task_type": task_type,
            "task_data": task_data,
            "timeout_seconds": timeout_seconds,
            "requirements": requirements or {},
            "context": {
                "source_agent": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        message = await self.send_message(
            message_type=MessageType.TASK_REQUEST,
            target_agent_id=target_agent_id,
            payload=payload,
            conversation_id=conversation_id,
            ttl_seconds=timeout_seconds,
            requires_ack=True,
            priority=MessagePriority.HIGH
        )
        
        return message
    
    async def send_task_response(
        self,
        target_agent_id: str,
        original_message: A2AMessage,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_seconds: Optional[float] = None
    ) -> A2AMessage:
        """Send a task response."""
        
        payload = {
            "task_id": original_message.id,
            "status": status,
            "result": result,
            "error_message": error_message,
            "execution_time_seconds": execution_time_seconds,
            "worker_agent_id": self.agent_id
        }
        
        message = await self.send_message(
            message_type=MessageType.TASK_RESPONSE,
            target_agent_id=target_agent_id,
            payload=payload,
            conversation_id=original_message.conversation_id,
            correlation_id=original_message.id,
            priority=MessagePriority.HIGH
        )
        
        return message
    
    async def broadcast_status_update(
        self,
        status: str,
        capabilities: List[str],
        current_load: Optional[int] = None,
        max_capacity: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """Broadcast status update to all connected agents."""
        
        payload = {
            "agent_id": self.agent_id,
            "status": status,
            "capabilities": capabilities,
            "current_load": current_load,
            "max_capacity": max_capacity,
            "metadata": metadata or {}
        }
        
        message = await self.send_message(
            message_type=MessageType.STATUS_UPDATE,
            target_agent_id=None,  # Broadcast
            payload=payload,
            priority=MessagePriority.NORMAL
        )
        
        return message
    
    async def _handle_incoming_message(self, message: A2AMessage) -> None:
        """Handle incoming A2A message."""
        try:
            logger.debug(f"Received message {message.id} of type {message.type.value}")
            self.stats["messages_received"] += 1
            
            # Send ACK if required
            if message.requires_ack:
                ack_message = A2AProtocol.create_ack_message(message, self.agent_id)
                await self.router.route_message(ack_message)
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for type {message.type.value}: {e}")
                    self.stats["errors"] += 1
            
            # Handle specific message types
            await self._handle_message_by_type(message)
            
        except Exception as e:
            logger.error(f"Error handling incoming message {message.id}: {e}")
            self.stats["errors"] += 1
    
    async def _handle_message_by_type(self, message: A2AMessage) -> None:
        """Handle message based on its type."""
        
        if message.type == MessageType.STATUS_UPDATE:
            await self._handle_status_update(message)
        elif message.type == MessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.type == MessageType.TASK_RESPONSE:
            await self._handle_task_response(message)
        elif message.type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.type == MessageType.ERROR:
            await self._handle_error_message(message)
    
    async def _handle_status_update(self, message: A2AMessage) -> None:
        """Handle status update message."""
        payload = message.payload
        source_agent = message.source_agent_id
        
        # Update connected agents registry
        self.connected_agents[source_agent] = {
            "status": payload.get("status"),
            "capabilities": payload.get("capabilities", []),
            "current_load": payload.get("current_load"),
            "max_capacity": payload.get("max_capacity"),
            "last_seen": datetime.now(timezone.utc),
            "metadata": payload.get("metadata", {})
        }
        
        logger.debug(f"Updated status for agent {source_agent}: {payload.get('status')}")
    
    async def _handle_task_request(self, message: A2AMessage) -> None:
        """Handle task request message."""
        # This would typically be overridden by specific agent implementations
        logger.info(f"Received task request {message.id} from {message.source_agent_id}")
        
        # Send a basic response
        await self.send_task_response(
            target_agent_id=message.source_agent_id,
            original_message=message,
            status="error",
            error_message="Task processing not implemented"
        )
    
    async def _handle_task_response(self, message: A2AMessage) -> None:
        """Handle task response message."""
        payload = message.payload
        original_task_id = payload.get("task_id")
        
        # Remove from pending requests
        if original_task_id in self.pending_requests:
            del self.pending_requests[original_task_id]
            self.stats["tasks_completed"] += 1
        
        logger.info(f"Received task response for task {original_task_id}: {payload.get('status')}")
    
    async def _handle_heartbeat(self, message: A2AMessage) -> None:
        """Handle heartbeat message."""
        # Update last seen time for source agent
        if message.source_agent_id in self.connected_agents:
            self.connected_agents[message.source_agent_id]["last_seen"] = datetime.now(timezone.utc)
    
    async def _handle_error_message(self, message: A2AMessage) -> None:
        """Handle error message."""
        logger.error(f"Received error from {message.source_agent_id}: {message.payload.get('error')}")
        self.stats["errors"] += 1
    
    async def _handle_user_prompt_event(self, event: BaseEvent) -> None:
        """Handle user prompt events from the event bus."""
        # This could trigger task distribution to worker agents
        logger.debug(f"User prompt event received: {event.payload}")
    
    async def _broadcast_status_update(self, status: str) -> None:
        """Internal method to broadcast status update."""
        try:
            await self.broadcast_status_update(
                status=status,
                capabilities=["upee", "orchestrator"],
                current_load=0,
                max_capacity=100
            )
        except Exception as e:
            logger.error(f"Error broadcasting status update: {e}")
    
    async def _store_message(self, message: A2AMessage) -> None:
        """Store message in database for persistence and audit."""
        try:
            async with AsyncSessionLocal() as db:
                # Store as event for audit trail
                db_event = Event(
                    event_type="a2a_message",
                    payload={
                        "message_id": message.id,
                        "type": message.type.value,
                        "source_agent_id": message.source_agent_id,
                        "target_agent_id": message.target_agent_id,
                        "priority": message.priority.value,
                        "payload": message.payload
                    },
                    source=f"a2a_bridge_{self.agent_id}"
                )
                db.add(db_event)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error storing message {message.id}: {e}")
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "connected_agents": len(self.connected_agents),
            "active_conversations": len(self.active_conversations),
            "pending_requests": len(self.pending_requests),
            "router_status": self.router.get_queue_status(),
            "stats": self.stats.copy()
        }
    
    def get_connected_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get list of connected agents and their status."""
        return self.connected_agents.copy()
    
    async def cleanup_expired_requests(self) -> int:
        """Clean up expired pending requests."""
        current_time = datetime.now(timezone.utc)
        expired_requests = []
        
        for request_id, request_data in self.pending_requests.items():
            sent_at = request_data["sent_at"]
            timeout = request_data["timeout_seconds"]
            
            if (current_time - sent_at).total_seconds() > timeout:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            del self.pending_requests[request_id]
            logger.warning(f"Request {request_id} expired")
        
        return len(expired_requests)