"""
Message routing and delivery system for A2A communication.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timezone

from .protocol import A2AMessage, MessageType, MessagePriority

logger = logging.getLogger(__name__)


class MessageRouter:
    """Routes A2A messages between agents with priority queuing and delivery guarantees."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.agent_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self.delivery_callbacks: Dict[str, Callable] = {}
        self.running = False
        self.delivery_task: Optional[asyncio.Task] = None
        self.stats = {
            "messages_routed": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_expired": 0
        }
    
    async def start(self) -> None:
        """Start the message router."""
        if self.running:
            logger.warning("Message router is already running")
            return
        
        self.running = True
        self.delivery_task = asyncio.create_task(self._delivery_loop())
        logger.info("Message router started")
    
    async def stop(self) -> None:
        """Stop the message router."""
        if not self.running:
            return
        
        self.running = False
        
        if self.delivery_task:
            self.delivery_task.cancel()
            try:
                await self.delivery_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message router stopped")
    
    def register_agent_handler(self, agent_id: str, handler: Callable[[A2AMessage], Any]) -> None:
        """Register a message handler for an agent."""
        self.message_handlers[agent_id].append(handler)
        logger.info(f"Registered message handler for agent {agent_id}")
    
    def unregister_agent_handler(self, agent_id: str, handler: Callable[[A2AMessage], Any]) -> None:
        """Unregister a message handler for an agent."""
        if agent_id in self.message_handlers:
            try:
                self.message_handlers[agent_id].remove(handler)
                logger.info(f"Unregistered message handler for agent {agent_id}")
            except ValueError:
                logger.warning(f"Handler not found for agent {agent_id}")
    
    def register_delivery_callback(self, agent_id: str, callback: Callable[[A2AMessage, bool], Any]) -> None:
        """Register a delivery confirmation callback for an agent."""
        self.delivery_callbacks[agent_id] = callback
        logger.info(f"Registered delivery callback for agent {agent_id}")
    
    async def route_message(self, message: A2AMessage) -> bool:
        """Route a message to its destination(s)."""
        try:
            # Check if message has expired
            if self._is_expired(message):
                logger.warning(f"Message {message.id} has expired, discarding")
                self.stats["messages_expired"] += 1
                return False
            
            # Route based on target
            if message.target_agent_id:
                # Unicast
                return await self._route_unicast(message)
            else:
                # Broadcast
                return await self._route_broadcast(message)
        
        except Exception as e:
            logger.error(f"Error routing message {message.id}: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def _route_unicast(self, message: A2AMessage) -> bool:
        """Route message to specific agent."""
        target_agent_id = message.target_agent_id
        
        # Add to agent's queue with priority ordering
        queue = self.agent_queues[target_agent_id]
        self._insert_by_priority(queue, message)
        
        self.stats["messages_routed"] += 1
        logger.debug(f"Routed message {message.id} to agent {target_agent_id}")
        return True
    
    async def _route_broadcast(self, message: A2AMessage) -> bool:
        """Route message to all registered agents."""
        routed_count = 0
        
        for agent_id in self.message_handlers.keys():
            # Don't send back to source
            if agent_id == message.source_agent_id:
                continue
            
            queue = self.agent_queues[agent_id]
            self._insert_by_priority(queue, message)
            routed_count += 1
        
        self.stats["messages_routed"] += routed_count
        logger.debug(f"Broadcast message {message.id} to {routed_count} agents")
        return routed_count > 0
    
    def _insert_by_priority(self, queue: deque, message: A2AMessage) -> None:
        """Insert message into queue based on priority."""
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3
        }
        
        message_priority = priority_order.get(message.priority, 2)
        
        # Find insertion point
        insert_index = len(queue)
        for i, existing_message in enumerate(queue):
            existing_priority = priority_order.get(existing_message.priority, 2)
            if message_priority < existing_priority:
                insert_index = i
                break
        
        # Insert at correct position
        if insert_index >= len(queue):
            queue.append(message)
        else:
            # Convert to list, insert, convert back
            queue_list = list(queue)
            queue_list.insert(insert_index, message)
            queue.clear()
            queue.extend(queue_list)
    
    async def _delivery_loop(self) -> None:
        """Main delivery loop."""
        while self.running:
            try:
                delivered_any = False
                
                # Process queues for all agents
                for agent_id, queue in self.agent_queues.items():
                    if queue and agent_id in self.message_handlers:
                        message = queue.popleft()
                        
                        # Check expiration before delivery
                        if self._is_expired(message):
                            logger.warning(f"Message {message.id} expired before delivery")
                            self.stats["messages_expired"] += 1
                            continue
                        
                        # Deliver to all handlers for this agent
                        delivery_success = await self._deliver_to_agent(agent_id, message)
                        
                        if delivery_success:
                            self.stats["messages_delivered"] += 1
                            delivered_any = True
                        else:
                            self.stats["messages_failed"] += 1
                
                # Sleep briefly if no messages were delivered
                if not delivered_any:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in delivery loop: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_to_agent(self, agent_id: str, message: A2AMessage) -> bool:
        """Deliver message to specific agent."""
        try:
            handlers = self.message_handlers.get(agent_id, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler for agent {agent_id}: {e}")
                    return False
            
            # Call delivery callback if registered
            if agent_id in self.delivery_callbacks:
                try:
                    callback = self.delivery_callbacks[agent_id]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message, True)
                    else:
                        callback(message, True)
                except Exception as e:
                    logger.error(f"Error in delivery callback for agent {agent_id}: {e}")
            
            logger.debug(f"Delivered message {message.id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deliver message {message.id} to agent {agent_id}: {e}")
            
            # Call delivery callback with failure
            if agent_id in self.delivery_callbacks:
                try:
                    callback = self.delivery_callbacks[agent_id]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message, False)
                    else:
                        callback(message, False)
                except Exception:
                    pass
            
            return False
    
    def _is_expired(self, message: A2AMessage) -> bool:
        """Check if message has expired."""
        if not message.ttl_seconds:
            return False
        
        age = (datetime.now(timezone.utc) - message.created_at).total_seconds()
        return age > message.ttl_seconds
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "agent_queues": {
                agent_id: len(queue) 
                for agent_id, queue in self.agent_queues.items()
            },
            "total_queued": sum(len(queue) for queue in self.agent_queues.values()),
            "registered_agents": list(self.message_handlers.keys()),
            "stats": self.stats.copy()
        }
    
    def clear_expired_messages(self) -> int:
        """Clear expired messages from all queues."""
        cleared_count = 0
        
        for agent_id, queue in self.agent_queues.items():
            to_remove = []
            for i, message in enumerate(queue):
                if self._is_expired(message):
                    to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(to_remove):
                del queue[i]
                cleared_count += 1
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} expired messages")
            self.stats["messages_expired"] += cleared_count
        
        return cleared_count