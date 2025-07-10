"""
Event bus implementation using PostgreSQL LISTEN/NOTIFY.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Callable, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.database import AsyncSessionLocal, engine
from app.db.models.jobs import Event
from app.core.events import BaseEvent, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Event bus using PostgreSQL LISTEN/NOTIFY for real-time event distribution."""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.listener_connection: Optional[Connection] = None
        self.listener_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the database and notify listeners."""
        try:
            async with AsyncSessionLocal() as db:
                # Store event in database
                db_event = Event(
                    event_type=event.event_type.value,
                    payload=event.payload,
                    source=event.source,
                    tenant_id=event.tenant_id
                )
                db.add(db_event)
                await db.commit()
                await db.refresh(db_event)
                
                # Notify PostgreSQL listeners
                notification_payload = json.dumps({
                    "event_id": str(db_event.id),
                    "event_type": event.event_type.value,
                    "payload": event.payload,
                    "source": event.source,
                    "tenant_id": str(event.tenant_id) if event.tenant_id else None
                })
                
                await db.execute(
                    text("SELECT pg_notify('event_bus', :payload)"),
                    {"payload": notification_payload}
                )
                await db.commit()
                
                logger.info(f"Published event {event.event_type.value} with ID {db_event.id}")
                
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type.value}: {e}")
            raise
    
    def subscribe(self, event_type: EventType, handler: Callable[[BaseEvent], Any]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to event type {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[BaseEvent], Any]) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler from event type {event_type.value}")
            except ValueError:
                logger.warning(f"Handler not found for event type {event_type.value}")
    
    async def start_listener(self) -> None:
        """Start the PostgreSQL LISTEN loop."""
        if self.running:
            logger.warning("Event bus listener is already running")
            return
        
        self.running = True
        self.listener_task = asyncio.create_task(self._listener_loop())
        logger.info("Event bus listener started")
    
    async def stop_listener(self) -> None:
        """Stop the PostgreSQL LISTEN loop."""
        if not self.running:
            return
        
        self.running = False
        
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        if self.listener_connection:
            try:
                await self.listener_connection.close()
            except:
                pass
            self.listener_connection = None
        
        logger.info("Event bus listener stopped")
    
    async def _listener_loop(self) -> None:
        """Main listener loop for PostgreSQL notifications."""
        import asyncpg
        
        while self.running:
            try:
                # Create dedicated asyncpg connection for listening
                database_url = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/paf_core_agent")
                # Convert asyncpg URL format
                database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
                
                self.listener_connection = await asyncpg.connect(database_url)
                
                # Set up notification handler
                async def notification_handler(connection, pid, channel, payload):
                    await self._handle_notification(payload)
                
                await self.listener_connection.add_listener('event_bus', notification_handler)
                
                logger.info("Started listening for PostgreSQL notifications")
                
                while self.running:
                    # Keep connection alive and process notifications
                    await asyncio.sleep(1)
                        
            except asyncio.CancelledError:
                logger.info("Event bus listener cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event bus listener: {e}")
                await asyncio.sleep(1)  # Wait before retrying
            finally:
                if self.listener_connection:
                    await self.listener_connection.close()
    
    async def _handle_notification(self, notification_data: str) -> None:
        """Handle a PostgreSQL notification."""
        try:
            data = json.loads(notification_data)
            event_type_str = data["event_type"]
            
            # Convert string to EventType enum
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type_str}")
                return
            
            # Create event object
            event = BaseEvent(
                event_type=event_type,
                payload=data["payload"],
                source=data.get("source"),
                tenant_id=data.get("tenant_id")
            )
            
            # Call all subscribers for this event type
            if event_type in self.subscribers:
                for handler in self.subscribers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_type.value}: {e}")
            
            logger.debug(f"Processed notification for event type {event_type.value}")
            
        except Exception as e:
            logger.error(f"Error handling notification: {e}")
    
    async def get_events(
        self, 
        event_type: Optional[EventType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Event]:
        """Get events from the database."""
        async with AsyncSessionLocal() as db:
            if event_type:
                query = text("""
                    SELECT id, event_type, payload, source, tenant_id, created_at, processed_at
                    FROM flex_events
                    WHERE event_type = :event_type
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                result = await db.execute(query, {
                    "event_type": event_type.value,
                    "limit": limit,
                    "offset": offset
                })
            else:
                query = text("""
                    SELECT id, event_type, payload, source, tenant_id, created_at, processed_at
                    FROM flex_events
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                result = await db.execute(query, {
                    "limit": limit,
                    "offset": offset
                })
            
            events = []
            for row in result:
                events.append(Event(
                    id=row.id,
                    event_type=row.event_type,
                    payload=row.payload,
                    source=row.source,
                    tenant_id=row.tenant_id,
                    created_at=row.created_at,
                    processed_at=row.processed_at
                ))
            
            return events
    
    async def mark_event_processed(self, event_id: str) -> None:
        """Mark an event as processed."""
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("UPDATE flex_events SET processed_at = NOW() WHERE id = :event_id"),
                {"event_id": event_id}
            )
            await db.commit()


# Global event bus instance
event_bus = EventBus()


async def publish_event(event: BaseEvent) -> None:
    """Convenience function to publish an event."""
    await event_bus.publish(event)


def subscribe_to_event(event_type: EventType, handler: Callable[[BaseEvent], Any]) -> None:
    """Convenience function to subscribe to an event."""
    event_bus.subscribe(event_type, handler)