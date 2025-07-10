"""
Worker pool management system.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.db.database import AsyncSessionLocal
from app.db.models.workers import WorkerInstance, WorkerTask
from app.core.event_bus import event_bus
from app.core.events import BaseEvent, EventType
from .worker_node import WorkerNode, WorkerStatus, WorkerCapability, TaskAssignment
from .load_balancer import LoadBalancer, LoadBalancingStrategy
from .health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a pool of worker nodes for distributed task execution."""
    
    def __init__(self, pool_name: str = "default"):
        self.pool_name = pool_name
        self.workers: Dict[str, WorkerNode] = {}
        self.load_balancer = LoadBalancer()
        self.health_monitor = HealthMonitor()
        self.running = False
        
        # Pool configuration
        self.auto_scaling_enabled = True
        self.min_workers = 1
        self.max_workers = 10
        self.scale_up_threshold = 80  # CPU percentage
        self.scale_down_threshold = 20
        
        # Task management
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_callbacks: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "total_tasks_assigned": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "workers_added": 0,
            "workers_removed": 0,
            "scaling_events": 0,
            "last_scaling_event": None
        }
        
        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        self.auto_scale_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self.running:
            logger.warning(f"Worker pool {self.pool_name} is already running")
            return
        
        self.running = True
        
        # Start health monitor
        await self.health_monitor.start()
        
        # Register health callback
        self.health_monitor.register_health_callback(self._handle_worker_health_change)
        
        # Load existing workers from database
        await self._load_workers_from_db()
        
        # Start background tasks
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        if self.auto_scaling_enabled:
            self.auto_scale_task = asyncio.create_task(self._auto_scaling_loop())
        
        # Subscribe to events
        event_bus.subscribe(EventType.USER_PROMPT, self._handle_user_prompt)
        
        logger.info(f"Worker pool {self.pool_name} started with {len(self.workers)} workers")
    
    async def stop(self) -> None:
        """Stop the worker pool."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.maintenance_task:
            self.maintenance_task.cancel()
        if self.auto_scale_task:
            self.auto_scale_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self.maintenance_task, self.auto_scale_task]
        for task in tasks:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop health monitor
        await self.health_monitor.stop()
        
        # Save worker states to database
        await self._save_workers_to_db()
        
        logger.info(f"Worker pool {self.pool_name} stopped")
    
    async def add_worker(
        self,
        worker_id: str,
        endpoint: str,
        capabilities: List[WorkerCapability] = None,
        max_concurrent_tasks: int = 5,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a worker to the pool."""
        try:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already exists in pool")
                return False
            
            # Create worker node
            worker = WorkerNode(
                worker_id=worker_id,
                endpoint=endpoint,
                capabilities=capabilities or [],
                max_concurrent_tasks=max_concurrent_tasks,
                metadata=metadata or {}
            )
            
            # Add to pool
            self.workers[worker_id] = worker
            self.stats["workers_added"] += 1
            
            # Save to database
            await self._save_worker_to_db(worker)
            
            # Update status to online
            worker.update_status(WorkerStatus.ONLINE)
            
            logger.info(f"Added worker {worker_id} to pool {self.pool_name}")
            
            # Publish event
            await event_bus.publish(BaseEvent(
                event_type=EventType.JOB_SCHEDULED,  # Reusing for worker events
                payload={
                    "action": "worker_added",
                    "worker_id": worker_id,
                    "pool_name": self.pool_name,
                    "endpoint": endpoint
                },
                source=f"worker_pool_{self.pool_name}"
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding worker {worker_id}: {e}")
            return False
    
    async def remove_worker(self, worker_id: str, graceful: bool = True) -> bool:
        """Remove a worker from the pool."""
        try:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found in pool")
                return False
            
            worker = self.workers[worker_id]
            
            if graceful:
                # Wait for active tasks to complete
                await self._wait_for_worker_tasks(worker)
            else:
                # Force remove - reassign active tasks
                await self._reassign_worker_tasks(worker)
            
            # Remove from pool
            del self.workers[worker_id]
            self.stats["workers_removed"] += 1
            
            # Remove from database
            await self._remove_worker_from_db(worker_id)
            
            logger.info(f"Removed worker {worker_id} from pool {self.pool_name}")
            
            # Publish event
            await event_bus.publish(BaseEvent(
                event_type=EventType.JOB_SCHEDULED,  # Reusing for worker events
                payload={
                    "action": "worker_removed",
                    "worker_id": worker_id,
                    "pool_name": self.pool_name,
                    "graceful": graceful
                },
                source=f"worker_pool_{self.pool_name}"
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing worker {worker_id}: {e}")
            return False
    
    async def assign_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        requirements: Dict[str, Any] = None,
        priority: int = 0,
        timeout_seconds: int = 300,
        callback: Callable = None
    ) -> Optional[str]:
        """Assign a task to an available worker."""
        try:
            # Select worker using load balancer
            worker = self.load_balancer.select_worker(
                list(self.workers.values()),
                task_type,
                requirements,
                priority
            )
            
            if not worker:
                logger.warning(f"No available worker for task {task_id} of type {task_type}")
                return None
            
            # Create task assignment
            task_assignment = TaskAssignment(
                task_id=task_id,
                task_type=task_type,
                assigned_at=datetime.now(timezone.utc),
                deadline=datetime.now(timezone.utc).replace(microsecond=0) + 
                         timedelta(seconds=timeout_seconds) if timeout_seconds else None,
                priority=priority,
                payload=task_data
            )
            
            # Assign to worker
            if worker.assign_task(task_assignment):
                # Store task info
                self.pending_tasks[task_id] = {
                    "worker_id": worker.worker_id,
                    "task_assignment": task_assignment,
                    "assigned_at": datetime.now(timezone.utc),
                    "callback": callback
                }
                
                if callback:
                    self.task_callbacks[task_id] = callback
                
                # Update statistics
                self.stats["total_tasks_assigned"] += 1
                
                # Save to database
                await self._save_task_to_db(task_assignment, worker.worker_id)
                
                logger.info(f"Assigned task {task_id} to worker {worker.worker_id}")
                return worker.worker_id
            else:
                logger.error(f"Failed to assign task {task_id} to worker {worker.worker_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error assigning task {task_id}: {e}")
            return None
    
    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        success: bool = True,
        result: Dict[str, Any] = None,
        error_message: str = None,
        execution_time: float = 0.0
    ) -> bool:
        """Mark a task as completed."""
        try:
            # Get worker
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Worker {worker_id} not found for task completion")
                return False
            
            # Complete task on worker
            if worker.complete_task(task_id, success, execution_time):
                # Update statistics
                if success:
                    self.stats["total_tasks_completed"] += 1
                else:
                    self.stats["total_tasks_failed"] += 1
                
                # Remove from pending tasks
                task_info = self.pending_tasks.pop(task_id, None)
                
                # Call callback if registered
                if task_id in self.task_callbacks:
                    callback = self.task_callbacks.pop(task_id)
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(task_id, success, result, error_message)
                        else:
                            callback(task_id, success, result, error_message)
                    except Exception as e:
                        logger.error(f"Error in task callback: {e}")
                
                # Update database
                await self._update_task_in_db(task_id, success, result, error_message)
                
                logger.info(f"Task {task_id} completed on worker {worker_id} (success: {success})")
                return True
            else:
                logger.warning(f"Task {task_id} not found on worker {worker_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return False
    
    def get_worker(self, worker_id: str) -> Optional[WorkerNode]:
        """Get a specific worker."""
        return self.workers.get(worker_id)
    
    def get_all_workers(self) -> List[WorkerNode]:
        """Get all workers in the pool."""
        return list(self.workers.values())
    
    def get_healthy_workers(self) -> List[WorkerNode]:
        """Get all healthy workers."""
        return [worker for worker in self.workers.values() if worker.is_healthy()]
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status."""
        workers = list(self.workers.values())
        healthy_workers = self.get_healthy_workers()
        
        # Calculate load statistics
        total_capacity = sum(w.max_concurrent_tasks for w in workers)
        total_active_tasks = sum(len(w.active_tasks) for w in workers)
        avg_load = (total_active_tasks / total_capacity * 100) if total_capacity > 0 else 0
        
        return {
            "pool_name": self.pool_name,
            "running": self.running,
            "total_workers": len(workers),
            "healthy_workers": len(healthy_workers),
            "total_capacity": total_capacity,
            "active_tasks": total_active_tasks,
            "pending_tasks": len(self.pending_tasks),
            "average_load": avg_load,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "load_balancer_strategy": self.load_balancer.strategy.value,
            "health_summary": self.health_monitor.get_health_summary(workers),
            "statistics": self.stats.copy()
        }
    
    async def _handle_worker_health_change(self, worker: WorkerNode, is_healthy: bool) -> None:
        """Handle worker health status changes."""
        if not is_healthy:
            logger.warning(f"Worker {worker.worker_id} became unhealthy")
            # Reassign tasks if worker is unhealthy
            await self._reassign_worker_tasks(worker)
        else:
            logger.info(f"Worker {worker.worker_id} recovered")
    
    async def _handle_user_prompt(self, event: BaseEvent) -> None:
        """Handle user prompt events - could trigger task creation."""
        # This is where you might analyze the prompt and create tasks
        logger.debug(f"Worker pool received user prompt event: {event.payload}")
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance tasks."""
        while self.running:
            try:
                # Perform health checks
                await self.health_monitor.check_all_workers(list(self.workers.values()))
                
                # Clean up completed tasks
                await self._cleanup_old_tasks()
                
                # Check for rebalancing needs
                if self.load_balancer.rebalance_needed(list(self.workers.values())):
                    suggestions = self.load_balancer.suggest_rebalancing(list(self.workers.values()))
                    logger.info(f"Load rebalancing suggested: {len(suggestions)} task movements")
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(30)
    
    async def _auto_scaling_loop(self) -> None:
        """Background auto-scaling logic."""
        while self.running:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_scaling_needs(self) -> None:
        """Check if pool needs to be scaled up or down."""
        healthy_workers = self.get_healthy_workers()
        
        if not healthy_workers:
            return
        
        # Calculate average load
        avg_load = sum(w.get_current_load() for w in healthy_workers) / len(healthy_workers)
        
        # Scale up if load is high and we're under max workers
        if (avg_load > self.scale_up_threshold and 
            len(self.workers) < self.max_workers):
            await self._scale_up()
        
        # Scale down if load is low and we're over min workers
        elif (avg_load < self.scale_down_threshold and 
              len(self.workers) > self.min_workers):
            await self._scale_down()
    
    async def _scale_up(self) -> None:
        """Add workers to the pool."""
        # In a real implementation, this would integrate with container orchestration
        logger.info(f"Scaling up worker pool {self.pool_name}")
        self.stats["scaling_events"] += 1
        self.stats["last_scaling_event"] = datetime.now(timezone.utc)
    
    async def _scale_down(self) -> None:
        """Remove workers from the pool."""
        # In a real implementation, this would gracefully remove least utilized workers
        logger.info(f"Scaling down worker pool {self.pool_name}")
        self.stats["scaling_events"] += 1
        self.stats["last_scaling_event"] = datetime.now(timezone.utc)
    
    async def _wait_for_worker_tasks(self, worker: WorkerNode, timeout: int = 300) -> None:
        """Wait for worker to complete all active tasks."""
        start_time = datetime.now(timezone.utc)
        
        while worker.active_tasks and self.running:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for worker {worker.worker_id} tasks to complete")
                break
            
            await asyncio.sleep(5)
    
    async def _reassign_worker_tasks(self, worker: WorkerNode) -> None:
        """Reassign active tasks from a worker to other workers."""
        for task_id, task_assignment in worker.active_tasks.items():
            logger.info(f"Reassigning task {task_id} from unhealthy worker {worker.worker_id}")
            
            # Try to assign to another worker
            new_worker_id = await self.assign_task(
                task_id=f"{task_id}_reassigned",
                task_type=task_assignment.task_type,
                task_data=task_assignment.payload,
                priority=task_assignment.priority
            )
            
            if new_worker_id:
                logger.info(f"Task {task_id} reassigned to worker {new_worker_id}")
            else:
                logger.error(f"Failed to reassign task {task_id}")
        
        # Clear active tasks from the unhealthy worker
        worker.active_tasks.clear()
    
    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks from memory."""
        # Remove tasks older than 1 hour
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        
        to_remove = []
        for task_id, task_info in self.pending_tasks.items():
            if task_info["assigned_at"] < cutoff:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            self.pending_tasks.pop(task_id, None)
            self.task_callbacks.pop(task_id, None)
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")
    
    # Database operations
    async def _load_workers_from_db(self) -> None:
        """Load worker instances from database."""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(WorkerInstance).where(WorkerInstance.pool_name == self.pool_name)
                )
                db_workers = result.scalars().all()
                
                for db_worker in db_workers:
                    capabilities = [
                        WorkerCapability(**cap) for cap in db_worker.capabilities
                    ]
                    
                    worker = WorkerNode(
                        worker_id=db_worker.worker_id,
                        endpoint=db_worker.endpoint,
                        capabilities=capabilities,
                        max_concurrent_tasks=db_worker.max_concurrent_tasks,
                        metadata=db_worker.meta_data or {}
                    )
                    
                    # Set status from database
                    worker.update_status(WorkerStatus(db_worker.status))
                    worker.last_seen = db_worker.last_seen or datetime.now(timezone.utc)
                    
                    self.workers[worker.worker_id] = worker
                
                logger.info(f"Loaded {len(db_workers)} workers from database")
                
        except Exception as e:
            logger.error(f"Error loading workers from database: {e}")
    
    async def _save_worker_to_db(self, worker: WorkerNode) -> None:
        """Save worker to database."""
        try:
            async with AsyncSessionLocal() as db:
                db_worker = WorkerInstance(
                    worker_id=worker.worker_id,
                    pool_name=self.pool_name,
                    endpoint=worker.endpoint,
                    status=worker.status.value,
                    capabilities=[cap.model_dump() for cap in worker.capabilities],
                    max_concurrent_tasks=worker.max_concurrent_tasks,
                    meta_data=worker.metadata,
                    last_seen=worker.last_seen
                )
                
                db.add(db_worker)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error saving worker to database: {e}")
    
    async def _save_workers_to_db(self) -> None:
        """Save all workers to database."""
        for worker in self.workers.values():
            await self._save_worker_to_db(worker)
    
    async def _remove_worker_from_db(self, worker_id: str) -> None:
        """Remove worker from database."""
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(WorkerInstance)
                    .where(WorkerInstance.worker_id == worker_id)
                    .values(status="removed")
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error removing worker from database: {e}")
    
    async def _save_task_to_db(self, task_assignment: TaskAssignment, worker_id: str) -> None:
        """Save task assignment to database."""
        try:
            async with AsyncSessionLocal() as db:
                db_task = WorkerTask(
                    task_id=task_assignment.task_id,
                    worker_id=worker_id,
                    task_type=task_assignment.task_type,
                    payload=task_assignment.payload,
                    priority=task_assignment.priority,
                    assigned_at=task_assignment.assigned_at,
                    deadline=task_assignment.deadline,
                    status="assigned"
                )
                
                db.add(db_task)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error saving task to database: {e}")
    
    async def _update_task_in_db(
        self,
        task_id: str,
        success: bool,
        result: Dict[str, Any] = None,
        error_message: str = None
    ) -> None:
        """Update task status in database."""
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(WorkerTask)
                    .where(WorkerTask.task_id == task_id)
                    .values(
                        status="completed" if success else "failed",
                        completed_at=datetime.now(timezone.utc),
                        result=result,
                        error_message=error_message
                    )
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating task in database: {e}")