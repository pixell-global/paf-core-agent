"""
Worker node representation and management.
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from pydantic import BaseModel
import uuid

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    """Worker node status."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    ONLINE = "online"
    BUSY = "busy"
    IDLE = "idle"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class WorkerCapability(BaseModel):
    """Worker capability definition."""
    name: str
    version: str
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class WorkerMetrics(BaseModel):
    """Worker performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    task_queue_size: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskAssignment(BaseModel):
    """Task assignment to worker."""
    task_id: str
    task_type: str
    assigned_at: datetime
    deadline: Optional[datetime] = None
    priority: int = 0
    payload: Dict[str, Any] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkerNode:
    """Represents a worker node in the pool."""
    
    def __init__(
        self,
        worker_id: str,
        endpoint: str,
        capabilities: List[WorkerCapability] = None,
        max_concurrent_tasks: int = 5,
        metadata: Dict[str, Any] = None
    ):
        self.worker_id = worker_id
        self.endpoint = endpoint
        self.capabilities = capabilities or []
        self.max_concurrent_tasks = max_concurrent_tasks
        self.metadata = metadata or {}
        
        # Status tracking
        self.status = WorkerStatus.UNKNOWN
        self.last_seen = datetime.now(timezone.utc)
        self.connected_at: Optional[datetime] = None
        self.error_count = 0
        self.consecutive_failures = 0
        
        # Task management
        self.active_tasks: Dict[str, TaskAssignment] = {}
        self.completed_tasks_count = 0
        self.failed_tasks_count = 0
        
        # Performance tracking
        self.metrics = WorkerMetrics()
        self.average_response_time = 0.0
        self.success_rate = 1.0
        
        # Health monitoring
        self.health_check_failures = 0
        self.last_health_check: Optional[datetime] = None
        self.health_check_interval = 30  # seconds
        
        # Connection management
        self.connection_retries = 0
        self.max_retries = 3
        self.retry_delay = 5.0  # seconds
    
    def update_status(self, status: WorkerStatus) -> None:
        """Update worker status."""
        if self.status != status:
            previous_status = self.status
            self.status = status
            self.last_seen = datetime.now(timezone.utc)
            
            # Handle status transitions
            if status == WorkerStatus.ONLINE and previous_status != WorkerStatus.ONLINE:
                self.connected_at = datetime.now(timezone.utc)
                self.connection_retries = 0
                self.consecutive_failures = 0
            elif status == WorkerStatus.ERROR:
                self.error_count += 1
                self.consecutive_failures += 1
            
            logger.info(f"Worker {self.worker_id} status changed: {previous_status} -> {status}")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update worker metrics."""
        try:
            self.metrics = WorkerMetrics(**metrics)
            self.last_seen = datetime.now(timezone.utc)
            
            # Update calculated metrics
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks > 0:
                self.success_rate = self.metrics.tasks_completed / total_tasks
            
            # Determine status based on metrics
            self._update_status_from_metrics()
            
        except Exception as e:
            logger.error(f"Error updating metrics for worker {self.worker_id}: {e}")
    
    def _update_status_from_metrics(self) -> None:
        """Update status based on current metrics."""
        if self.status == WorkerStatus.OFFLINE:
            return
        
        current_load = self.get_current_load()
        
        if current_load >= 100:
            self.update_status(WorkerStatus.OVERLOADED)
        elif current_load >= 80:
            self.update_status(WorkerStatus.BUSY)
        elif current_load > 0:
            self.update_status(WorkerStatus.BUSY)
        else:
            self.update_status(WorkerStatus.IDLE)
    
    def assign_task(self, task_assignment: TaskAssignment) -> bool:
        """Assign a task to this worker."""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"Worker {self.worker_id} at capacity, cannot assign task {task_assignment.task_id}")
            return False
        
        if self.status not in [WorkerStatus.ONLINE, WorkerStatus.IDLE, WorkerStatus.BUSY]:
            logger.warning(f"Worker {self.worker_id} not available (status: {self.status})")
            return False
        
        self.active_tasks[task_assignment.task_id] = task_assignment
        self.update_status(WorkerStatus.BUSY)
        
        logger.info(f"Assigned task {task_assignment.task_id} to worker {self.worker_id}")
        return True
    
    def complete_task(self, task_id: str, success: bool = True, execution_time: float = 0.0) -> bool:
        """Mark a task as completed."""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found in active tasks for worker {self.worker_id}")
            return False
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Update counters
        if success:
            self.completed_tasks_count += 1
            self.consecutive_failures = 0
        else:
            self.failed_tasks_count += 1
            self.consecutive_failures += 1
        
        # Update average response time
        if execution_time > 0:
            total_completed = self.completed_tasks_count + self.failed_tasks_count
            if total_completed == 1:
                self.average_response_time = execution_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.average_response_time = (alpha * execution_time + 
                                            (1 - alpha) * self.average_response_time)
        
        # Update status
        self._update_status_from_metrics()
        
        logger.info(f"Task {task_id} completed on worker {self.worker_id} (success: {success})")
        return True
    
    def can_handle_task(self, task_type: str, requirements: Dict[str, Any] = None) -> bool:
        """Check if worker can handle a specific task type."""
        # Check if worker is available
        if self.status not in [WorkerStatus.ONLINE, WorkerStatus.IDLE, WorkerStatus.BUSY]:
            return False
        
        # Check capacity
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check capabilities
        has_capability = any(
            cap.name == task_type for cap in self.capabilities
        )
        
        if not has_capability:
            return False
        
        # Check specific requirements if provided
        if requirements:
            for requirement, value in requirements.items():
                if not self._meets_requirement(requirement, value):
                    return False
        
        return True
    
    def _meets_requirement(self, requirement: str, value: Any) -> bool:
        """Check if worker meets a specific requirement."""
        if requirement == "min_memory":
            # Check if worker has enough available memory
            available_memory = 100 - self.metrics.memory_usage
            return available_memory >= value
        
        elif requirement == "max_latency":
            return self.metrics.network_latency <= value
        
        elif requirement == "capability_version":
            # Check if worker has capability with minimum version
            capability_name, min_version = value
            for cap in self.capabilities:
                if cap.name == capability_name:
                    # Simple version comparison
                    return cap.version >= min_version
            return False
        
        elif requirement in self.metadata:
            return self.metadata[requirement] == value
        
        return True
    
    def get_current_load(self) -> float:
        """Get current load percentage (0-100)."""
        if self.max_concurrent_tasks == 0:
            return 0.0
        
        return (len(self.active_tasks) / self.max_concurrent_tasks) * 100
    
    def get_health_score(self) -> float:
        """Calculate worker health score (0-100)."""
        score = 100.0
        
        # Penalize for errors
        if self.consecutive_failures > 0:
            score -= min(self.consecutive_failures * 10, 50)
        
        # Penalize for high resource usage
        if self.metrics.cpu_usage > 90:
            score -= 20
        if self.metrics.memory_usage > 90:
            score -= 20
        
        # Penalize for high latency
        if self.metrics.network_latency > 1000:  # 1 second
            score -= 15
        
        # Penalize for being offline too long
        if self.status == WorkerStatus.OFFLINE:
            offline_minutes = (datetime.now(timezone.utc) - self.last_seen).total_seconds() / 60
            score -= min(offline_minutes * 2, 50)
        
        return max(score, 0.0)
    
    def is_healthy(self) -> bool:
        """Check if worker is considered healthy."""
        return (
            self.status in [WorkerStatus.ONLINE, WorkerStatus.IDLE, WorkerStatus.BUSY] and
            self.consecutive_failures < 3 and
            self.get_health_score() > 50
        )
    
    def should_retry_connection(self) -> bool:
        """Check if connection should be retried."""
        return (
            self.status in [WorkerStatus.ERROR, WorkerStatus.OFFLINE] and
            self.connection_retries < self.max_retries
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive worker information."""
        return {
            "worker_id": self.worker_id,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "capabilities": [cap.model_dump() for cap in self.capabilities],
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_load": self.get_current_load(),
            "health_score": self.get_health_score(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks_count,
            "failed_tasks": self.failed_tasks_count,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "last_seen": self.last_seen.isoformat(),
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "metrics": self.metrics.model_dump(),
            "metadata": self.metadata,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures
        }
    
    def __str__(self) -> str:
        return f"WorkerNode({self.worker_id}, {self.status.value}, {self.endpoint})"
    
    def __repr__(self) -> str:
        return f"<WorkerNode: {self.worker_id} ({self.status.value})>"