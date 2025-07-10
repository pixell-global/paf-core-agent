"""
Health monitoring system for worker nodes.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta

from .worker_node import WorkerNode, WorkerStatus

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Health monitoring system for worker nodes."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.health_callbacks: List[Callable[[WorkerNode, bool], Any]] = []
        
        # Health check configuration
        self.unhealthy_threshold = 3  # consecutive failures
        self.recovery_threshold = 2   # consecutive successes
        self.timeout_seconds = 10
        
        # Statistics
        self.stats = {
            "checks_performed": 0,
            "healthy_workers": 0,
            "unhealthy_workers": 0,
            "recovered_workers": 0,
            "failed_workers": 0,
            "last_check": None
        }
    
    async def start(self) -> None:
        """Start the health monitoring system."""
        if self.running:
            logger.warning("Health monitor is already running")
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Health monitor started with {self.check_interval}s interval")
    
    async def stop(self) -> None:
        """Stop the health monitoring system."""
        if not self.running:
            return
        
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    def register_health_callback(self, callback: Callable[[WorkerNode, bool], Any]) -> None:
        """Register a callback for health status changes."""
        self.health_callbacks.append(callback)
        logger.debug("Registered health status callback")
    
    async def check_worker_health(self, worker: WorkerNode) -> bool:
        """Perform health check on a single worker."""
        try:
            # Update check timestamp
            worker.last_health_check = datetime.now(timezone.utc)
            
            # Perform basic connectivity check
            is_healthy = await self._perform_health_check(worker)
            
            if is_healthy:
                # Reset failure counter on success
                worker.health_check_failures = 0
                
                # Update status if worker was marked as unhealthy
                if worker.status in [WorkerStatus.ERROR, WorkerStatus.OFFLINE]:
                    old_status = worker.status
                    worker.update_status(WorkerStatus.ONLINE)
                    logger.info(f"Worker {worker.worker_id} recovered: {old_status} -> {worker.status}")
                    self.stats["recovered_workers"] += 1
                    
                    # Notify callbacks
                    await self._notify_health_callbacks(worker, True)
            else:
                # Increment failure counter
                worker.health_check_failures += 1
                
                # Mark as unhealthy if threshold exceeded
                if worker.health_check_failures >= self.unhealthy_threshold:
                    if worker.status != WorkerStatus.ERROR:
                        old_status = worker.status
                        worker.update_status(WorkerStatus.ERROR)
                        logger.warning(f"Worker {worker.worker_id} marked unhealthy after {worker.health_check_failures} failures")
                        self.stats["failed_workers"] += 1
                        
                        # Notify callbacks
                        await self._notify_health_callbacks(worker, False)
            
            # Update statistics
            self.stats["checks_performed"] += 1
            if is_healthy:
                self.stats["healthy_workers"] += 1
            else:
                self.stats["unhealthy_workers"] += 1
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Error checking health for worker {worker.worker_id}: {e}")
            worker.health_check_failures += 1
            return False
    
    async def _perform_health_check(self, worker: WorkerNode) -> bool:
        """Perform the actual health check."""
        try:
            # Check if worker has been seen recently
            time_since_last_seen = datetime.now(timezone.utc) - worker.last_seen
            if time_since_last_seen > timedelta(seconds=self.check_interval * 3):
                logger.debug(f"Worker {worker.worker_id} hasn't been seen for {time_since_last_seen}")
                return False
            
            # Check basic status
            if worker.status == WorkerStatus.MAINTENANCE:
                return True  # Maintenance is considered healthy
            
            # Check resource utilization
            if worker.metrics.cpu_usage > 95 or worker.metrics.memory_usage > 95:
                logger.debug(f"Worker {worker.worker_id} has high resource usage")
                return False
            
            # Check if worker is responding to heartbeats
            if worker.metrics.last_heartbeat:
                heartbeat_age = datetime.now(timezone.utc) - worker.metrics.last_heartbeat
                if heartbeat_age > timedelta(seconds=self.check_interval * 2):
                    logger.debug(f"Worker {worker.worker_id} heartbeat is stale: {heartbeat_age}")
                    return False
            
            # Check consecutive failures
            if worker.consecutive_failures >= 5:
                logger.debug(f"Worker {worker.worker_id} has too many consecutive failures")
                return False
            
            # Additional checks could include:
            # - Network connectivity test
            # - Service endpoint availability
            # - Custom health check endpoint
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for worker {worker.worker_id}: {e}")
            return False
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # This will be called from the worker pool with current workers
                await asyncio.sleep(self.check_interval)
                self.stats["last_check"] = datetime.now(timezone.utc)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def check_all_workers(self, workers: List[WorkerNode]) -> Dict[str, bool]:
        """Check health of all workers."""
        results = {}
        
        # Create tasks for parallel health checks
        tasks = []
        for worker in workers:
            task = asyncio.create_task(self.check_worker_health(worker))
            tasks.append((worker.worker_id, task))
        
        # Wait for all checks to complete
        for worker_id, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
                results[worker_id] = result
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for worker {worker_id}")
                results[worker_id] = False
            except Exception as e:
                logger.error(f"Health check error for worker {worker_id}: {e}")
                results[worker_id] = False
        
        return results
    
    async def _notify_health_callbacks(self, worker: WorkerNode, is_healthy: bool) -> None:
        """Notify registered callbacks about health status changes."""
        for callback in self.health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(worker, is_healthy)
                else:
                    callback(worker, is_healthy)
            except Exception as e:
                logger.error(f"Error in health callback: {e}")
    
    def get_health_summary(self, workers: List[WorkerNode]) -> Dict[str, Any]:
        """Get health summary for all workers."""
        total_workers = len(workers)
        healthy_count = sum(1 for w in workers if w.is_healthy())
        unhealthy_count = total_workers - healthy_count
        
        # Group by status
        status_counts = {}
        for worker in workers:
            status = worker.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average health score
        avg_health_score = 0.0
        if workers:
            avg_health_score = sum(w.get_health_score() for w in workers) / len(workers)
        
        return {
            "total_workers": total_workers,
            "healthy_workers": healthy_count,
            "unhealthy_workers": unhealthy_count,
            "health_percentage": (healthy_count / total_workers * 100) if total_workers > 0 else 0,
            "average_health_score": avg_health_score,
            "status_distribution": status_counts,
            "monitoring_active": self.running,
            "check_interval": self.check_interval,
            "last_check": self.stats["last_check"].isoformat() if self.stats["last_check"] else None
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health monitor statistics."""
        return {
            **self.stats,
            "last_check": self.stats["last_check"].isoformat() if self.stats["last_check"] else None,
            "running": self.running,
            "check_interval": self.check_interval,
            "thresholds": {
                "unhealthy_threshold": self.unhealthy_threshold,
                "recovery_threshold": self.recovery_threshold,
                "timeout_seconds": self.timeout_seconds
            }
        }
    
    def set_check_interval(self, interval: int) -> None:
        """Update health check interval."""
        self.check_interval = max(10, interval)  # Minimum 10 seconds
        logger.info(f"Health check interval updated to {self.check_interval} seconds")
    
    def set_thresholds(self, unhealthy_threshold: int = None, recovery_threshold: int = None) -> None:
        """Update health check thresholds."""
        if unhealthy_threshold is not None:
            self.unhealthy_threshold = max(1, unhealthy_threshold)
        
        if recovery_threshold is not None:
            self.recovery_threshold = max(1, recovery_threshold)
        
        logger.info(f"Health thresholds updated: unhealthy={self.unhealthy_threshold}, recovery={self.recovery_threshold}")
    
    def reset_statistics(self) -> None:
        """Reset health monitor statistics."""
        self.stats = {
            "checks_performed": 0,
            "healthy_workers": 0,
            "unhealthy_workers": 0,
            "recovered_workers": 0,
            "failed_workers": 0,
            "last_check": None
        }
        logger.info("Health monitor statistics reset")