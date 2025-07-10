"""
Load balancer for distributing tasks across workers.
"""

import logging
import random
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from .worker_node import WorkerNode, WorkerStatus, TaskAssignment

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_BASED = "health_based"
    RANDOM = "random"
    CAPABILITY_AWARE = "capability_aware"


class LoadBalancer:
    """Load balancer for worker task distribution."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.round_robin_index = 0
        self.statistics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "strategy_switches": 0,
            "last_assignment": None
        }
    
    def select_worker(
        self,
        workers: List[WorkerNode],
        task_type: str,
        requirements: Dict[str, Any] = None,
        priority: int = 0
    ) -> Optional[WorkerNode]:
        """Select the best worker for a task."""
        
        # Filter available workers
        available_workers = self._filter_available_workers(workers, task_type, requirements)
        
        if not available_workers:
            logger.warning(f"No available workers for task type {task_type}")
            self.statistics["failed_assignments"] += 1
            return None
        
        # Select worker based on strategy
        selected_worker = self._apply_strategy(available_workers, priority)
        
        if selected_worker:
            self.statistics["total_assignments"] += 1
            self.statistics["successful_assignments"] += 1
            self.statistics["last_assignment"] = datetime.now(timezone.utc)
            
            logger.debug(f"Selected worker {selected_worker.worker_id} using {self.strategy.value} strategy")
        else:
            self.statistics["failed_assignments"] += 1
        
        return selected_worker
    
    def _filter_available_workers(
        self,
        workers: List[WorkerNode],
        task_type: str,
        requirements: Dict[str, Any] = None
    ) -> List[WorkerNode]:
        """Filter workers that can handle the task."""
        
        available_workers = []
        
        for worker in workers:
            if worker.can_handle_task(task_type, requirements):
                available_workers.append(worker)
        
        return available_workers
    
    def _apply_strategy(self, workers: List[WorkerNode], priority: int = 0) -> Optional[WorkerNode]:
        """Apply the selected load balancing strategy."""
        
        if not workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(workers)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(workers)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(workers)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(workers)
        
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based(workers)
        
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(workers)
        
        elif self.strategy == LoadBalancingStrategy.CAPABILITY_AWARE:
            return self._capability_aware(workers, priority)
        
        else:
            logger.warning(f"Unknown strategy {self.strategy}, falling back to round robin")
            return self._round_robin(workers)
    
    def _round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round robin selection."""
        if not workers:
            return None
        
        selected_worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index = (self.round_robin_index + 1) % len(workers)
        
        return selected_worker
    
    def _least_connections(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: len(w.active_tasks))
    
    def _weighted_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin based on max capacity."""
        if not workers:
            return None
        
        # Create weighted list based on max concurrent tasks
        weighted_workers = []
        for worker in workers:
            weight = max(1, worker.max_concurrent_tasks)
            weighted_workers.extend([worker] * weight)
        
        selected_worker = weighted_workers[self.round_robin_index % len(weighted_workers)]
        self.round_robin_index = (self.round_robin_index + 1) % len(weighted_workers)
        
        return selected_worker
    
    def _least_response_time(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with lowest average response time."""
        return min(workers, key=lambda w: w.average_response_time)
    
    def _health_based(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with highest health score."""
        return max(workers, key=lambda w: w.get_health_score())
    
    def _random_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Random worker selection."""
        return random.choice(workers)
    
    def _capability_aware(self, workers: List[WorkerNode], priority: int = 0) -> WorkerNode:
        """Select worker based on capability matching and priority."""
        # Score workers based on multiple factors
        scored_workers = []
        
        for worker in workers:
            score = self._calculate_worker_score(worker, priority)
            scored_workers.append((worker, score))
        
        # Sort by score (highest first)
        scored_workers.sort(key=lambda x: x[1], reverse=True)
        
        return scored_workers[0][0] if scored_workers else None
    
    def _calculate_worker_score(self, worker: WorkerNode, priority: int = 0) -> float:
        """Calculate a comprehensive score for worker selection."""
        score = 0.0
        
        # Health score (0-100)
        health_score = worker.get_health_score()
        score += health_score * 0.3
        
        # Load factor (prefer less loaded workers)
        load_factor = 100 - worker.get_current_load()
        score += load_factor * 0.25
        
        # Response time factor (prefer faster workers)
        if worker.average_response_time > 0:
            response_factor = max(0, 100 - (worker.average_response_time / 10))
            score += response_factor * 0.2
        else:
            score += 50  # Default for workers with no history
        
        # Success rate factor
        success_factor = worker.success_rate * 100
        score += success_factor * 0.15
        
        # Capacity factor (prefer workers with more capacity)
        capacity_factor = min(100, worker.max_concurrent_tasks * 10)
        score += capacity_factor * 0.1
        
        # Priority adjustment
        if priority > 5:  # High priority tasks
            # Prefer workers with more capacity and better health
            score += (health_score * 0.1) + (capacity_factor * 0.1)
        
        return score
    
    def rebalance_needed(self, workers: List[WorkerNode]) -> bool:
        """Check if load rebalancing is needed."""
        if len(workers) < 2:
            return False
        
        # Calculate load distribution
        loads = [worker.get_current_load() for worker in workers if worker.is_healthy()]
        
        if not loads:
            return False
        
        avg_load = sum(loads) / len(loads)
        max_load = max(loads)
        min_load = min(loads)
        
        # Rebalance if there's significant load imbalance
        load_difference = max_load - min_load
        return load_difference > 30 and max_load > 70
    
    def suggest_rebalancing(self, workers: List[WorkerNode]) -> List[Tuple[str, str]]:
        """Suggest task movements for rebalancing."""
        suggestions = []
        
        if not self.rebalance_needed(workers):
            return suggestions
        
        # Sort workers by load
        healthy_workers = [w for w in workers if w.is_healthy()]
        sorted_workers = sorted(healthy_workers, key=lambda w: w.get_current_load())
        
        overloaded_workers = [w for w in sorted_workers if w.get_current_load() > 80]
        underloaded_workers = [w for w in sorted_workers if w.get_current_load() < 50]
        
        # Suggest moving tasks from overloaded to underloaded workers
        for overloaded in overloaded_workers:
            for underloaded in underloaded_workers:
                if len(overloaded.active_tasks) > 0 and len(underloaded.active_tasks) < underloaded.max_concurrent_tasks:
                    # Pick a task to move (prefer lower priority tasks)
                    task_id = next(iter(overloaded.active_tasks.keys()))
                    suggestions.append((task_id, underloaded.worker_id))
                    break
        
        return suggestions
    
    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Change the load balancing strategy."""
        if strategy != self.strategy:
            logger.info(f"Changing load balancing strategy from {self.strategy.value} to {strategy.value}")
            self.strategy = strategy
            self.statistics["strategy_switches"] += 1
            self.round_robin_index = 0  # Reset round robin counter
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_assignments = self.statistics["total_assignments"]
        success_rate = 0.0
        if total_assignments > 0:
            success_rate = self.statistics["successful_assignments"] / total_assignments
        
        return {
            "strategy": self.strategy.value,
            "total_assignments": total_assignments,
            "successful_assignments": self.statistics["successful_assignments"],
            "failed_assignments": self.statistics["failed_assignments"],
            "success_rate": success_rate,
            "strategy_switches": self.statistics["strategy_switches"],
            "last_assignment": self.statistics["last_assignment"].isoformat() if self.statistics["last_assignment"] else None
        }
    
    def reset_statistics(self) -> None:
        """Reset load balancer statistics."""
        self.statistics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "strategy_switches": 0,
            "last_assignment": None
        }
        logger.info("Load balancer statistics reset")