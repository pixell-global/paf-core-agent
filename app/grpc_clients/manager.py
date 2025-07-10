"""gRPC Client Manager for coordinating all gRPC clients."""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from app.grpc_clients.base import ServiceHealth, ServiceStatus
from app.grpc_clients.worker_agent_client import WorkerAgentClient, TaskResponse, TaskStatus
from app.utils.logging_config import get_logger
from app.settings import Settings

logger = get_logger("grpc_manager")


@dataclass
class ServiceInfo:
    """Information about a gRPC service."""
    name: str
    client_type: str
    status: ServiceStatus
    health: Optional[ServiceHealth] = None
    capabilities: Optional[Dict[str, Any]] = None
    active_connections: int = 0


class GRPCClientManager:
    """Manager for all gRPC clients."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("grpc_manager")
        
        # Initialize clients
        self.worker_agent_client: Optional[WorkerAgentClient] = None
        self.clients: Dict[str, Any] = {}
        
        # Service monitoring
        self.service_info: Dict[str, ServiceInfo] = {}
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Initialize all clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all gRPC clients."""
        try:
            # Initialize Worker Agent client
            self.worker_agent_client = WorkerAgentClient(self.settings)
            self.clients["worker_agent"] = self.worker_agent_client
            
            self.service_info["worker_agent"] = ServiceInfo(
                name="Worker Agent",
                client_type="WorkerAgentClient",
                status=ServiceStatus.UNKNOWN
            )
            
            self.logger.info("Initialized gRPC clients", clients=list(self.clients.keys()))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize gRPC clients: {e}")
            raise
    
    async def startup(self):
        """Startup all gRPC clients."""
        self.logger.info("Starting up gRPC client manager")
        
        # Perform initial health checks
        await self.health_check_all_services()
        
        # Start background health monitoring
        asyncio.create_task(self._health_monitor_loop())
        
        self.logger.info("gRPC client manager started successfully")
    
    async def shutdown(self):
        """Shutdown all gRPC clients."""
        self.logger.info("Shutting down gRPC client manager")
        
        # Close all client connections
        for client_name, client in self.clients.items():
            try:
                if hasattr(client, 'close_all_channels'):
                    await client.close_all_channels()
                self.logger.debug(f"Closed {client_name} client")
            except Exception as e:
                self.logger.warning(f"Error closing {client_name} client: {e}")
        
        self.logger.info("gRPC client manager shutdown complete")
    
    async def health_check_all_services(self) -> Dict[str, ServiceHealth]:
        """Perform health checks on all services."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return {
                name: info.health for name, info in self.service_info.items()
                if info.health is not None
            }
        
        self.logger.debug("Performing health checks on all gRPC services")
        
        health_results = {}
        
        # Check all clients
        for client_name, client in self.clients.items():
            try:
                if hasattr(client, 'health_check_all'):
                    client_health = await client.health_check_all()
                    
                    # Aggregate health for this client
                    if client_health:
                        # Take the best status available
                        healthy_count = sum(
                            1 for h in client_health.values() 
                            if h.status == ServiceStatus.HEALTHY
                        )
                        
                        if healthy_count > 0:
                            overall_status = ServiceStatus.HEALTHY
                        else:
                            overall_status = ServiceStatus.UNHEALTHY
                        
                        health = ServiceHealth(
                            status=overall_status,
                            last_check=current_time,
                            metadata={
                                "endpoints": len(client_health),
                                "healthy_endpoints": healthy_count,
                                "details": client_health
                            }
                        )
                    else:
                        health = ServiceHealth(
                            status=ServiceStatus.UNKNOWN,
                            last_check=current_time,
                            error="No health data available"
                        )
                    
                    health_results[client_name] = health
                    self.service_info[client_name].health = health
                    self.service_info[client_name].status = health.status
                
            except Exception as e:
                health = ServiceHealth(
                    status=ServiceStatus.UNHEALTHY,
                    last_check=current_time,
                    error=str(e)
                )
                health_results[client_name] = health
                self.service_info[client_name].health = health
                self.service_info[client_name].status = ServiceStatus.UNHEALTHY
                
                self.logger.warning(f"Health check failed for {client_name}: {e}")
        
        self.last_health_check = current_time
        return health_results
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check_all_services()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    # Worker Agent methods
    
    async def execute_worker_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TaskResponse:
        """Execute a task on a worker agent."""
        if not self.worker_agent_client:
            raise RuntimeError("Worker agent client not initialized")
        
        self.logger.info(
            f"Executing worker task",
            task_id=task_id,
            task_type=task_type
        )
        
        try:
            response = await self.worker_agent_client.execute_task(
                task_id=task_id,
                task_type=task_type,
                payload=payload,
                context=context,
                timeout=timeout
            )
            
            self.logger.info(
                f"Worker task completed successfully",
                task_id=task_id,
                execution_time=response.execution_time
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                f"Worker task execution failed",
                task_id=task_id,
                error=str(e)
            )
            raise
    
    async def get_worker_task_status(self, task_id: str) -> TaskStatus:
        """Get status of a worker task."""
        if not self.worker_agent_client:
            raise RuntimeError("Worker agent client not initialized")
        
        return await self.worker_agent_client.get_task_status(task_id)
    
    async def get_worker_capabilities(self) -> Dict[str, Any]:
        """Get worker agent capabilities."""
        if not self.worker_agent_client:
            raise RuntimeError("Worker agent client not initialized")
        
        return await self.worker_agent_client.list_capabilities()
    
    def get_active_worker_tasks(self) -> Dict[str, TaskStatus]:
        """Get all active worker tasks."""
        if not self.worker_agent_client:
            return {}
        
        return self.worker_agent_client.get_active_tasks()
    
    def clear_completed_worker_tasks(self):
        """Clear completed worker tasks."""
        if self.worker_agent_client:
            self.worker_agent_client.clear_completed_tasks()
    
    # Service information methods
    
    def get_service_info(self) -> Dict[str, ServiceInfo]:
        """Get information about all services."""
        return self.service_info.copy()
    
    def get_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services."""
        return {
            name: info.health for name, info in self.service_info.items()
            if info.health is not None
        }

    def get_service_health_non_blocking(self) -> Dict[str, ServiceHealth]:
        """Get cached health status without triggering new checks (non-blocking)."""
        current_time = time.time()
        result = {}
        
        for name, info in self.service_info.items():
            if info.health is not None:
                result[name] = info.health
            else:
                # Provide default status for services without health info
                result[name] = ServiceHealth(
                    status=ServiceStatus.UNKNOWN,
                    last_check=current_time,
                    error="Health status not available"
                )
        
        return result
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a specific service is healthy."""
        info = self.service_info.get(service_name)
        if not info or not info.health:
            return False
        
        return info.health.status == ServiceStatus.HEALTHY
    
    def get_healthy_services(self) -> List[str]:
        """Get list of healthy service names."""
        return [
            name for name, info in self.service_info.items()
            if info.health and info.health.status == ServiceStatus.HEALTHY
        ]
    
    async def refresh_service_capabilities(self):
        """Refresh capabilities for all services."""
        for service_name, client in self.clients.items():
            try:
                if service_name == "worker_agent" and hasattr(client, 'list_capabilities'):
                    capabilities = await client.list_capabilities()
                    self.service_info[service_name].capabilities = capabilities
                    
                    self.logger.debug(
                        f"Refreshed capabilities for {service_name}",
                        capabilities=capabilities
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Failed to refresh capabilities for {service_name}: {e}"
                )
    
    # Utility methods
    
    def get_client(self, service_name: str) -> Optional[Any]:
        """Get a specific client by service name."""
        return self.clients.get(service_name)
    
    def list_clients(self) -> List[str]:
        """List all available client names."""
        return list(self.clients.keys())
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test connections to all services."""
        results = {}
        
        for service_name, client in self.clients.items():
            try:
                if hasattr(client, 'health_check_all'):
                    health_status = await client.health_check_all()
                    # Consider service connected if any endpoint is healthy
                    results[service_name] = any(
                        h.status == ServiceStatus.HEALTHY 
                        for h in health_status.values()
                    )
                else:
                    results[service_name] = False
            except Exception:
                results[service_name] = False
        
        return results 