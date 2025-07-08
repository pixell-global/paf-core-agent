"""Worker Agent gRPC client implementation."""

import time
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict

from app.grpc_clients.base import (
    BaseGRPCClient, 
    GRPCRequest, 
    GRPCResponse, 
    ServiceEndpoint,
    ServiceUnavailableError,
    ServiceTimeoutError,
    GRPC_AVAILABLE
)
from app.utils.logging_config import get_logger
from app.settings import Settings

logger = get_logger("worker_agent_client")


@dataclass
class TaskRequest:
    """Task execution request."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TaskResponse:
    """Task execution response."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResponse':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TaskStatus:
    """Task status information."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


class WorkerAgentClient(BaseGRPCClient):
    """gRPC client for Worker Agent communication."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, "worker_agent")
        self.active_tasks: Dict[str, TaskStatus] = {}
    
    async def call_service(self, request: GRPCRequest) -> GRPCResponse:
        """Make a gRPC call to worker agent."""
        start_time = time.time()
        
        # Select endpoint
        endpoint = self.select_endpoint()
        if not endpoint:
            raise ServiceUnavailableError("No healthy worker agent endpoints available")
        
        try:
            # For now, simulate the gRPC call since we don't have actual proto definitions
            # In a real implementation, this would use generated gRPC stubs
            
            if not GRPC_AVAILABLE:
                # Fallback to simulation when gRPC is not available
                return await self._simulate_worker_call(request, endpoint)
            
            # Get or create channel
            channel = await self.get_channel(endpoint)
            
            # Create the actual gRPC call here
            # This is a placeholder - would use actual proto-generated stubs
            response_data = await self._simulate_worker_call(request, endpoint)
            
            response_time = time.time() - start_time
            
            return GRPCResponse(
                success=True,
                data=response_data.data,
                response_time=response_time,
                service_endpoint=endpoint.address,
                metadata={"method": request.method}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(
                f"Worker agent call failed",
                method=request.method,
                endpoint=endpoint.address,
                error=str(e)
            )
            
            return GRPCResponse(
                success=False,
                data=None,
                error=str(e),
                response_time=response_time,
                service_endpoint=endpoint.address
            )
    
    async def _simulate_worker_call(self, request: GRPCRequest, endpoint: ServiceEndpoint) -> GRPCResponse:
        """Simulate worker agent call for development."""
        # Simulate processing time
        import asyncio
        await asyncio.sleep(0.1)
        
        method = request.method
        payload = request.payload
        
        if method == "execute_task":
            return await self._simulate_task_execution(payload)
        elif method == "get_task_status":
            return await self._simulate_task_status(payload)
        elif method == "list_capabilities":
            return await self._simulate_list_capabilities()
        elif method == "health_check":
            return GRPCResponse(
                success=True,
                data={"status": "healthy", "endpoint": endpoint.address}
            )
        else:
            return GRPCResponse(
                success=False,
                data=None,
                error=f"Unknown method: {method}"
            )
    
    async def _simulate_task_execution(self, payload: Dict[str, Any]) -> GRPCResponse:
        """Simulate task execution."""
        task_request = TaskRequest(**payload)
        task_type = task_request.task_type
        
        # Simulate different task types
        if task_type == "code_analysis":
            result = {
                "analysis": {
                    "complexity": "medium",
                    "issues": ["unused_variable", "long_function"],
                    "suggestions": ["refactor_function", "add_type_hints"],
                    "metrics": {
                        "lines_of_code": 234,
                        "cyclomatic_complexity": 8,
                        "maintainability_index": 72
                    }
                }
            }
        elif task_type == "file_processing":
            result = {
                "processed_files": 15,
                "total_size": "2.3MB",
                "file_types": ["py", "js", "md"],
                "processing_time": 1.2
            }
        elif task_type == "data_extraction":
            result = {
                "extracted_records": 1250,
                "schema": {
                    "fields": ["id", "name", "email", "created_at"],
                    "types": ["int", "str", "str", "datetime"]
                },
                "quality_score": 0.94
            }
        elif task_type == "computation":
            result = {
                "computation_result": 42,
                "steps_executed": 100,
                "memory_used": "128MB",
                "execution_details": {
                    "algorithm": "optimized_search",
                    "iterations": 50,
                    "convergence": True
                }
            }
        else:
            return GRPCResponse(
                success=False,
                data=None,
                error=f"Unsupported task type: {task_type}"
            )
        
        # Create task response
        task_response = TaskResponse(
            task_id=task_request.task_id,
            success=True,
            result=result,
            execution_time=1.2,
            metadata={
                "worker_id": "worker-001",
                "task_type": task_type,
                "resources_used": {
                    "cpu": "15%",
                    "memory": "64MB",
                    "disk": "2MB"
                }
            }
        )
        
        # Update task status
        self.active_tasks[task_request.task_id] = TaskStatus(
            task_id=task_request.task_id,
            status="completed",
            progress=1.0,
            started_at=time.time() - 1.2,
            completed_at=time.time()
        )
        
        return GRPCResponse(
            success=True,
            data=asdict(task_response)
        )
    
    async def _simulate_task_status(self, payload: Dict[str, Any]) -> GRPCResponse:
        """Simulate task status retrieval."""
        task_id = payload.get("task_id")
        
        if task_id in self.active_tasks:
            status = self.active_tasks[task_id]
            return GRPCResponse(
                success=True,
                data=asdict(status)
            )
        else:
            return GRPCResponse(
                success=False,
                data=None,
                error=f"Task not found: {task_id}"
            )
    
    async def _simulate_list_capabilities(self) -> GRPCResponse:
        """Simulate worker capabilities listing."""
        capabilities = {
            "supported_tasks": [
                "code_analysis",
                "file_processing", 
                "data_extraction",
                "computation",
                "text_processing",
                "image_analysis"
            ],
            "resource_limits": {
                "max_memory": "1GB",
                "max_cpu_time": "300s",
                "max_file_size": "100MB"
            },
            "features": [
                "parallel_processing",
                "streaming_responses",
                "progress_tracking",
                "error_recovery"
            ],
            "versions": {
                "worker_agent": "1.0.0",
                "python": "3.11.0",
                "grpc": "1.50.0"
            }
        }
        
        return GRPCResponse(
            success=True,
            data=capabilities
        )
    
    # High-level API methods
    
    async def execute_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TaskResponse:
        """Execute a task on a worker agent."""
        
        task_request = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            context=context,
            timeout=timeout
        )
        
        # Update task status
        self.active_tasks[task_id] = TaskStatus(
            task_id=task_id,
            status="running",
            progress=0.0,
            started_at=time.time()
        )
        
        grpc_request = GRPCRequest(
            service_type="worker_agent",
            method="execute_task",
            payload=task_request.to_dict(),
            timeout=timeout
        )
        
        try:
            response = await self.call_with_retry(grpc_request)
            
            if response.success and response.data:
                return TaskResponse.from_dict(response.data)
            else:
                # Update task status on failure
                self.active_tasks[task_id].status = "failed"
                self.active_tasks[task_id].error = response.error
                
                # Return a failed TaskResponse instead of raising exception
                return TaskResponse(
                    task_id=task_id,
                    success=False,
                    result=None,
                    error=response.error or "Task execution failed",
                    execution_time=0.0
                )
                
        except Exception as e:
            # Update task status on exception
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = "failed"
                self.active_tasks[task_id].error = str(e)
            
            # Return a failed TaskResponse instead of raising exception
            return TaskResponse(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=0.0
            )
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get status of a running task."""
        grpc_request = GRPCRequest(
            service_type="worker_agent",
            method="get_task_status",
            payload={"task_id": task_id}
        )
        
        response = await self.call_with_retry(grpc_request)
        
        if response.success:
            return TaskStatus(**response.data)
        else:
            raise ServiceUnavailableError(f"Failed to get task status: {response.error}")
    
    async def list_capabilities(self) -> Dict[str, Any]:
        """List worker agent capabilities."""
        grpc_request = GRPCRequest(
            service_type="worker_agent",
            method="list_capabilities",
            payload={}
        )
        
        response = await self.call_with_retry(grpc_request)
        
        if response.success:
            return response.data
        else:
            raise ServiceUnavailableError(f"Failed to list capabilities: {response.error}")
    
    async def health_check_worker(self, endpoint: ServiceEndpoint) -> bool:
        """Perform health check on specific worker."""
        grpc_request = GRPCRequest(
            service_type="worker_agent",
            method="health_check",
            payload={}
        )
        
        try:
            response = await self.call_service(grpc_request)
            return response.success
        except Exception:
            return False
    
    def get_active_tasks(self) -> Dict[str, TaskStatus]:
        """Get all active tasks."""
        return self.active_tasks.copy()
    
    def clear_completed_tasks(self):
        """Clear completed tasks from memory."""
        completed_tasks = [
            task_id for task_id, status in self.active_tasks.items()
            if status.status in ["completed", "failed"]
        ]
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
        
        self.logger.info(f"Cleared {len(completed_tasks)} completed tasks") 