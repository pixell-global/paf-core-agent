"""Worker Agent API endpoints for gRPC task management."""

import uuid
import time
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.grpc_clients.worker_agent_client import TaskResponse, TaskStatus
from app.grpc_clients.base import ServiceUnavailableError, ServiceTimeoutError
from app.utils.logging_config import get_logger

logger = get_logger("worker_api")

router = APIRouter()


# Request/Response Models

class TaskExecutionRequest(BaseModel):
    """Request model for task execution."""
    task_type: str = Field(..., description="Type of task to execute")
    payload: Dict[str, Any] = Field(..., description="Task payload")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    timeout: Optional[float] = Field(default=30.0, description="Task timeout in seconds")


class TaskExecutionResponse(BaseModel):
    """Response model for task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    success: bool = Field(..., description="Whether task execution was successful")
    result: Optional[Any] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Task execution time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    progress: float = Field(..., description="Task progress (0.0 to 1.0)")
    started_at: Optional[float] = Field(default=None, description="Task start timestamp")
    completed_at: Optional[float] = Field(default=None, description="Task completion timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class WorkerCapabilitiesResponse(BaseModel):
    """Response model for worker capabilities."""
    supported_tasks: List[str] = Field(..., description="List of supported task types")
    resource_limits: Dict[str, str] = Field(..., description="Resource limitations")
    features: List[str] = Field(..., description="Supported features")
    versions: Dict[str, str] = Field(..., description="Component versions")


class ActiveTasksResponse(BaseModel):
    """Response model for active tasks listing."""
    tasks: Dict[str, TaskStatusResponse] = Field(..., description="Active tasks by ID")
    total_count: int = Field(..., description="Total number of active tasks")


# API Endpoints

@router.post("/execute", response_model=TaskExecutionResponse)
async def execute_task(
    request: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """Execute a task on a worker agent."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        raise HTTPException(
            status_code=503,
            detail="Worker agent functionality is not available (gRPC manager not initialized)"
        )
    
    grpc_manager = http_request.app.state.grpc_manager
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    logger.info(
        f"Executing worker task",
        task_id=task_id,
        task_type=request.task_type
    )
    
    try:
        # Execute task
        response = await grpc_manager.execute_worker_task(
            task_id=task_id,
            task_type=request.task_type,
            payload=request.payload,
            context=request.context,
            timeout=request.timeout
        )
        
        # Schedule cleanup of completed task
        background_tasks.add_task(
            _cleanup_completed_task_delayed,
            grpc_manager,
            task_id,
            delay=300  # 5 minutes
        )
        
        return TaskExecutionResponse(
            task_id=response.task_id,
            success=response.success,
            result=response.result,
            error=response.error,
            execution_time=response.execution_time,
            metadata=response.metadata
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Worker service unavailable for task {task_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    
    except ServiceTimeoutError as e:
        logger.error(f"Worker task timeout for task {task_id}: {e}")
        raise HTTPException(status_code=408, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error executing task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, http_request: Request):
    """Get status of a specific task."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        raise HTTPException(
            status_code=503,
            detail="Worker agent functionality is not available"
        )
    
    grpc_manager = http_request.app.state.grpc_manager
    
    try:
        status = await grpc_manager.get_worker_task_status(task_id)
        
        return TaskStatusResponse(
            task_id=status.task_id,
            status=status.status,
            progress=status.progress,
            started_at=status.started_at,
            completed_at=status.completed_at,
            error=status.error
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error getting task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tasks", response_model=ActiveTasksResponse)
async def list_active_tasks(http_request: Request):
    """List all active tasks."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        raise HTTPException(
            status_code=503,
            detail="Worker agent functionality is not available"
        )
    
    grpc_manager = http_request.app.state.grpc_manager
    
    try:
        active_tasks = grpc_manager.get_active_worker_tasks()
        
        # Convert to response format
        tasks_response = {}
        for task_id, status in active_tasks.items():
            tasks_response[task_id] = TaskStatusResponse(
                task_id=status.task_id,
                status=status.status,
                progress=status.progress,
                started_at=status.started_at,
                completed_at=status.completed_at,
                error=status.error
            )
        
        return ActiveTasksResponse(
            tasks=tasks_response,
            total_count=len(active_tasks)
        )
        
    except Exception as e:
        logger.error(f"Error listing active tasks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/capabilities", response_model=WorkerCapabilitiesResponse)
async def get_worker_capabilities(http_request: Request):
    """Get worker agent capabilities."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        raise HTTPException(
            status_code=503,
            detail="Worker agent functionality is not available"
        )
    
    grpc_manager = http_request.app.state.grpc_manager
    
    try:
        capabilities = await grpc_manager.get_worker_capabilities()
        
        return WorkerCapabilitiesResponse(
            supported_tasks=capabilities.get("supported_tasks", []),
            resource_limits=capabilities.get("resource_limits", {}),
            features=capabilities.get("features", []),
            versions=capabilities.get("versions", {})
        )
        
    except ServiceUnavailableError as e:
        logger.error(f"Failed to get worker capabilities: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error getting worker capabilities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tasks/cleanup")
async def cleanup_completed_tasks(http_request: Request):
    """Manually trigger cleanup of completed tasks."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        raise HTTPException(
            status_code=503,
            detail="Worker agent functionality is not available"
        )
    
    grpc_manager = http_request.app.state.grpc_manager
    
    try:
        # Get task count before cleanup
        before_count = len(grpc_manager.get_active_worker_tasks())
        
        # Cleanup completed tasks
        grpc_manager.clear_completed_worker_tasks()
        
        # Get task count after cleanup
        after_count = len(grpc_manager.get_active_worker_tasks())
        
        cleaned_count = before_count - after_count
        
        logger.info(f"Cleaned up {cleaned_count} completed tasks")
        
        return {
            "success": True,
            "cleaned_tasks": cleaned_count,
            "remaining_tasks": after_count
        }
        
    except Exception as e:
        logger.error(f"Error during task cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def worker_health_check(http_request: Request):
    """Check health of worker agent connections."""
    
    # Check if gRPC manager is available
    if not hasattr(http_request.app.state, 'grpc_manager') or not http_request.app.state.grpc_manager:
        return {
            "status": "unavailable",
            "message": "gRPC manager not initialized",
            "services": {}
        }
    
    grpc_manager = http_request.app.state.grpc_manager
    
    try:
        # Get service health information
        service_health = grpc_manager.get_service_health()
        healthy_services = grpc_manager.get_healthy_services()
        
        # Test connections
        connection_tests = await grpc_manager.test_all_connections()
        
        overall_status = "healthy" if "worker_agent" in healthy_services else "unhealthy"
        
        return {
            "status": overall_status,
            "message": f"Worker agents: {len(healthy_services)} healthy",
            "services": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check,
                    "response_time": health.response_time,
                    "error": health.error,
                    "connected": connection_tests.get(name, False)
                }
                for name, health in service_health.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking worker health: {e}")
        return {
            "status": "error",
            "message": str(e),
            "services": {}
        }


# Background task helpers

async def _cleanup_completed_task_delayed(grpc_manager, task_id: str, delay: int = 300):
    """Clean up a specific completed task after delay."""
    import asyncio
    
    # Wait for the specified delay
    await asyncio.sleep(delay)
    
    try:
        # Check if task still exists and is completed
        active_tasks = grpc_manager.get_active_worker_tasks()
        if task_id in active_tasks:
            task_status = active_tasks[task_id]
            if task_status.status in ["completed", "failed"]:
                # Remove from active tasks
                if hasattr(grpc_manager.worker_agent_client, 'active_tasks'):
                    grpc_manager.worker_agent_client.active_tasks.pop(task_id, None)
                    logger.debug(f"Cleaned up completed task: {task_id}")
    
    except Exception as e:
        logger.warning(f"Error cleaning up task {task_id}: {e}") 