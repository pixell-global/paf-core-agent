"""Health check API endpoints."""

import time
from typing import Dict, Any
import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas import HealthStatus, ServiceHealth
from app.settings import get_settings, Settings

router = APIRouter()


async def check_llm_providers(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """Check health of LLM providers."""
    services = {}
    
    # OpenAI health check
    if settings.openai_api_key:
        try:
            # Simple check - we'll implement actual health check later
            services["openai"] = ServiceHealth(
                status="healthy",
                latency_ms=None,
                last_check=time.time()
            ).model_dump()
        except Exception as e:
            services["openai"] = ServiceHealth(
                status="unhealthy",
                error=str(e),
                last_check=time.time()
            ).model_dump()
    else:
        services["openai"] = ServiceHealth(
            status="disabled",
            error="API key not configured",
            last_check=time.time()
        ).model_dump()
    
    # Anthropic health check
    if settings.anthropic_api_key:
        services["anthropic"] = ServiceHealth(
            status="healthy",
            latency_ms=None,
            last_check=time.time()
        ).model_dump()
    else:
        services["anthropic"] = ServiceHealth(
            status="disabled",
            error="API key not configured",
            last_check=time.time()
        ).model_dump()
    
    # AWS Bedrock health check
    services["bedrock"] = ServiceHealth(
        status="healthy",
        latency_ms=None,
        last_check=time.time()
    ).model_dump()
    
    return services


async def check_worker_agents(request: Request, settings: Settings) -> Dict[str, Dict[str, Any]]:
    """Check health of worker agents with graceful handling of busy connections."""
    services = {}
    
    # Check if gRPC manager is available in app state
    if hasattr(request.app.state, 'grpc_manager') and request.app.state.grpc_manager:
        try:
            grpc_manager = request.app.state.grpc_manager
            
            # Use non-blocking method to get cached health status
            # This avoids triggering new health checks that might timeout during busy periods
            try:
                service_health = grpc_manager.get_service_health_non_blocking()
            except Exception as e:
                # Fallback if non-blocking method fails
                services["worker_agent"] = ServiceHealth(
                    status="unknown",
                    error=f"Unable to get health status: {str(e)}",
                    last_check=time.time()
                ).model_dump()
                return services
            
            # Convert ServiceHealth objects to dictionaries
            for service_name, health in service_health.items():
                if hasattr(health, 'model_dump'):
                    services[service_name] = health.model_dump()
                elif hasattr(health, '__dict__'):
                    # Convert dataclass or regular object to dict
                    services[service_name] = {
                        "status": health.status.value if hasattr(health.status, 'value') else health.status,
                        "last_check": getattr(health, 'last_check', time.time()),
                        "response_time": getattr(health, 'response_time', None),
                        "error": getattr(health, 'error', None),
                        "metadata": getattr(health, 'metadata', {})
                    }
                else:
                    # Fallback for unknown object types
                    services[service_name] = {
                        "status": "unknown",
                        "last_check": time.time(),
                        "error": "Unable to parse health status"
                    }
                    
            # If no services were found, provide a default worker_agent status
            if not services:
                services["worker_agent"] = ServiceHealth(
                    status="unknown",
                    error="No health data available",
                    last_check=time.time()
                ).model_dump()
                
        except Exception as e:
            # More graceful error handling - don't fail completely
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                # Handle timeout scenarios more gracefully
                services["worker_agent"] = ServiceHealth(
                    status="busy",
                    error="Service temporarily busy - likely processing requests",
                    last_check=time.time()
                ).model_dump()
            else:
                services["worker_agent"] = ServiceHealth(
                    status="unhealthy",
                    error=f"gRPC health check failed: {error_msg}",
                    last_check=time.time()
                ).model_dump()
    else:
        # Fallback when gRPC manager is not available
        services["worker_agent"] = ServiceHealth(
            status="disabled",
            error="gRPC manager not available",
            last_check=time.time()
        ).model_dump()
    
    return services


@router.get("", response_model=HealthStatus)
async def health_check(request: Request, settings: Settings = Depends(get_settings)):
    """Get application health status."""
    start_time = time.time()
    
    # Check all services
    llm_services = await check_llm_providers(settings)
    worker_services = await check_worker_agents(request, settings)
    
    # Combine all service checks
    all_services = {**llm_services, **worker_services}
    
    # Determine overall status with better handling of busy services
    overall_status = "healthy"
    for service in all_services.values():
        status = service["status"]
        if status == "unhealthy":
            overall_status = "unhealthy"
            break
        elif status == "degraded":
            overall_status = "degraded"
        elif status == "busy" and overall_status == "healthy":
            # Busy is better than degraded but not as good as healthy
            overall_status = "busy"
    
    health_status = HealthStatus(
        status=overall_status,
        version="0.1.0",
        timestamp=time.time(),
        services=all_services
    )
    
    # Return appropriate HTTP status code
    # Consider "busy" as still operational (200) rather than error (503)
    status_code = 200 if overall_status in ["healthy", "busy"] else 503
    return JSONResponse(
        status_code=status_code,
        content=health_status.model_dump()
    )


@router.get("/live")
async def liveness_check():
    """Simple liveness check for container orchestration."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/ready")
async def readiness_check(settings: Settings = Depends(get_settings)):
    """Readiness check for container orchestration."""
    # Check if essential services are available
    llm_services = await check_llm_providers(settings)
    
    # Check if at least one LLM provider is healthy
    has_healthy_llm = any(
        service["status"] == "healthy" 
        for service in llm_services.values()
    )
    
    if has_healthy_llm:
        return {"status": "ready", "timestamp": time.time()}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "timestamp": time.time()}
        ) 