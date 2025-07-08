"""Chat API with Server-Sent Events streaming."""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, SSEEvent, EventType, ThinkingEvent, ContentEvent, CompleteEvent
from app.core.upee_engine import UPEEEngine
from app.llm_providers import LLMProviderManager
from app.settings import get_settings, Settings

router = APIRouter()

# Global LLM manager instance
_llm_manager: LLMProviderManager = None


def get_llm_manager() -> LLMProviderManager:
    """Get or create the LLM provider manager instance."""
    global _llm_manager
    if _llm_manager is None:
        settings = get_settings()
        _llm_manager = LLMProviderManager(settings)
    return _llm_manager


async def create_sse_event(event_type: EventType, data: str, event_id: str = None) -> str:
    """Create a Server-Sent Event formatted string."""
    sse_event = SSEEvent(event=event_type, data=data, id=event_id)
    
    lines = []
    if sse_event.id:
        lines.append(f"id: {sse_event.id}")
    lines.append(f"event: {sse_event.event}")
    lines.append(f"data: {sse_event.data}")
    lines.append("")  # Empty line to end the event
    
    return "\n".join(lines)


async def upee_chat_stream(
    request: ChatRequest, 
    settings: Settings,
    grpc_manager=None
) -> AsyncGenerator[str, None]:
    """Process chat request through UPEE engine with streaming."""
    request_id = str(uuid.uuid4())
    
    try:
        # Initialize UPEE engine
        upee_engine = UPEEEngine(settings, grpc_manager)
        
        # Process through UPEE loop
        async for event in upee_engine.process_request(request, request_id):
            event_type = event.get("event")
            
            if event_type == EventType.THINKING:
                # Stream thinking events
                yield await create_sse_event(
                    EventType.THINKING,
                    event["data"],
                    event.get("id")
                )
            
            elif event_type == EventType.CONTENT:
                # Stream content events
                yield await create_sse_event(
                    EventType.CONTENT,
                    event["data"],
                    event.get("id")
                )
            
            elif event_type == EventType.COMPLETE:
                # Stream completion event
                yield await create_sse_event(
                    EventType.COMPLETE,
                    event["data"],
                    event.get("id")
                )
            
            elif event_type == EventType.ERROR:
                # Stream error event
                error_data = event.get("data", {})
                if isinstance(error_data, dict):
                    error_data = json.dumps(error_data)
                
                yield await create_sse_event(
                    EventType.ERROR,
                    error_data,
                    event.get("id")
                )
                break  # Stop streaming on error
        
        # Final DONE event
        yield await create_sse_event(EventType.DONE, "[DONE]", f"{request_id}-done")
        
    except Exception as e:
        # Handle unexpected errors
        error_data = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time(),
            "request_id": request_id
        }
        
        yield await create_sse_event(
            EventType.ERROR,
            json.dumps(error_data),
            f"{request_id}-error"
        )


@router.post("/stream")
async def stream_chat(
    chat_request: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Stream chat response using Server-Sent Events with UPEE processing."""
    
    # Basic validation
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Validate file contexts if provided
    if chat_request.files:
        for file_context in chat_request.files:
            if not file_context.path or not file_context.content:
                raise HTTPException(
                    status_code=400, 
                    detail="File context must have both path and content"
                )
    
    # Validate model if specified
    if chat_request.model:
        try:
            models = await llm_manager.get_all_models()
            available_model_names = [m["name"] for m in models if m["available"]]
            
            if chat_request.model not in available_model_names:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{chat_request.model}' not available. Available models: {available_model_names}"
                )
        except Exception as e:
            # Fallback validation if LLM manager fails
            available_models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-haiku"]
            if chat_request.model not in available_models:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{chat_request.model}' not supported. Error checking models: {str(e)}"
                )
    
    # Get gRPC manager from app state
    grpc_manager = getattr(request.app.state, 'grpc_manager', None)
    
    # Create streaming response
    return StreamingResponse(
        upee_chat_stream(chat_request, settings, grpc_manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/models")
async def list_models(
    settings: Settings = Depends(get_settings),
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """List available models and their status."""
    try:
        # Get models from LLM provider manager
        models = await llm_manager.get_all_models()
        
        # Format for API response
        formatted_models = []
        for model in models:
            formatted_models.append({
                "id": model["name"],
                "name": model["display_name"],
                "provider": model["provider"],
                "available": model["available"],
                "context_window": model.get("context_length"),
                "description": model["description"],
                "is_preferred": model.get("is_preferred", False)
            })
        
        return {
            "models": formatted_models,
            "default": settings.resolved_default_model,
            "total_available": len([m for m in formatted_models if m.get("available", False)]),
            "preferred": [m for m in formatted_models if m.get("is_preferred", False)]
        }
    
    except Exception as e:
        # Fallback to error response
        return {
            "models": [],
            "default": settings.resolved_default_model,
            "total_available": 0,
            "error": f"Failed to fetch models: {str(e)}"
        }


@router.get("/status")
async def chat_status(
    settings: Settings = Depends(get_settings),
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get chat service status and capabilities."""
    
    try:
        # Get provider health status
        provider_health = await llm_manager.health_check()
        provider_status = await llm_manager.get_available_providers()
        
        # Extract available provider names
        available_providers = [
            provider_type.value 
            for provider_type, status in provider_status.items() 
            if status.get("available", False)
        ]
        
        return {
            "service": "PAF Core Agent Chat",
            "status": "operational",
            "upee_enabled": True,
            "streaming_enabled": True,
            "available_providers": available_providers,
            "provider_health": provider_health,
            "features": {
                "thinking_events": True,
                "file_context": True,
                "multi_provider": True,
                "external_calls": True,
                "evaluation": True,
                "fallback_support": True
            },
            "limits": {
                "max_context_tokens": settings.max_context_tokens,
                "max_concurrent_requests": settings.max_concurrent_requests,
                "request_timeout": settings.request_timeout
            }
        }
    
    except Exception as e:
        return {
            "service": "PAF Core Agent Chat",
            "status": "degraded",
            "error": f"Provider health check failed: {str(e)}",
            "upee_enabled": True,
            "streaming_enabled": True,
            "available_providers": [],
            "features": {
                "thinking_events": True,
                "file_context": True,
                "multi_provider": False,
                "external_calls": True,
                "evaluation": True,
                "fallback_support": False
            }
        }


@router.get("/providers")
async def list_providers(
    llm_manager: LLMProviderManager = Depends(get_llm_manager)
):
    """Get detailed information about all LLM providers."""
    try:
        provider_status = await llm_manager.get_available_providers()
        
        # Format provider information
        providers = {}
        for provider_type, status in provider_status.items():
            providers[provider_type.value] = {
                "name": provider_type.value.title(),
                "available": status.get("available", False),
                "models": status.get("models", []),
                "health": status.get("health", {}),
                "model_count": len(status.get("models", []))
            }
        
        return {
            "providers": providers,
            "summary": {
                "total_providers": len(providers),
                "healthy_providers": len([p for p in providers.values() if p["available"]]),
                "total_models": sum(p["model_count"] for p in providers.values())
            }
        }
    
    except Exception as e:
        return {
            "providers": {},
            "error": f"Failed to fetch provider information: {str(e)}",
            "summary": {
                "total_providers": 0,
                "healthy_providers": 0,
                "total_models": 0
            }
        } 