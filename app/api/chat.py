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
        
        # A2A 서버로부터 에이전트 카드 가져오기 (A2A 표준 준수)
        if settings.a2a_enabled:
            yield await create_sse_event(
                EventType.THINKING,
                json.dumps({
                    "phase": "init",
                    "content": "A2A 서버에서 에이전트를 탐색하는 중...",
                    "timestamp": time.time()
                }),
                f"{request_id}-a2a-discover"
            )
            
            # A2A 표준에 따른 에이전트 탐색
            agents = await upee_engine.a2a_client.discover_agents()
            if agents:
                agent_info = []
                for agent in agents:
                    info = {
                        "name": agent.get("name", "Unknown Agent"),
                        "description": agent.get("description", ""),
                        "version": agent.get("version", ""),
                        "url": agent.get("url", ""),
                        "capabilities": agent.get("capabilities", {}),
                        "skills": agent.get("skills", [])
                    }
                    agent_info.append(info)
                
                yield await create_sse_event(
                    EventType.CONTENT,
                    json.dumps({
                        "type": "a2a_agents",
                        "agents": agent_info,
                        "count": len(agents),
                        "timestamp": time.time()
                    }),
                    f"{request_id}-a2a-agents"
                )
                upee_engine.logger.info(f"Discovered {len(agents)} A2A agents for request {request_id}")
            else:
                yield await create_sse_event(
                    EventType.THINKING,
                    json.dumps({
                        "phase": "init",
                        "content": "A2A 서버에서 에이전트를 찾을 수 없습니다. 기본 모드로 진행합니다.",
                        "timestamp": time.time()
                    }),
                    f"{request_id}-a2a-fallback"
                )
        
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
        for i, file_item in enumerate(chat_request.files):
            # Handle both legacy FileContext and new FileContent formats
            if hasattr(file_item, 'path') and hasattr(file_item, 'content'):
                # Legacy FileContext
                if not file_item.path or not file_item.content:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File context {i} must have both path and content"
                    )
            elif hasattr(file_item, 'file_name'):
                # New FileContent format
                if not file_item.file_name:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {i} must have a file_name"
                    )
                if not file_item.content and not file_item.signed_url:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {i} must have either content or signed_url"
                    )
                # Validate file size
                if file_item.file_size > 100 * 1024 * 1024:  # 100MB limit
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {i} size ({file_item.file_size} bytes) exceeds 100MB limit"
                    )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file format at index {i}"
                )
    
    # Validate conversation history if provided
    if chat_request.history:
        if len(chat_request.history) > 50:
            raise HTTPException(
                status_code=400, 
                detail="Conversation history cannot exceed 50 messages"
            )
        
        for i, message in enumerate(chat_request.history):
            if not message.role or message.role not in ['user', 'assistant', 'system']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Message {i} must have a valid role (user, assistant, or system)"
                )
            if not message.content.strip():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Message {i} cannot have empty content"
                )
    
    # Validate memory limit
    if chat_request.memory_limit and (chat_request.memory_limit < 1 or chat_request.memory_limit > 20):
        raise HTTPException(
            status_code=400, 
            detail="Memory limit must be between 1 and 20 messages"
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
                "signed_url_support": True,
                "conversation_history": True,
                "short_term_memory": True,
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
                "signed_url_support": True,
                "conversation_history": True,
                "short_term_memory": True,
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")


@router.get("/a2a/agents")
async def discover_a2a_agents(
    settings: Settings = Depends(get_settings)
):
    """A2A 서버에서 사용 가능한 에이전트들을 탐색합니다 (A2A 표준 준수)."""
    if not settings.a2a_enabled:
        raise HTTPException(status_code=503, detail="A2A functionality is disabled")
    
    try:
        from app.utils.a2a_client import A2AClient
        
        a2a_client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
        agents = await a2a_client.discover_agents()
        
        # A2A 표준에 따른 에이전트 정보 구성
        agent_info = []
        for agent in agents:
            info = {
                "name": agent.get("name", "Unknown Agent"),
                "description": agent.get("description", ""),
                "version": agent.get("version", ""),
                "url": agent.get("url", ""),
                "capabilities": agent.get("capabilities", {}),
                "skills": agent.get("skills", []),
                "provider": agent.get("provider", {}),
                "authentication": agent.get("authentication", {})
            }
            agent_info.append(info)
        
        return {
            "agents": agent_info,
            "count": len(agents),
            "server_url": settings.a2a_server_url,
            "discovery_method": "standard_a2a",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover A2A agents: {str(e)}")

@router.get("/a2a/cards")
async def list_a2a_cards_legacy(
    settings: Settings = Depends(get_settings)
):
    """[Legacy] A2A 서버로부터 사용 가능한 카드 목록을 가져옵니다. /a2a/agents 사용을 권장합니다."""
    # Legacy endpoint - redirect to new endpoint
    return await discover_a2a_agents(settings)


@router.get("/a2a/agents/{agent_id}")
async def get_a2a_agent_details(
    agent_id: str,
    settings: Settings = Depends(get_settings)
):
    """특정 A2A 에이전트의 상세 정보를 가져옵니다 (A2A 표준 준수)."""
    if not settings.a2a_enabled:
        raise HTTPException(status_code=503, detail="A2A functionality is disabled")
    
    try:
        from app.utils.a2a_client import A2AClient
        
        a2a_client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
        # A2A 표준에 따른 에이전트 카드 가져오기
        agent_card = await a2a_client.get_agent_card()
        
        if not agent_card:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or not accessible")
        
        # A2A 표준 Agent Card 구조에 따른 응답
        return {
            "agent": {
                "id": agent_id,
                "name": agent_card.get("name", "Unknown Agent"),
                "description": agent_card.get("description", ""),
                "version": agent_card.get("version", ""),
                "url": agent_card.get("url", ""),
                "documentationUrl": agent_card.get("documentationUrl", ""),
                "capabilities": agent_card.get("capabilities", {}),
                "skills": agent_card.get("skills", []),
                "provider": agent_card.get("provider", {}),
                "authentication": agent_card.get("authentication", {}),
                "defaultInputModes": agent_card.get("defaultInputModes", ["text"]),
                "defaultOutputModes": agent_card.get("defaultOutputModes", ["text"])
            },
            "server_url": settings.a2a_server_url,
            "discovery_method": "agent_card",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch A2A agent details: {str(e)}")

@router.get("/a2a/cards/{card_id}")
async def get_a2a_card_legacy(
    card_id: str,
    settings: Settings = Depends(get_settings)
):
    """[Legacy] 특정 A2A 카드의 상세 정보를 가져옵니다. /a2a/agents/{agent_id} 사용을 권장합니다."""
    # Legacy endpoint - redirect to new endpoint
    return await get_a2a_agent_details(card_id, settings)


@router.get("/a2a/status")
async def check_a2a_status(
    settings: Settings = Depends(get_settings)
):
    """A2A 서버의 상태를 확인합니다."""
    if not settings.a2a_enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "message": "A2A functionality is disabled in settings"
        }
    
    try:
        from app.utils.a2a_client import A2AClient
        
        a2a_client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
        is_healthy = await a2a_client.health_check()
        
        return {
            "enabled": True,
            "status": "healthy" if is_healthy else "unhealthy",
            "server_url": settings.a2a_server_url,
            "timeout": settings.a2a_timeout,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "enabled": True,
            "status": "error",
            "error": str(e),
            "server_url": settings.a2a_server_url,
            "timestamp": time.time()
        } 