"""
PAR Adapter for PAF-Core Agent

Provides REST and gRPC interfaces for PAR (Pixell Agent Runtime).
This module is the bridge between PAR's server infrastructure and PAF-Core's UPEE engine.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from src.core.upee_engine import UPEEEngine
from src.schemas import ChatRequest, EventType
from src.settings import Settings
from src.api.chat import upee_chat_stream, create_sse_event
from src.api.health import router as health_router
from src.api.debug import router as debug_router
from src.api.agents import router as agents_router
from src.api.bridge import router as bridge_router
from src.api.activity_manager import router as activity_manager_router
from src.llm_providers import LLMProviderManager


# ============================================================================
# REST Surface - PAR calls this to mount routes
# ============================================================================

def mount(app: FastAPI) -> None:
    """
    Mount PAF-Core routes onto PAR's FastAPI app.

    PAR will call this function during startup to register our routes.
    Routes will be available under the agent's base path (e.g., /agents/{id}/api/*)

    Args:
        app: PAR's FastAPI application instance
    """

    # Include existing routers (but NOT the main.py root route)
    app.include_router(health_router, prefix="/api/health", tags=["health"])
    app.include_router(debug_router, prefix="/api/debug", tags=["debug"])
    app.include_router(agents_router, tags=["agents"])
    app.include_router(bridge_router, prefix="/api/bridge", tags=["bridge"])
    app.include_router(activity_manager_router, prefix="/api/activity-manager", tags=["activity-manager"])

    # Add the main chat endpoint
    @app.post("/api/chat/stream")
    async def stream_chat(chat_request: ChatRequest, request: Request):
        """Stream chat response using Server-Sent Events with UPEE processing."""

        # Basic validation
        if not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Get settings from environment
        settings = Settings()

        # Get gRPC manager from app state (PAR may provide this)
        grpc_manager = getattr(request.app.state, 'grpc_manager', None)

        # Create streaming response
        return StreamingResponse(
            upee_chat_stream(chat_request, settings, grpc_manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )

    @app.get("/api/chat/models")
    async def list_models():
        """List available LLM models."""
        settings = Settings()
        llm_manager = LLMProviderManager(settings)

        try:
            models = await llm_manager.get_all_models()
            return {
                "models": [
                    {
                        "id": m["name"],
                        "name": m["display_name"],
                        "provider": m["provider"],
                        "available": m["available"],
                    }
                    for m in models
                ],
                "default": settings.resolved_default_model,
            }
        except Exception as e:
            return {"models": [], "error": str(e)}

    @app.get("/api/chat/status")
    async def chat_status():
        """Get chat service status."""
        settings = Settings()
        llm_manager = LLMProviderManager(settings)

        try:
            provider_health = await llm_manager.health_check()
            return {
                "service": "PAF Core Agent",
                "status": "operational",
                "upee_enabled": True,
                "streaming_enabled": True,
                "provider_health": provider_health,
            }
        except Exception as e:
            return {
                "service": "PAF Core Agent",
                "status": "degraded",
                "error": str(e),
            }

    print("✅ PAF-Core routes mounted to PAR app")


# ============================================================================
# gRPC Surface - PAR calls this to get gRPC service handlers
# ============================================================================

def create_service():
    """
    Create gRPC service for A2A protocol.

    PAR will call this to get custom gRPC handlers for A2A communication.
    Returns a dictionary of action handlers.

    Returns:
        Dict with 'custom_handlers' containing action name -> handler function mapping
    """

    settings = Settings()

    async def handle_chat_request(parameters: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle incoming A2A chat request via gRPC.

        Parameters from gRPC ActionRequest:
            message: User message (required)
            model: Optional LLM model
            show_thinking: Optional boolean (as string)

        Returns:
            Dict with success, result, and metadata
        """
        try:
            # Parse parameters
            message = parameters.get("message", "")
            if not message:
                return {
                    "success": False,
                    "error": "Missing required parameter: message"
                }

            model = parameters.get("model")
            show_thinking = parameters.get("show_thinking", "false").lower() == "true"

            # Create chat request
            chat_request = ChatRequest(
                message=message,
                model=model,
                show_thinking=show_thinking
            )

            # Process through UPEE engine
            upee_engine = UPEEEngine(settings)
            await upee_engine.startup()

            # Collect results
            result_content = []
            async for event in upee_engine.process_request(chat_request):
                if event.get("event") == EventType.CONTENT:
                    data = event.get("data", "{}")
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except:
                            pass
                    if isinstance(data, dict) and "content" in data:
                        result_content.append(data["content"])
                    elif isinstance(data, str):
                        result_content.append(data)

            await upee_engine.shutdown()

            return {
                "success": True,
                "result": "".join(result_content),
                "metadata": {"model": model or settings.resolved_default_model}
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def handle_health_check(parameters: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle health check request.

        Returns:
            Dict with health status
        """
        try:
            llm_manager = LLMProviderManager(settings)
            provider_health = await llm_manager.health_check()

            return {
                "success": True,
                "result": "healthy",
                "metadata": {
                    "service": "PAF Core Agent",
                    "upee_enabled": True,
                    "providers": provider_health
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # Return handler dictionary
    # PAR will map these to gRPC ActionRequest.action field
    return {
        "custom_handlers": {
            "chat": handle_chat_request,
            "upee_chat": handle_chat_request,  # Alias
            "health": handle_health_check,
        }
    }


# ============================================================================
# Initialization Handler (Optional - for pre-warming, etc.)
# ============================================================================

async def initialize():
    """
    Optional initialization function called by PAR on agent load.
    Use this for pre-warming models, loading caches, etc.

    Returns:
        Dict with initialization status
    """
    print("🚀 Initializing PAF-Core Agent...")
    settings = Settings()

    try:
        # Pre-warm LLM providers
        llm_manager = LLMProviderManager(settings)
        await llm_manager.health_check()

        print("✅ PAF-Core Agent initialized")
        return {"status": "ready", "message": "PAF-Core Agent initialized successfully"}
    except Exception as e:
        print(f"⚠️  PAF-Core Agent initialization warning: {e}")
        return {"status": "ready_with_warnings", "message": str(e)}


async def shutdown():
    """
    Optional cleanup function called by PAR on agent unload.

    Returns:
        Dict with shutdown status
    """
    print("🛑 Shutting down PAF-Core Agent...")
    # Cleanup resources if needed
    print("✅ PAF-Core Agent shutdown complete")
    return {"status": "shutdown_complete"}
