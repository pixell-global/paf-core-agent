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

# Multi-agent support
from src.agents.agent_app_discovery import AgentAppDiscoveryService
from src.agents.agent_app_selector import AgentAppSelector
from src.agents.agent_client_pool import AgentClientPool
from src.utils.logging_config import get_logger

# Module-level logger
logger = get_logger("par_adapter")

# Module-level state for multi-agent components
# These are initialized once at PAR startup and reused across all requests
_multi_agent_state = {
    "discovery_service": None,
    "selector": None,
    "client_pool": None,
    "registry": None,
    "initialized": False
}


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
        global _multi_agent_state

        try:
            # Validate multi-agent initialization
            if not _multi_agent_state.get("initialized"):
                logger.warning(
                    "⚠️  Multi-agent components not initialized - routing may fail. "
                    "Agent requests will be handled directly by PAF Core."
                )

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

            # Create UPEE engine WITH multi-agent components from global state
            # No startup/shutdown needed - components are already initialized
            upee_engine = UPEEEngine(
                settings,
                registry=_multi_agent_state.get("registry"),
                selector=_multi_agent_state.get("selector"),
                client_pool=_multi_agent_state.get("client_pool")
            )

            logger.debug(
                "Processing chat request",
                message_preview=message[:100],
                has_registry=_multi_agent_state.get("registry") is not None,
                has_selector=_multi_agent_state.get("selector") is not None,
                has_client_pool=_multi_agent_state.get("client_pool") is not None
            )

            # Collect results and metadata
            result_content = []
            complete_metadata = {}

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

                elif event.get("event") == EventType.COMPLETE:
                    # Extract metadata from CompleteEvent
                    data = event.get("data", "{}")
                    if isinstance(data, str):
                        try:
                            complete_event_data = json.loads(data)
                            # Include agent attribution fields
                            complete_metadata = {
                                "model": complete_event_data.get("model", model or settings.resolved_default_model),
                                "agent_used": complete_event_data.get("agent_used"),
                                "agent_app_id": complete_event_data.get("agent_app_id"),
                                "skill_used": complete_event_data.get("skill_used"),
                                "skill_id": complete_event_data.get("skill_id"),
                                "routing_source": complete_event_data.get("routing_source"),
                                "total_tokens": complete_event_data.get("total_tokens"),
                                "duration": complete_event_data.get("duration")
                            }
                        except:
                            pass

            # No shutdown needed - components are managed by lifecycle hooks

            # If no complete event was captured, use fallback metadata
            if not complete_metadata:
                complete_metadata = {"model": model or settings.resolved_default_model}

            logger.debug(
                "Chat request completed",
                routing_source=complete_metadata.get("routing_source"),
                agent_used=complete_metadata.get("agent_used"),
                result_length=len("".join(result_content))
            )

            return {
                "success": True,
                "result": "".join(result_content),
                "metadata": complete_metadata
            }

        except Exception as e:
            logger.error(f"Error handling chat request: {e}", exc_info=True)
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
# Lifecycle Management - PAR application startup/shutdown hooks
# ============================================================================

async def startup(app: FastAPI = None) -> Dict[str, Any]:
    """
    PAR lifecycle hook: Initialize multi-agent components once at startup.

    This function is called by PAR when the agent is loaded. It initializes
    the discovery service, agent registry, selector, and client pool that
    will be reused across all requests.

    Args:
        app: Optional FastAPI app instance (for storing in app.state)

    Returns:
        Dict with initialization status
    """
    global _multi_agent_state

    logger.info("🚀 PAF-Core Agent startup: Initializing multi-agent components")
    settings = Settings()

    try:
        # Initialize discovery service
        logger.info("Initializing agent discovery service...")
        discovery_service = AgentAppDiscoveryService(settings)
        await discovery_service.startup()

        # Get registry
        registry = discovery_service.get_registry()

        # Initialize selector
        logger.info("Initializing agent selector...")
        selector = AgentAppSelector(settings, registry)

        # Initialize client pool
        logger.info("Initializing agent client pool...")
        client_pool = AgentClientPool(registry)
        client_pool.initialize()

        # Store in module-level state
        _multi_agent_state = {
            "discovery_service": discovery_service,
            "selector": selector,
            "client_pool": client_pool,
            "registry": registry,
            "initialized": True
        }

        # Also store in app.state if app provided (for REST endpoints)
        if app:
            app.state.discovery_service = discovery_service
            app.state.selector = selector
            app.state.client_pool = client_pool

        # Log configuration details
        agents_configured = len(registry.agents) if registry else 0
        logger.info(
            "Multi-agent startup complete",
            a2a_enabled=settings.a2a_enabled,
            agent_apps_configured=len(settings.a2a_agent_apps or []),
            agents_discovered=agents_configured,
            discovery_interval=settings.a2a_card_refresh_interval
        )

        # Validate configuration
        if not settings.a2a_agent_apps:
            logger.warning(
                "⚠️  No A2A_AGENT_APPS configured - multi-agent routing will not work. "
                "Set A2A_AGENT_APPS environment variable to enable agent coordination."
            )

        # Pre-warm LLM providers
        try:
            llm_manager = LLMProviderManager(settings)
            await llm_manager.health_check()
            logger.info("✅ LLM providers initialized")
        except Exception as e:
            logger.warning(f"⚠️  LLM provider initialization warning: {e}")

        logger.info("✅ PAF-Core Agent startup complete")
        return {
            "status": "ready",
            "multi_agent_enabled": True,
            "agents_configured": agents_configured,
            "message": "PAF-Core Agent initialized successfully"
        }

    except Exception as e:
        logger.error(f"❌ PAF-Core Agent startup failed: {e}", exc_info=True)
        # Set partial initialization state
        _multi_agent_state["initialized"] = False
        return {
            "status": "ready_with_warnings",
            "multi_agent_enabled": False,
            "error": str(e),
            "message": "PAF-Core Agent started with limited functionality"
        }


async def shutdown() -> Dict[str, Any]:
    """
    PAR lifecycle hook: Clean up multi-agent components on shutdown.

    This function is called by PAR when the agent is unloaded.

    Returns:
        Dict with shutdown status
    """
    global _multi_agent_state

    logger.info("🛑 PAF-Core Agent shutdown: Cleaning up multi-agent components")

    try:
        # Shutdown discovery service
        if _multi_agent_state.get("discovery_service"):
            logger.info("Shutting down discovery service...")
            await _multi_agent_state["discovery_service"].shutdown()

        # Close client pool
        if _multi_agent_state.get("client_pool"):
            logger.info("Closing client pool...")
            _multi_agent_state["client_pool"].close_all()

        # Reset state
        _multi_agent_state = {
            "discovery_service": None,
            "selector": None,
            "client_pool": None,
            "registry": None,
            "initialized": False
        }

        logger.info("✅ PAF-Core Agent shutdown complete")
        return {"status": "shutdown_complete"}

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}", exc_info=True)
        return {"status": "shutdown_with_errors", "error": str(e)}


# Backward compatibility: Keep initialize() as alias for startup()
async def initialize(app: FastAPI = None) -> Dict[str, Any]:
    """
    Legacy initialization function - calls startup() for backward compatibility.

    Returns:
        Dict with initialization status
    """
    return await startup(app)
