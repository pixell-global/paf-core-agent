"""Main FastAPI application for PAF Core Agent."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, health, files, worker, auth
from app.middleware.security import SecurityHeadersMiddleware, RateLimitMiddleware, AuditLoggingMiddleware, InputValidationMiddleware
from app.utils.logging_config import setup_logging
from app.settings import get_settings
from app.grpc_clients.manager import GRPCClientManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    setup_logging()
    logging.info("PAF Core Agent starting up...")
    
    # Initialize gRPC client manager
    grpc_manager = None
    if settings.grpc_enabled:
        try:
            grpc_manager = GRPCClientManager(settings)
            await grpc_manager.startup()
            app.state.grpc_manager = grpc_manager
            logging.info("gRPC client manager initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize gRPC client manager: {e}")
            # Continue without gRPC functionality
            app.state.grpc_manager = None
    else:
        app.state.grpc_manager = None
        logging.info("gRPC functionality disabled in settings")
    
    yield
    
    # Shutdown
    logging.info("PAF Core Agent shutting down...")
    
    # Shutdown gRPC client manager
    if hasattr(app.state, 'grpc_manager') and app.state.grpc_manager:
        try:
            await app.state.grpc_manager.shutdown()
            logging.info("gRPC client manager shutdown complete")
        except Exception as e:
            logging.error(f"Error during gRPC client manager shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="PAF Core Agent",
        description="UPEE-based chat system with multi-provider LLM support",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add security middleware (order matters)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(InputValidationMiddleware)
    app.add_middleware(AuditLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, default_requests_per_minute=60, default_burst_limit=10)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(files.router, prefix="/api/files", tags=["files"])
    app.include_router(worker.router, prefix="/api/worker", tags=["worker"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logging.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    ) 