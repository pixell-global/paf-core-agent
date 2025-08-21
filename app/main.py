"""
Main FastAPI application for PAF Core Agent.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import existing components
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.debug import router as debug_router
from app.api.agents import router as agents_router
from app.api.bridge import router as bridge_router
from app.api.activity_manager import router as activity_manager_router
from app.settings import Settings

# Acitivty Manager
from app.core.activity_manger import ActivityManager


# Initialize settings
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Starting PAF Core Agent...")
    
    # Basic startup - no database required
    print("✅ Core UPEE functionality initialized")
    print("✅ LLM providers configured")
    print("✅ File processing available")
    print(f"✅ PAF Core Agent started on http://localhost:8000")
    
    yield
    
    print("🛑 Shutting down PAF Core Agent...")
    print("✅ Graceful shutdown completed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="PAF Core Agent",
    description="Backend orchestrator for managing worker agents with UPEE loop",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core routers (only non-database dependent ones)
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(health_router, prefix="/api/health", tags=["health"])
app.include_router(debug_router, prefix="/api/debug", tags=["debug"])
app.include_router(agents_router)
app.include_router(bridge_router, prefix="/api/bridge", tags=["bridge"])
app.include_router(activity_manager_router, prefix="/api/activity-manager", tags=["activity-manager"])

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "PAF Core Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("DEBUG") == "true" else False
    ) 