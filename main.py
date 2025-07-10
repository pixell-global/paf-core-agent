"""
Main FastAPI application with database initialization.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Import database components
from app.db.database import init_db, test_connection, get_db
from app.core.event_bus import event_bus
from app.core.scheduler import job_scheduler

# Import plugin components
from app.plugins.registry import PluginRegistry
from app.plugins.manager import PluginManager
from app.plugins.examples import ExampleUPEEEnhancer, ExampleTextWorker

# Import existing components
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.worker import router as worker_router
from app.api.bridge import router as bridge_router
from app.api.plugins import router as plugins_router
from app.api.debug import router as debug_router
from app.settings import Settings

# Initialize settings
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Starting PAF Core Agent...")
    
    # Test database connection
    connection_ok = await test_connection()
    if not connection_ok:
        print("❌ Failed to connect to database. Please check your DATABASE_URL.")
        exit(1)
    
    # Initialize database (create tables if they don't exist)
    await init_db()
    
    # Start event bus
    await event_bus.start_listener()
    
    # Start job scheduler
    await job_scheduler.start()
    
    # Initialize plugin system
    from app.api.plugins import get_registry, get_manager
    plugin_registry = get_registry()
    plugin_manager = get_manager()
    
    # Register example plugins
    plugin_registry.register_plugin_class(ExampleUPEEEnhancer)
    plugin_registry.register_plugin_class(ExampleTextWorker)
    
    print("✅ Database initialized successfully")
    print("✅ Event bus started")
    print("✅ Job scheduler started")
    print("✅ Plugin system initialized")
    print(f"✅ PAF Core Agent started on http://localhost:8000")
    
    yield
    
    print("🛑 Shutting down PAF Core Agent...")
    
    # Stop event bus
    await event_bus.stop_listener()
    
    # Stop job scheduler
    await job_scheduler.stop()
    
    print("✅ Event bus stopped")
    print("✅ Job scheduler stopped")

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

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(health_router, prefix="/api/health", tags=["health"])
app.include_router(worker_router, prefix="/api/workers", tags=["workers"])
app.include_router(bridge_router, prefix="/api/bridge", tags=["bridge"])
app.include_router(plugins_router, prefix="/api/plugins", tags=["plugins"])
app.include_router(debug_router, prefix="/api/debug", tags=["debug"])

# Import and include jobs router
from app.api.jobs import router as jobs_router
app.include_router(jobs_router, prefix="/api/jobs", tags=["jobs"])

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "PAF Core Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/db/test")
async def test_db_connection(db: AsyncSession = Depends(get_db)):
    """Test database connection endpoint."""
    try:
        from sqlalchemy import text
        result = await db.execute(text("SELECT version()"))
        version = result.scalar()
        return {
            "status": "connected",
            "database": "PostgreSQL",
            "version": version
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("DEBUG") == "true" else False
    ) 