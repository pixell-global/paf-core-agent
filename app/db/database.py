"""
Database configuration and session management.
This file handles all database connections and provides session factories.
"""

import os
from app.settings import get_settings
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
import aiomysql

# Database URL assembly
settings = get_settings()

def _build_db_url() -> str:
    # Prefer explicit DATABASE_URL if provided
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # Fallback: compose from DB_HOST/DB_USER/DB_PASSWORD/DB_NAME
    if settings.db_host and settings.db_user and settings.db_password and settings.db_name:
        host = settings.db_host
        user = settings.db_user
        password = settings.db_password
        name = settings.db_name
        return f"mysql+aiomysql://{user}:{password}@{host}:3306/{name}"

    # Default local
    return "mysql+aiomysql://root:password@localhost:3306/paf_core_agent"

DATABASE_URL = _build_db_url()

# Create async engine
# echo=True will log all SQL queries (useful for debugging)
engine = create_async_engine(
    DATABASE_URL,
    echo=True if os.getenv("DEBUG") == "true" else False,
    pool_size=20,  # Connection pool size
    max_overflow=0  # Additional connections beyond pool_size
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )

# Dependency for FastAPI to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session to FastAPI endpoints.
    
    Usage in FastAPI endpoint:
    @app.get("/users")
    async def get_users(db: AsyncSession = Depends(get_db)):
        # Use db session here
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Test database connection
async def test_connection():
    """Test database connection."""
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Initialize database (create all tables)
async def init_db():
    """Initialize database by creating all tables."""
    async with engine.begin() as conn:
        # Import all models to ensure they're registered (including Activity)
        from app.db.models import core, workers, jobs
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully!")