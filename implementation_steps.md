# Implementation Steps - PAF Core Agent Enhancement

**Complete Entry-Level Developer Guide**

This document provides a comprehensive, step-by-step implementation guide to transform the PAF Core Agent from a UPEE-based chat agent into a full backend orchestrator managing worker agents. Every step includes detailed explanations, complete code examples, and testing instructions suitable for entry-level developers.

## 🎯 Implementation Overview

**Current State**: UPEE-based chat agent with basic gRPC worker communication  
**Target State**: Backend orchestrator with Event Bus, A2A Bridge, Job Scheduler, Worker Pool, and plugin system

**Prerequisites**: 
- Python 3.11+ installed
- PostgreSQL 14+ installed and running
- Basic understanding of FastAPI, SQLAlchemy, and async/await
- Git for version control

---

## 📋 Phase 1: Database Foundation & Core Infrastructure (2-3 weeks)

### 🎯 Phase Goal
Establish persistent storage and enhance the existing authentication system with proper database models and migrations.

### 📚 Learning Resources
- [SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [PostgreSQL UUID Tutorial](https://www.postgresql.org/docs/current/datatype-uuid.html)

### 🛠️ Step 1.1: Install Dependencies

First, update your `requirements.txt` file:

```bash
# Navigate to project root
cd /Users/syum/dev/paf-core-agent

# Add new dependencies to requirements.txt
cat >> requirements.txt << 'EOF'

# Database dependencies
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
asyncpg==0.29.0

# Additional utilities
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
EOF

# Install dependencies
pip install -r requirements.txt
```

**Why these dependencies?**
- `sqlalchemy`: ORM for database operations
- `alembic`: Database migration tool
- `psycopg2-binary`: PostgreSQL adapter for Python
- `asyncpg`: Async PostgreSQL driver for better performance
- `python-multipart`: For handling form data in FastAPI
- `python-jose`: For JWT token handling
- `passlib`: For password hashing

### 🛠️ Step 1.2: Setup Database Structure

Create the database directory structure:

```bash
# Create directory structure
mkdir -p app/db/models
mkdir -p app/db/migrations
mkdir -p tests/db

# Initialize alembic (database migration tool)
alembic init app/db/migrations
```

### 🛠️ Step 1.3: Configure Database Connection

Create `app/db/database.py`:

```python
"""
Database configuration and session management.
This file handles all database connections and provides session factories.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent"
)

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
        async with AsyncSessionLocal() as session:
            result = await session.execute("SELECT 1")
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Initialize database (create all tables)
async def init_db():
    """Initialize database by creating all tables."""
    async with engine.begin() as conn:
        # Import all models to ensure they're registered
        from app.db.models import core, workers, jobs
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully!")
```

### 🛠️ Step 1.4: Create Core Models

Create `app/db/models/core.py`:

```python
"""
Core database models for users, conversations, messages, and UPEE sessions.
These models represent the fundamental data structures of the application.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base

class User(Base):
    """
    User model representing system users.
    
    Fields:
    - id: Unique identifier (UUID)
    - username: Unique username for login
    - email: User's email address
    - password_hash: Hashed password (never store plain text!)
    - role: User role (admin, developer, user, readonly)
    - created_at: When user was created
    - last_login: Last login timestamp
    - is_active: Whether user account is active
    - tenant_id: For multi-tenancy support (added in Phase 5)
    """
    __tablename__ = "users"

    # Primary key using UUID for better security and distribution
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # User credentials
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # User metadata
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps (automatically managed)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Multi-tenancy support (will be used in Phase 5)
    tenant_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    
    # Relationships (SQLAlchemy will handle foreign keys automatically)
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", 
        back_populates="user",
        cascade="all, delete-orphan"  # Delete conversations when user is deleted
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class Conversation(Base):
    """
    Conversation model representing chat sessions between users and the AI.
    Each conversation contains multiple messages.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Conversation metadata
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()  # Automatically update when record changes
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"  # Messages ordered by creation time
    )
    upee_sessions: Mapped[List["UPEESession"]] = relationship(
        "UPEESession", 
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', user_id={self.user_id})>"

class Message(Base):
    """
    Message model representing individual messages within conversations.
    Can be from user, assistant, or system.
    """
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to conversation
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Message content
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"

class UPEESession(Base):
    """
    UPEE Session model for tracking the Understand-Plan-Execute-Evaluate cycles.
    This stores detailed information about each UPEE processing session.
    """
    __tablename__ = "upee_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to conversation
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # UPEE tracking
    request_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    phases_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Store all phase results
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="upee_sessions")
    
    def __repr__(self) -> str:
        return f"<UPEESession(id={self.id}, request_id='{self.request_id}', quality_score={self.quality_score})>"

# API Key model for authentication
class APIKey(Base):
    """
    API Key model for API authentication.
    Supports different permission levels and expiration.
    """
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # API key metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Permissions and status
    permissions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"
```

### 🛠️ Step 1.5: Create Worker Models

Create `app/db/models/workers.py`:

```python
"""
Worker management models for tracking worker instances and their tasks.
These models support the distributed worker architecture.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base

class WorkerInstance(Base):
    """
    Worker Instance model representing individual worker agents.
    Each worker can handle specific types of tasks based on capabilities.
    """
    __tablename__ = "worker_instances"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Worker identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)  # gRPC endpoint
    
    # Worker capabilities and metadata
    capabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Worker status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")  # active, inactive, unhealthy
    
    # Health monitoring
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    tasks: Mapped[List["WorkerTask"]] = relationship(
        "WorkerTask", 
        back_populates="worker",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<WorkerInstance(id={self.id}, name='{self.name}', status='{self.status}')>"

class WorkerTask(Base):
    """
    Worker Task model representing individual tasks assigned to workers.
    Tracks task lifecycle from creation to completion.
    """
    __tablename__ = "worker_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to worker
    worker_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("worker_instances.id", ondelete="SET NULL"),
        nullable=True,  # Can be null if worker is deleted
        index=True
    )
    
    # Task details
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Task status tracking
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps for performance tracking
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Priority and retry logic
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)  # 1=highest, 10=lowest
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    
    # Relationships
    worker: Mapped[Optional["WorkerInstance"]] = relationship("WorkerInstance", back_populates="tasks")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self) -> str:
        return f"<WorkerTask(id={self.id}, task_type='{self.task_type}', status='{self.status}')>"
```

### 🛠️ Step 1.6: Create Job Models

Create `app/db/models/jobs.py`:

```python
"""
Job scheduling models for managing scheduled tasks and job execution.
Supports cron jobs, one-time jobs, and interval-based jobs.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.database import Base

class Job(Base):
    """
    Job model representing scheduled tasks in the system.
    Supports cron expressions, intervals, and one-time execution.
    """
    __tablename__ = "core_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'one_time', 'cron', 'interval'
    
    # Scheduling configuration
    schedule_spec: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Cron expression or interval
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Execution tracking
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")  # queued, running, completed, failed, paused
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Worker assignment
    worker_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("worker_instances.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Job configuration
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeout_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    executions: Mapped[List["JobExecution"]] = relationship(
        "JobExecution", 
        back_populates="job",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, name='{self.name}', job_type='{self.job_type}', status='{self.status}')>"

class JobExecution(Base):
    """
    Job Execution model tracking individual runs of scheduled jobs.
    Useful for monitoring job history and performance.
    """
    __tablename__ = "job_executions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to job
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("core_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Execution details
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # running, completed, failed
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Performance tracking
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Resource usage (for monitoring)
    cpu_usage: Mapped[Optional[float]] = mapped_column(JSON, nullable=True)
    memory_usage: Mapped[Optional[float]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="executions")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self) -> str:
        return f"<JobExecution(id={self.id}, job_id={self.job_id}, status='{self.status}')>"

# Event models for the event bus (Phase 2)
class Event(Base):
    """
    Event model for the event bus system.
    Stores all events flowing through the system.
    """
    __tablename__ = "flex_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Multi-tenancy support
    tenant_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    
    # Processing tracking
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<Event(id={self.id}, event_type='{self.event_type}', source='{self.source}')>"
```

### 🛠️ Step 1.7: Configure Alembic for Migrations

Edit `alembic.ini`:

```ini
# Alembic configuration file

[alembic]
# Path to migration scripts
script_location = app/db/migrations

# Template used to generate migration files
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
prepend_sys_path = .

# Timezone to use when rendering the date within the migration file
timezone = UTC

# Max length of characters to apply to the
# "slug" field
truncate_slug_length = 40

# Set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
revision_environment = false

# Set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# Version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
version_path_separator = :

# Set to 'true' to search source files recursively
# in each "version_locations" directory
recursive_version_locations = false

# The output encoding used when revision files are written from script.py.mako
output_encoding = utf-8

# Database URL (will be set programmatically)
sqlalchemy.url = 

[post_write_hooks]
# Post-write hooks define scripts or Python functions that are run
# on newly generated revision scripts.

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Edit `app/db/migrations/env.py`:

```python
"""
Alembic environment configuration.
This file is executed when running alembic commands.
"""

import asyncio
import os
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import your models here so Alembic can detect them
from app.db.database import Base
from app.db.models import core, workers, jobs  # Import all model modules

# This is the Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from environment
database_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent"
)
config.set_main_option("sqlalchemy.url", database_url)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    This configures the context with just a URL and not an Engine.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a database connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())

# Determine if we're running in offline or online mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 🛠️ Step 1.8: Create Database and Run Migrations

```bash
# Create PostgreSQL database
createdb paf_core_agent
createdb paf_core_agent_test

# Set environment variable for database URL
export DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent"

# Generate initial migration
alembic revision --autogenerate -m "Initial database schema"

# Apply migrations to create tables
alembic upgrade head

# Verify tables were created
psql paf_core_agent -c "\dt"
```

### 🛠️ Step 1.9: Update Authentication Models

Modify `app/auth/models.py` to use database models:

```python
"""
Authentication models and utilities.
Updated to use SQLAlchemy database models.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models.core import User, APIKey

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthModels:
    """Authentication utilities for user and API key management."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        username: str,
        email: str,
        password: str,
        role: str = "user"
    ) -> User:
        """Create a new user in the database."""
        password_hash = AuthModels.hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    
    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
        """Get user by username."""
        result = await db.execute(
            select(User).where(User.username == username, User.is_active == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(
            select(User).where(User.email == email, User.is_active == True)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """Authenticate user with username and password."""
        user = await AuthModels.get_user_by_username(db, username)
        if not user:
            return None
        
        if not AuthModels.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        return user
    
    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        user_id: uuid.UUID,
        name: str,
        permissions: Dict[str, Any],
        expires_at: Optional[datetime] = None
    ) -> tuple[APIKey, str]:
        """Create a new API key for a user."""
        # Generate random API key
        import secrets
        api_key = f"pak_{secrets.token_urlsafe(32)}"
        key_hash = AuthModels.hash_password(api_key)
        
        api_key_obj = APIKey(
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        db.add(api_key_obj)
        await db.commit()
        await db.refresh(api_key_obj)
        
        return api_key_obj, api_key
    
    @staticmethod
    async def verify_api_key(db: AsyncSession, api_key: str) -> Optional[User]:
        """Verify API key and return associated user."""
        # Get all active API keys (we need to check hash against each)
        result = await db.execute(
            select(APIKey)
            .options(selectinload(APIKey.user))  # Load user relationship
            .where(APIKey.is_active == True)
        )
        api_keys = result.scalars().all()
        
        for key_obj in api_keys:
            if AuthModels.verify_password(api_key, key_obj.key_hash):
                # Check if key is expired
                if key_obj.expires_at and key_obj.expires_at < datetime.utcnow():
                    continue
                
                # Update last used
                key_obj.last_used = datetime.utcnow()
                await db.commit()
                
                return key_obj.user
        
        return None

# Role-based permissions
class Roles:
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    READONLY = "readonly"
    
    @staticmethod
    def get_permissions(role: str) -> Dict[str, bool]:
        """Get permissions for a role."""
        permissions = {
            "read_conversations": False,
            "write_conversations": False,
            "manage_users": False,
            "manage_workers": False,
            "manage_plugins": False,
            "admin_access": False,
        }
        
        if role == Roles.ADMIN:
            return {key: True for key in permissions}
        elif role == Roles.DEVELOPER:
            permissions.update({
                "read_conversations": True,
                "write_conversations": True,
                "manage_workers": True,
                "manage_plugins": True,
            })
        elif role == Roles.USER:
            permissions.update({
                "read_conversations": True,
                "write_conversations": True,
            })
        elif role == Roles.READONLY:
            permissions.update({
                "read_conversations": True,
            })
        
        return permissions
```

### 🛠️ Step 1.10: Update Main Application

Modify `app/main.py` to initialize database:

```python
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

# Import existing components
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.worker import router as worker_router
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
    
    print("✅ Database initialized successfully")
    print(f"✅ PAF Core Agent started on http://localhost:8000")
    
    yield
    
    print("🛑 Shutting down PAF Core Agent...")

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
```

### 🛠️ Step 1.11: Update Chat API for Persistence

Modify `app/api/chat.py` to save conversations:

```python
"""
Chat API with conversation persistence.
Updated to save conversations and messages to database.
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.db.models.core import User, Conversation, Message, UPEESession
from app.auth.models import AuthModels
from app.schemas import ChatRequest, ChatResponse
from app.core.upee_engine import UPEEEngine
from app.utils.sse import SSEResponse

router = APIRouter()

async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Get current user from request headers."""
    # Try API key authentication first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = await AuthModels.verify_api_key(db, api_key)
        if user:
            return user
    
    # For development, create a default user if none exists
    # TODO: Implement proper JWT authentication in production
    default_user = await AuthModels.get_user_by_username(db, "default_user")
    if not default_user:
        default_user = await AuthModels.create_user(
            db=db,
            username="default_user",
            email="default@example.com",
            password="default_password",
            role="user"
        )
    
    return default_user

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Stream chat response using Server-Sent Events.
    Saves conversation and messages to database.
    """
    
    # Get or create conversation
    conversation = None
    if request.conversation_id:
        # Load existing conversation
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == request.conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
            metadata={"model": request.model, "temperature": request.temperature}
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        metadata={"files": request.files if hasattr(request, 'files') else []}
    )
    db.add(user_message)
    
    # Create UPEE session for tracking
    request_id = str(uuid.uuid4())
    upee_session = UPEESession(
        conversation_id=conversation.id,
        request_id=request_id,
        phases_data={}
    )
    db.add(upee_session)
    await db.commit()
    
    # Initialize UPEE engine
    upee_engine = UPEEEngine()
    
    async def generate_response():
        """Generate streaming response with SSE."""
        assistant_content = ""
        phases_data = {}
        
        try:
            # Send conversation ID first
            yield SSEResponse.event("conversation", {
                "conversation_id": str(conversation.id),
                "request_id": request_id
            })
            
            # Stream UPEE response
            async for event_type, data in upee_engine.process_request(
                message=request.message,
                conversation_history=[
                    {"role": msg.role, "content": msg.content} 
                    for msg in conversation.messages[-10:]  # Last 10 messages for context
                ],
                files=getattr(request, 'files', []),
                model=request.model,
                show_thinking=request.show_thinking
            ):
                
                if event_type == "thinking":
                    phases_data[data.get("phase", "unknown")] = data
                    if request.show_thinking:
                        yield SSEResponse.event("thinking", data)
                
                elif event_type == "content":
                    assistant_content += data.get("content", "")
                    yield SSEResponse.event("content", data)
                
                elif event_type == "complete":
                    # Update UPEE session with final data
                    upee_session.phases_data = phases_data
                    upee_session.quality_score = data.get("quality_score")
                    
                    # Save assistant message
                    assistant_message = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=assistant_content,
                        metadata={
                            "model": request.model,
                            "request_id": request_id,
                            "quality_score": data.get("quality_score")
                        }
                    )
                    db.add(assistant_message)
                    
                    # Update conversation timestamp
                    conversation.updated_at = datetime.utcnow()
                    
                    # Commit all changes
                    await db.commit()
                    
                    yield SSEResponse.event("complete", {
                        **data,
                        "conversation_id": str(conversation.id),
                        "message_count": len(conversation.messages) + 2  # +2 for new messages
                    })
            
            yield SSEResponse.event("done", {})
            
        except Exception as e:
            # Log error and update session
            upee_session.phases_data = {
                **phases_data,
                "error": {"message": str(e), "timestamp": datetime.utcnow().isoformat()}
            }
            await db.commit()
            
            yield SSEResponse.event("error", {"error": str(e)})
            yield SSEResponse.event("done", {})
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """Get user's conversations with pagination."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    conversations = result.scalars().all()
    
    return {
        "conversations": [
            {
                "id": str(conv.id),
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "metadata": conv.metadata
            }
            for conv in conversations
        ],
        "limit": limit,
        "offset": offset,
        "total": len(conversations)
    }

@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get conversation with messages."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "metadata": conversation.metadata,
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "metadata": msg.metadata
            }
            for msg in conversation.messages
        ]
    }

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await db.delete(conversation)
    await db.commit()
    
    return {"message": "Conversation deleted successfully"}

@router.get("/models")
async def get_available_models():
    """Get available LLM models."""
    # This will be expanded in later phases
    return {
        "models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": "openai",
                "available": True
            },
            {
                "id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "available": True
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "available": True
            }
        ]
    }
```

### 🛠️ Step 1.12: Create Test Suite

Create `tests/test_db_models.py`:

```python
"""
Test suite for database models.
Tests all CRUD operations and relationships.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from app.db.database import Base
from app.db.models.core import User, Conversation, Message, UPEESession, APIKey
from app.db.models.workers import WorkerInstance, WorkerTask
from app.db.models.jobs import Job, JobExecution
from app.auth.models import AuthModels

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent_test"

@pytest.fixture
async def db_session():
    """Create test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Provide session
    async with async_session() as session:
        yield session
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession):
    """Test user creation."""
    user = await AuthModels.create_user(
        db=db_session,
        username="testuser",
        email="test@example.com",
        password="testpassword",
        role="user"
    )
    
    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.role == "user"
    assert user.is_active is True
    assert AuthModels.verify_password("testpassword", user.password_hash)

@pytest.mark.asyncio
async def test_authenticate_user(db_session: AsyncSession):
    """Test user authentication."""
    # Create user
    await AuthModels.create_user(
        db=db_session,
        username="authtest",
        email="auth@example.com",
        password="authpassword"
    )
    
    # Test valid authentication
    user = await AuthModels.authenticate_user(db_session, "authtest", "authpassword")
    assert user is not None
    assert user.username == "authtest"
    
    # Test invalid password
    invalid_user = await AuthModels.authenticate_user(db_session, "authtest", "wrongpassword")
    assert invalid_user is None
    
    # Test invalid username
    invalid_user = await AuthModels.authenticate_user(db_session, "wronguser", "authpassword")
    assert invalid_user is None

@pytest.mark.asyncio
async def test_conversation_and_messages(db_session: AsyncSession):
    """Test conversation and message creation."""
    # Create user
    user = await AuthModels.create_user(
        db=db_session,
        username="convuser",
        email="conv@example.com",
        password="password"
    )
    
    # Create conversation
    conversation = Conversation(
        user_id=user.id,
        title="Test Conversation",
        metadata={"test": True}
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    
    # Create messages
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content="Hello, how are you?"
    )
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="I'm doing well, thank you!"
    )
    
    db_session.add_all([user_message, assistant_message])
    await db_session.commit()
    
    # Test relationships
    await db_session.refresh(conversation)
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[1].role == "assistant"

@pytest.mark.asyncio
async def test_upee_session(db_session: AsyncSession):
    """Test UPEE session tracking."""
    # Create user and conversation
    user = await AuthModels.create_user(
        db=db_session,
        username="upeeuser",
        email="upee@example.com",
        password="password"
    )
    
    conversation = Conversation(
        user_id=user.id,
        title="UPEE Test"
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    
    # Create UPEE session
    upee_session = UPEESession(
        conversation_id=conversation.id,
        request_id="test_request_123",
        phases_data={
            "understand": {"intent": "greeting"},
            "plan": {"strategy": "friendly_response"},
            "execute": {"model": "gpt-4"},
            "evaluate": {"quality": 0.95}
        },
        quality_score=0.95
    )
    
    db_session.add(upee_session)
    await db_session.commit()
    
    # Test relationships
    await db_session.refresh(conversation)
    assert len(conversation.upee_sessions) == 1
    assert conversation.upee_sessions[0].request_id == "test_request_123"
    assert conversation.upee_sessions[0].quality_score == 0.95

@pytest.mark.asyncio
async def test_worker_instance_and_tasks(db_session: AsyncSession):
    """Test worker instance and task creation."""
    # Create worker instance
    worker = WorkerInstance(
        name="test_worker",
        endpoint="localhost:50051",
        capabilities={
            "languages": ["python", "javascript"],
            "frameworks": ["fastapi", "react"]
        },
        status="active"
    )
    
    db_session.add(worker)
    await db_session.commit()
    await db_session.refresh(worker)
    
    # Create worker task
    task = WorkerTask(
        worker_id=worker.id,
        task_type="code_analysis",
        payload={"file_path": "/app/main.py"},
        status="pending",
        priority=3
    )
    
    db_session.add(task)
    await db_session.commit()
    
    # Test relationships
    await db_session.refresh(worker)
    assert len(worker.tasks) == 1
    assert worker.tasks[0].task_type == "code_analysis"
    assert worker.tasks[0].status == "pending"

@pytest.mark.asyncio
async def test_job_scheduling(db_session: AsyncSession):
    """Test job scheduling functionality."""
    # Create job
    job = Job(
        name="daily_report",
        job_type="cron",
        schedule_spec="0 9 * * *",  # Daily at 9 AM
        payload={"report_type": "daily", "recipients": ["admin@example.com"]}
    )
    
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    # Create job execution
    execution = JobExecution(
        job_id=job.id,
        status="completed",
        result={"success": True, "records_processed": 100}
    )
    
    db_session.add(execution)
    await db_session.commit()
    
    # Test relationships
    await db_session.refresh(job)
    assert len(job.executions) == 1
    assert job.executions[0].status == "completed"

@pytest.mark.asyncio
async def test_api_key_creation(db_session: AsyncSession):
    """Test API key creation and verification."""
    # Create user
    user = await AuthModels.create_user(
        db=db_session,
        username="apiuser",
        email="api@example.com",
        password="password"
    )
    
    # Create API key
    api_key_obj, api_key = await AuthModels.create_api_key(
        db=db_session,
        user_id=user.id,
        name="Test API Key",
        permissions={"read": True, "write": False}
    )
    
    assert api_key.startswith("pak_")
    assert api_key_obj.name == "Test API Key"
    assert api_key_obj.permissions == {"read": True, "write": False}
    
    # Test API key verification
    verified_user = await AuthModels.verify_api_key(db_session, api_key)
    assert verified_user is not None
    assert verified_user.id == user.id
    
    # Test invalid API key
    invalid_user = await AuthModels.verify_api_key(db_session, "invalid_key")
    assert invalid_user is None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
```

### 🛠️ Step 1.13: Update Schemas

Update `app/schemas.py` to include conversation fields:

```python
"""
Updated Pydantic schemas with conversation support.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Chat-related schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[uuid.UUID] = None
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    show_thinking: bool = Field(default=False)
    files: Optional[List[str]] = Field(default=[])

class ChatResponse(BaseModel):
    content: str
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    model: str
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# Conversation schemas
class ConversationCreate(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]
    message_count: Optional[int] = None

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]]

class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse]

# User schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: str = Field(default="user")

class UserResponse(BaseModel):
    id: uuid.UUID
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

# API Key schemas
class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None

class APIKeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    permissions: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool

# Worker schemas
class WorkerInstanceCreate(BaseModel):
    name: str
    endpoint: str
    capabilities: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class WorkerInstanceResponse(BaseModel):
    id: uuid.UUID
    name: str
    endpoint: str
    capabilities: Optional[Dict[str, Any]]
    status: str
    last_heartbeat: datetime
    created_at: datetime

class WorkerTaskCreate(BaseModel):
    task_type: str
    payload: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    max_retries: int = Field(default=3, ge=0, le=10)

class WorkerTaskResponse(BaseModel):
    id: uuid.UUID
    task_type: str
    status: str
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    priority: int
    retry_count: int

# Job schemas
class JobCreate(BaseModel):
    name: str
    job_type: str  # 'one_time', 'cron', 'interval'
    schedule_spec: Optional[str] = None
    payload: Dict[str, Any]
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: Optional[int] = None

class JobResponse(BaseModel):
    id: uuid.UUID
    name: str
    job_type: str
    schedule_spec: Optional[str]
    status: str
    payload: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime]
    max_retries: int

# Health check schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, Any]
    version: str

# Error schemas
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None
```

### 🛠️ Step 1.14: Test the Implementation

Create `scripts/test_phase1.py`:

```python
"""
Test script for Phase 1 implementation.
Verifies database connectivity, model creation, and basic functionality.
"""

import asyncio
import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import AsyncSessionLocal, init_db, test_connection
from app.auth.models import AuthModels

async def test_phase1():
    """Test Phase 1 implementation."""
    print("🧪 Testing Phase 1: Database Foundation")
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    connection_ok = await test_connection()
    if not connection_ok:
        print("❌ Database connection failed!")
        return False
    
    # Test 2: Initialize database
    print("\n2. Initializing database...")
    await init_db()
    print("✅ Database initialized")
    
    # Test 3: User creation and authentication
    print("\n3. Testing user creation and authentication...")
    async with AsyncSessionLocal() as db:
        try:
            # Create test user
            user = await AuthModels.create_user(
                db=db,
                username="test_phase1",
                email="test@phase1.com",
                password="testpassword123",
                role="user"
            )
            print(f"✅ User created: {user.username} ({user.id})")
            
            # Test authentication
            auth_user = await AuthModels.authenticate_user(db, "test_phase1", "testpassword123")
            if auth_user and auth_user.id == user.id:
                print("✅ User authentication works")
            else:
                print("❌ User authentication failed")
                return False
            
            # Test API key creation
            api_key_obj, api_key = await AuthModels.create_api_key(
                db=db,
                user_id=user.id,
                name="Test Key",
                permissions={"read": True, "write": True}
            )
            print(f"✅ API key created: {api_key[:20]}...")
            
            # Test API key verification
            verified_user = await AuthModels.verify_api_key(db, api_key)
            if verified_user and verified_user.id == user.id:
                print("✅ API key verification works")
            else:
                print("❌ API key verification failed")
                return False
            
        except Exception as e:
            print(f"❌ User/API key test failed: {e}")
            return False
    
    # Test 4: Database models
    print("\n4. Testing database models...")
    async with AsyncSessionLocal() as db:
        try:
            from app.db.models.core import Conversation, Message
            from app.db.models.workers import WorkerInstance, WorkerTask
            from app.db.models.jobs import Job
            
            # Test conversation creation
            conversation = Conversation(
                user_id=user.id,
                title="Test Conversation",
                metadata={"test": True}
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            print(f"✅ Conversation created: {conversation.id}")
            
            # Test message creation
            message = Message(
                conversation_id=conversation.id,
                role="user",
                content="Test message"
            )
            db.add(message)
            await db.commit()
            print(f"✅ Message created: {message.id}")
            
            # Test worker instance
            worker = WorkerInstance(
                name="test_worker",
                endpoint="localhost:50051",
                capabilities={"test": True}
            )
            db.add(worker)
            await db.commit()
            await db.refresh(worker)
            print(f"✅ Worker instance created: {worker.id}")
            
            # Test job creation
            job = Job(
                name="test_job",
                job_type="one_time",
                payload={"test": True}
            )
            db.add(job)
            await db.commit()
            print(f"✅ Job created: {job.id}")
            
        except Exception as e:
            print(f"❌ Database models test failed: {e}")
            return False
    
    print("\n🎉 Phase 1 tests completed successfully!")
    print("\n✅ Database foundation is ready")
    print("✅ User authentication works")
    print("✅ API key system works")
    print("✅ All database models work")
    print("\n🚀 Ready to proceed to Phase 2!")
    
    return True

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent"
    
    asyncio.run(test_phase1())
```

### 🛠️ Step 1.15: Run Tests and Verification

```bash
# Make sure PostgreSQL is running
sudo service postgresql start  # Linux
# or
brew services start postgresql  # macOS

# Create databases if they don't exist
createdb paf_core_agent
createdb paf_core_agent_test

# Set environment variable
export DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/paf_core_agent"

# Run Phase 1 test script
python scripts/test_phase1.py

# Run pytest tests
pytest tests/test_db_models.py -v

# Start the application
uvicorn app.main:app --reload

# Test the API
curl http://localhost:8000/api/db/test
curl http://localhost:8000/api/health
```

### ✅ Phase 1 Deliverables Checklist

- [ ] ✅ PostgreSQL database setup and connection
- [ ] ✅ SQLAlchemy models for all core entities
- [ ] ✅ Alembic migrations configuration
- [ ] ✅ User authentication with password hashing
- [ ] ✅ API key generation and verification
- [ ] ✅ Conversation and message persistence
- [ ] ✅ UPEE session tracking
- [ ] ✅ Worker and job models
- [ ] ✅ FastAPI integration with database
- [ ] ✅ Comprehensive test suite
- [ ] ✅ Development and testing scripts

### 🎯 Success Criteria for Phase 1

1. **Database Connection**: Application successfully connects to PostgreSQL
2. **Migrations**: Alembic can create and manage database schema
3. **Authentication**: Users can be created, authenticated, and managed
4. **API Keys**: API keys can be generated and verified
5. **Conversations**: Chat conversations persist across sessions
6. **Models**: All database models work with proper relationships
7. **Tests**: All tests pass successfully
8. **API Integration**: Chat API saves conversations to database

### 🔄 Common Issues and Solutions

**Issue**: Database connection fails
**Solution**: Check PostgreSQL is running and DATABASE_URL is correct

**Issue**: Alembic migration fails
**Solution**: Ensure all models are imported in env.py

**Issue**: Tests fail with "table doesn't exist"
**Solution**: Run alembic upgrade head before tests

**Issue**: Password hashing is slow
**Solution**: Adjust bcrypt rounds in production (12-14 rounds)

---

## 📈 Phase 1 Summary

**What we accomplished:**
- ✅ Established robust database foundation with PostgreSQL
- ✅ Implemented proper user authentication and API key management
- ✅ Created comprehensive data models for all system entities
- ✅ Set up database migrations with Alembic
- ✅ Integrated database persistence into existing chat API
- ✅ Built comprehensive test suite for all functionality

**What's ready for Phase 2:**
- Database layer is fully operational
- User management system is complete
- Conversation persistence is working
- All models and relationships are tested
- API endpoints save data correctly

**Next Steps:**
Phase 2 will build on this foundation to implement the Event Bus and Job Scheduler, enabling real-time event-driven architecture for the worker agent orchestration system.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Review implementation_steps.md for improvements", "status": "completed", "priority": "high"}, {"id": "2", "content": "Rewrite implementation_steps.md with detailed entry-level developer guidance", "status": "completed", "priority": "high"}]