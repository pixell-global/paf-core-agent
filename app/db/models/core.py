"""
Core database models for users, conversations, messages, and UPEE sessions.
These models represent the fundamental data structures of the application.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, DateTime, Boolean, Float, JSON, ForeignKey, CHAR
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
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey", 
        back_populates="user",
        cascade="all, delete-orphan"
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
    meta_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
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
    meta_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
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
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class Activity(Base):
    """
    Activity model.
    - __tablename__: activity
    - id: CHAR(36)
    - organization_id: CHAR(36)
    - activity_contents: TEXT (stores JSON string)
    """
    __tablename__ = "activity"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    organization_id: Mapped[str] = mapped_column(CHAR(36), index=True, nullable=False)
    activity_contents: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<Activity(id={self.id}, organization_id={self.organization_id})>"