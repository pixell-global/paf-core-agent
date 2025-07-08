"""Authentication models and schemas."""

import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"           # Full system access
    DEVELOPER = "developer"   # API access, file operations, worker tasks
    USER = "user"            # Basic chat access
    READONLY = "readonly"    # Read-only access to status/health


class Permission(str, Enum):
    """System permissions."""
    # Chat permissions
    CHAT_BASIC = "chat:basic"
    CHAT_ADVANCED = "chat:advanced"
    CHAT_FILE_UPLOAD = "chat:file_upload"
    
    # API permissions
    API_MODELS = "api:models"
    API_PROVIDERS = "api:providers"
    API_STATUS = "api:status"
    
    # Worker permissions
    WORKER_EXECUTE = "worker:execute"
    WORKER_STATUS = "worker:status"
    WORKER_MANAGE = "worker:manage"
    
    # File permissions
    FILE_PROCESS = "file:process"
    FILE_ANALYZE = "file:analyze"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_LOGS = "admin:logs"
    
    # Health and monitoring
    HEALTH_READ = "health:read"
    METRICS_READ = "metrics:read"


class User(BaseModel):
    """User model."""
    id: str = Field(description="Unique user identifier")
    username: str = Field(description="Username")
    email: Optional[EmailStr] = Field(default=None, description="User email")
    full_name: Optional[str] = Field(default=None, description="Full display name")
    role: UserRole = Field(description="User role")
    is_active: bool = Field(default=True, description="Whether user is active")
    is_api_user: bool = Field(default=False, description="Whether this is an API-only user")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    api_key_hash: Optional[str] = Field(default=None, description="Hashed API key")
    permissions: List[Permission] = Field(default_factory=list, description="Additional permissions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")
    
    class Config:
        use_enum_values = True


class UserCreate(BaseModel):
    """User creation schema."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: Optional[EmailStr] = Field(default=None, description="User email")
    full_name: Optional[str] = Field(default=None, description="Full display name")
    password: Optional[str] = Field(default=None, min_length=8, description="Password for regular users")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    is_api_user: bool = Field(default=False, description="Whether this is an API-only user")
    permissions: List[Permission] = Field(default_factory=list, description="Additional permissions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = Field(default=None, description="User email")
    full_name: Optional[str] = Field(default=None, description="Full display name")
    role: Optional[UserRole] = Field(default=None, description="User role")
    is_active: Optional[bool] = Field(default=None, description="Whether user is active")
    permissions: Optional[List[Permission]] = Field(default=None, description="Additional permissions")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional user metadata")


class UserResponse(BaseModel):
    """User response schema (excludes sensitive data)."""
    id: str = Field(description="Unique user identifier")
    username: str = Field(description="Username")
    email: Optional[EmailStr] = Field(default=None, description="User email")
    full_name: Optional[str] = Field(default=None, description="Full display name")
    role: UserRole = Field(description="User role")
    is_active: bool = Field(description="Whether user is active")
    is_api_user: bool = Field(description="Whether this is an API-only user")
    created_at: datetime = Field(description="Creation timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    permissions: List[Permission] = Field(description="User permissions")
    
    class Config:
        use_enum_values = True


class Token(BaseModel):
    """JWT token response."""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration in seconds")
    user: UserResponse = Field(description="User information")


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str = Field(description="User ID")
    username: str = Field(description="Username")
    role: UserRole = Field(description="User role")
    permissions: List[Permission] = Field(description="User permissions")
    exp: float = Field(description="Expiration timestamp")
    iat: float = Field(description="Issued at timestamp")


class APIKey(BaseModel):
    """API key model."""
    id: str = Field(description="API key identifier")
    name: str = Field(description="Human-readable name")
    key_hash: str = Field(description="Hashed API key")
    user_id: str = Field(description="Associated user ID")
    permissions: List[Permission] = Field(description="API key permissions")
    is_active: bool = Field(default=True, description="Whether key is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last used timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    rate_limit: Optional[int] = Field(default=None, description="Requests per minute limit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    permissions: List[Permission] = Field(description="API key permissions")
    expires_days: Optional[int] = Field(default=None, description="Expiration in days (None = no expiry)")
    rate_limit: Optional[int] = Field(default=100, description="Requests per minute limit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class APIKeyResponse(BaseModel):
    """API key response (includes the actual key only once)."""
    id: str = Field(description="API key identifier")
    name: str = Field(description="Human-readable name")
    key: Optional[str] = Field(default=None, description="Actual API key (only shown once)")
    permissions: List[Permission] = Field(description="API key permissions")
    created_at: datetime = Field(description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    rate_limit: Optional[int] = Field(default=None, description="Requests per minute limit")


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(description="Username")
    password: str = Field(description="Password")


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    current_password: str = Field(description="Current password")
    new_password: str = Field(min_length=8, description="New password")


class AuditLog(BaseModel):
    """Audit log entry."""
    id: str = Field(description="Log entry ID")
    user_id: Optional[str] = Field(default=None, description="User ID (if authenticated)")
    username: Optional[str] = Field(default=None, description="Username (if authenticated)")
    action: str = Field(description="Action performed")
    resource: str = Field(description="Resource accessed")
    ip_address: str = Field(description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    success: bool = Field(description="Whether action was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")


class SecurityEvent(BaseModel):
    """Security event model."""
    id: str = Field(description="Event ID")
    event_type: str = Field(description="Type of security event")
    severity: str = Field(description="Event severity (low, medium, high, critical)")
    user_id: Optional[str] = Field(default=None, description="Associated user ID")
    ip_address: str = Field(description="Source IP address")
    description: str = Field(description="Event description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int = Field(description="Request limit")
    remaining: int = Field(description="Remaining requests")
    reset_time: float = Field(description="Reset timestamp")
    retry_after: Optional[int] = Field(default=None, description="Retry after seconds") 