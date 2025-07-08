"""Authentication API endpoints."""

import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.models import (
    User, UserCreate, UserUpdate, UserResponse, UserRole, Permission,
    Token, LoginRequest, PasswordChangeRequest,
    APIKey, APIKeyCreate, APIKeyResponse,
    AuditLog, SecurityEvent
)
from app.auth.jwt_handler import JWTHandler
from app.auth.dependencies import (
    get_current_user, get_current_active_user, get_jwt_handler, get_user_store,
    require_admin, require_permission_dependency, get_request_context
)
from app.auth.permissions import check_permission, filter_permissions_by_role, can_assign_permission
from app.utils.logging_config import get_logger
from app.settings import get_settings, Settings

logger = get_logger("auth_api")

router = APIRouter()


# Authentication endpoints

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: Dict[str, User] = Depends(get_user_store),
    request: Request = None
):
    """Login with username and password."""
    
    # Find user by username
    user = None
    for u in user_store.values():
        if u.username == form_data.username:
            user = u
            break
    
    if not user:
        logger.warning(
            "Login attempt with invalid username",
            username=form_data.username,
            ip_address=request.client.host if request and request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # For development, we'll accept any password for now
    # In production, this would verify against stored password hash
    if not user.is_active:
        logger.warning(
            "Login attempt by inactive user",
            user_id=user.id,
            username=user.username
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token = jwt_handler.create_access_token(user)
    
    # Update last login
    user.last_login = datetime.utcnow()
    
    logger.info(
        "User logged in successfully",
        user_id=user.id,
        username=user.username,
        role=user.role.value if hasattr(user.role, 'value') else user.role
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=jwt_handler.access_token_expire_minutes * 60,
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_api_user=user.is_api_user,
            created_at=user.created_at,
            last_login=user.last_login,
            permissions=list(user.permissions)
        )
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """Refresh access token."""
    
    # Create new access token
    access_token = jwt_handler.create_access_token(current_user)
    
    logger.info(
        "Token refreshed",
        user_id=current_user.id,
        username=current_user.username
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=jwt_handler.access_token_expire_minutes * 60,
        user=UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            role=current_user.role,
            is_active=current_user.is_active,
            is_api_user=current_user.is_api_user,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            permissions=list(current_user.permissions)
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        is_api_user=current_user.is_api_user,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        permissions=list(current_user.permissions)
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """Logout (client should discard token)."""
    
    logger.info(
        "User logged out",
        user_id=current_user.id,
        username=current_user.username
    )
    
    return {"message": "Logged out successfully"}


# User management endpoints

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(require_admin),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """List all users (admin only)."""
    
    users = []
    for user in user_store.values():
        users.append(UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_api_user=user.is_api_user,
            created_at=user.created_at,
            last_login=user.last_login,
            permissions=list(user.permissions)
        ))
    
    return users


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Create a new user (admin only)."""
    
    # Check if username already exists
    for existing_user in user_store.values():
        if existing_user.username == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
    
    # Validate permissions assignment
    for permission in user_data.permissions:
        if not can_assign_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot assign permission: {permission.value if hasattr(permission, 'value') else permission}"
            )
    
    # Create user
    user_id = str(uuid.uuid4())
    new_user = User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True,
        is_api_user=user_data.is_api_user,
        permissions=user_data.permissions,
        metadata=user_data.metadata,
        created_at=datetime.utcnow()
    )
    
    # Hash password if provided (for API users, password is optional)
    if user_data.password:
        # In a real implementation, store password hash
        pass  # For development, we skip password storage
    
    user_store[user_id] = new_user
    
    logger.info(
        "User created",
        new_user_id=user_id,
        new_username=user_data.username,
        new_role=user_data.role.value if hasattr(user_data.role, 'value') else user_data.role,
        created_by=current_user.username
    )
    
    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
        is_api_user=new_user.is_api_user,
        created_at=new_user.created_at,
        last_login=new_user.last_login,
        permissions=list(new_user.permissions)
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Get user by ID (admin only)."""
    
    user = user_store.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        is_api_user=user.is_api_user,
        created_at=user.created_at,
        last_login=user.last_login,
        permissions=list(user.permissions)
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: User = Depends(require_admin),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Update user (admin only)."""
    
    user = user_store.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    if user_data.email is not None:
        user.email = user_data.email
    if user_data.full_name is not None:
        user.full_name = user_data.full_name
    if user_data.role is not None:
        user.role = user_data.role
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    if user_data.permissions is not None:
        # Validate permissions assignment
        for permission in user_data.permissions:
            if not can_assign_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Cannot assign permission: {permission.value if hasattr(permission, 'value') else permission}"
                )
        user.permissions = user_data.permissions
    if user_data.metadata is not None:
        user.metadata.update(user_data.metadata)
    
    logger.info(
        "User updated",
        user_id=user_id,
        username=user.username,
        updated_by=current_user.username
    )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        is_api_user=user.is_api_user,
        created_at=user.created_at,
        last_login=user.last_login,
        permissions=list(user.permissions)
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Delete user (admin only)."""
    
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    user = user_store.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    del user_store[user_id]
    
    logger.info(
        "User deleted",
        user_id=user_id,
        username=user.username,
        deleted_by=current_user.username
    )
    
    return {"message": "User deleted successfully"}


# API Key management

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """Create an API key for the current user."""
    
    # Validate permissions
    for permission in api_key_data.permissions:
        if not check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot create API key with permission: {permission.value if hasattr(permission, 'value') else permission}"
            )
    
    # Generate API key
    api_key = jwt_handler.generate_api_key()
    api_key_hash = jwt_handler.hash_api_key(api_key)
    
    # Calculate expiration
    expires_at = None
    if api_key_data.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=api_key_data.expires_days)
    
    # For development, we'll store the API key hash in the user
    # In production, this would be stored in a separate API keys table
    current_user.api_key_hash = api_key_hash
    
    logger.info(
        "API key created",
        user_id=current_user.id,
        username=current_user.username,
        api_key_name=api_key_data.name,
        permissions=[p.value if hasattr(p, 'value') else p for p in api_key_data.permissions]
    )
    
    return APIKeyResponse(
        id=str(uuid.uuid4()),
        name=api_key_data.name,
        key=api_key,  # Only shown once
        permissions=api_key_data.permissions,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
        rate_limit=api_key_data.rate_limit
    )


# Utility endpoints

@router.get("/permissions")
async def list_permissions(
    current_user: User = Depends(get_current_active_user)
):
    """List available permissions for the current user's role."""
    
    available_permissions = filter_permissions_by_role(current_user.role)
    
    return {
        "role": current_user.role.value if hasattr(current_user.role, 'value') else current_user.role,
        "available_permissions": [p.value if hasattr(p, 'value') else p for p in available_permissions],
        "current_permissions": [p.value if hasattr(p, 'value') else p for p in current_user.permissions],
        "all_permissions": [p.value for p in Permission]
    }


@router.get("/roles")
async def list_roles():
    """List available user roles."""
    
    return {
        "roles": [
            {
                "name": role.value,
                "permissions": [p.value for p in filter_permissions_by_role(role)]
            }
            for role in UserRole
        ]
    }


@router.post("/test-auth")
async def test_authentication(
    current_user: User = Depends(get_current_active_user)
):
    """Test endpoint for authentication."""
    
    return {
        "message": "Authentication successful",
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role.value if hasattr(current_user.role, 'value') else current_user.role
        },
        "timestamp": datetime.utcnow().isoformat()
    } 