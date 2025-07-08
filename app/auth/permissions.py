"""Permission system for role-based access control."""

from typing import List, Optional, Set
from functools import wraps

from fastapi import HTTPException, status

from app.auth.models import User, UserRole, Permission, TokenData
from app.utils.logging_config import get_logger

logger = get_logger("permissions")


def check_permission(
    user: User, 
    required_permission: Permission
) -> bool:
    """Check if a user has a specific permission."""
    
    # Admin has all permissions
    if user.role == UserRole.ADMIN:
        return True
    
    # Get all user permissions (role-based + additional)
    user_permissions = get_user_permissions(user)
    
    has_permission = required_permission in user_permissions
    
    logger.debug(
        "Permission check",
        user_id=user.id,
        username=user.username,
        required_permission=required_permission.value,
        has_permission=has_permission
    )
    
    return has_permission


def check_permissions(
    user: User, 
    required_permissions: List[Permission],
    require_all: bool = True
) -> bool:
    """Check if a user has multiple permissions."""
    
    if user.role == UserRole.ADMIN:
        return True
    
    user_permissions = get_user_permissions(user)
    
    if require_all:
        # User must have ALL required permissions
        has_permissions = all(
            perm in user_permissions 
            for perm in required_permissions
        )
    else:
        # User must have ANY of the required permissions
        has_permissions = any(
            perm in user_permissions 
            for perm in required_permissions
        )
    
    logger.debug(
        "Multiple permission check",
        user_id=user.id,
        username=user.username,
        required_permissions=[p.value for p in required_permissions],
        require_all=require_all,
        has_permissions=has_permissions
    )
    
    return has_permissions


def get_user_permissions(user: User) -> Set[Permission]:
    """Get all permissions for a user based on role and additional permissions."""
    permissions = set()
    
    # Role-based permissions
    if user.role == UserRole.ADMIN:
        # Admins get all permissions
        permissions.update(Permission)
    
    elif user.role == UserRole.DEVELOPER:
        permissions.update([
            # Chat permissions
            Permission.CHAT_BASIC,
            Permission.CHAT_ADVANCED,
            Permission.CHAT_FILE_UPLOAD,
            
            # API permissions
            Permission.API_MODELS,
            Permission.API_PROVIDERS,
            Permission.API_STATUS,
            
            # Worker permissions
            Permission.WORKER_EXECUTE,
            Permission.WORKER_STATUS,
            Permission.WORKER_MANAGE,
            
            # File permissions
            Permission.FILE_PROCESS,
            Permission.FILE_ANALYZE,
            
            # Health and monitoring
            Permission.HEALTH_READ,
            Permission.METRICS_READ
        ])
    
    elif user.role == UserRole.USER:
        permissions.update([
            # Basic chat permissions
            Permission.CHAT_BASIC,
            Permission.CHAT_FILE_UPLOAD,
            
            # Basic API permissions
            Permission.API_MODELS,
            Permission.API_STATUS,
            
            # Worker status (read-only)
            Permission.WORKER_STATUS,
            
            # File processing
            Permission.FILE_PROCESS,
            
            # Health read
            Permission.HEALTH_READ
        ])
    
    elif user.role == UserRole.READONLY:
        permissions.update([
            # Read-only permissions
            Permission.API_STATUS,
            Permission.WORKER_STATUS,
            Permission.HEALTH_READ,
            Permission.METRICS_READ
        ])
    
    # Add additional user-specific permissions
    permissions.update(user.permissions)
    
    return permissions


def require_permission(required_permission: Permission):
    """Decorator to require a specific permission for an endpoint."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the current user in the function arguments
            current_user = None
            
            # Check function signature for current_user parameter
            for key, value in kwargs.items():
                if key == "current_user" and isinstance(value, User):
                    current_user = value
                    break
            
            if not current_user:
                # Check positional arguments
                for arg in args:
                    if isinstance(arg, User):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not check_permission(current_user, required_permission):
                logger.warning(
                    "Permission denied",
                    user_id=current_user.id,
                    username=current_user.username,
                    required_permission=required_permission.value,
                    user_role=current_user.role.value
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {required_permission.value}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_permissions(
    required_permissions: List[Permission], 
    require_all: bool = True
):
    """Decorator to require multiple permissions for an endpoint."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the current user in the function arguments
            current_user = None
            
            for key, value in kwargs.items():
                if key == "current_user" and isinstance(value, User):
                    current_user = value
                    break
            
            if not current_user:
                for arg in args:
                    if isinstance(arg, User):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not check_permissions(current_user, required_permissions, require_all):
                logger.warning(
                    "Multiple permissions denied",
                    user_id=current_user.id,
                    username=current_user.username,
                    required_permissions=[p.value for p in required_permissions],
                    require_all=require_all,
                    user_role=current_user.role.value
                )
                
                permission_names = [p.value for p in required_permissions]
                operator = "ALL" if require_all else "ANY"
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {operator} of {permission_names}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(required_role: UserRole):
    """Decorator to require a specific role for an endpoint."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the current user in the function arguments
            current_user = None
            
            for key, value in kwargs.items():
                if key == "current_user" and isinstance(value, User):
                    current_user = value
                    break
            
            if not current_user:
                for arg in args:
                    if isinstance(arg, User):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check role hierarchy
            if not has_role_or_higher(current_user.role, required_role):
                logger.warning(
                    "Role requirement not met",
                    user_id=current_user.id,
                    username=current_user.username,
                    user_role=current_user.role.value,
                    required_role=required_role.value
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role requirement not met. Required: {required_role.value} or higher"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def has_role_or_higher(user_role: UserRole, required_role: UserRole) -> bool:
    """Check if user has the required role or higher in the hierarchy."""
    
    # Define role hierarchy (higher number = more permissions)
    role_hierarchy = {
        UserRole.READONLY: 0,
        UserRole.USER: 1,
        UserRole.DEVELOPER: 2,
        UserRole.ADMIN: 3
    }
    
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    return user_level >= required_level


def check_token_permissions(
    token_data: TokenData, 
    required_permission: Permission
) -> bool:
    """Check if a token has a specific permission (for API key validation)."""
    
    # Check if permission is in token
    has_permission = required_permission in token_data.permissions
    
    logger.debug(
        "Token permission check",
        user_id=token_data.user_id,
        username=token_data.username,
        required_permission=required_permission.value,
        has_permission=has_permission
    )
    
    return has_permission


def filter_permissions_by_role(role: UserRole) -> List[Permission]:
    """Get all available permissions for a specific role."""
    
    if role == UserRole.ADMIN:
        return list(Permission)
    
    elif role == UserRole.DEVELOPER:
        return [
            Permission.CHAT_BASIC,
            Permission.CHAT_ADVANCED,
            Permission.CHAT_FILE_UPLOAD,
            Permission.API_MODELS,
            Permission.API_PROVIDERS,
            Permission.API_STATUS,
            Permission.WORKER_EXECUTE,
            Permission.WORKER_STATUS,
            Permission.WORKER_MANAGE,
            Permission.FILE_PROCESS,
            Permission.FILE_ANALYZE,
            Permission.HEALTH_READ,
            Permission.METRICS_READ
        ]
    
    elif role == UserRole.USER:
        return [
            Permission.CHAT_BASIC,
            Permission.CHAT_FILE_UPLOAD,
            Permission.API_MODELS,
            Permission.API_STATUS,
            Permission.WORKER_STATUS,
            Permission.FILE_PROCESS,
            Permission.HEALTH_READ
        ]
    
    elif role == UserRole.READONLY:
        return [
            Permission.API_STATUS,
            Permission.WORKER_STATUS,
            Permission.HEALTH_READ,
            Permission.METRICS_READ
        ]
    
    return []


def can_assign_permission(
    assigner: User, 
    permission: Permission
) -> bool:
    """Check if a user can assign a specific permission to another user."""
    
    # Only admins can assign admin permissions
    admin_permissions = [
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
        Permission.ADMIN_LOGS
    ]
    
    if permission in admin_permissions and assigner.role != UserRole.ADMIN:
        return False
    
    # Users can only assign permissions they have
    return check_permission(assigner, permission) 