"""Authentication dependencies for FastAPI."""

import time
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth.models import User, UserRole, Permission, TokenData
from app.auth.jwt_handler import JWTHandler
from app.auth.permissions import check_permission
from app.settings import get_settings, Settings
from app.utils.logging_config import get_logger

logger = get_logger("auth_dependencies")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)

# Global instances
_jwt_handler: Optional[JWTHandler] = None
_user_store: Dict[str, User] = {}  # In-memory user store for development


def get_jwt_handler(settings: Settings = Depends(get_settings)) -> JWTHandler:
    """Get or create JWT handler instance."""
    global _jwt_handler
    
    if _jwt_handler is None:
        _jwt_handler = JWTHandler(settings)
    
    return _jwt_handler


def get_user_store() -> Dict[str, User]:
    """Get the user store (in-memory for development)."""
    global _user_store
    
    # Initialize with default admin user if empty
    if not _user_store:
        admin_user = User(
            id="admin-001",
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.utcnow()
        )
        _user_store[admin_user.id] = admin_user
        
        # Create a default developer user
        dev_user = User(
            id="dev-001",
            username="developer",
            email="dev@example.com",
            full_name="Developer User",
            role=UserRole.DEVELOPER,
            is_active=True,
            created_at=datetime.utcnow()
        )
        _user_store[dev_user.id] = dev_user
    
    return _user_store


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: Dict[str, User] = Depends(get_user_store)
) -> Optional[User]:
    """Get current user from JWT token (optional - returns None if not authenticated)."""
    
    # Check for Bearer token
    if credentials and credentials.scheme.lower() == "bearer":
        token_data = jwt_handler.decode_token(credentials.credentials)
        
        if token_data:
            user = user_store.get(token_data.user_id)
            if user and user.is_active:
                # Update last login
                user.last_login = datetime.utcnow()
                
                logger.debug(
                    "User authenticated via JWT",
                    user_id=user.id,
                    username=user.username,
                    role=user.role.value if hasattr(user.role, 'value') else user.role
                )
                
                return user
    
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = await authenticate_api_key(api_key, jwt_handler, user_store)
        if user:
            return user
    
    # Check for API key in query parameters (less secure, but useful for testing)
    api_key = request.query_params.get("api_key")
    if api_key:
        user = await authenticate_api_key(api_key, jwt_handler, user_store)
        if user:
            return user
    
    return None


async def get_current_user(
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Get current user from JWT token (required - raises exception if not authenticated)."""
    
    if not current_user:
        logger.warning("Authentication required but no valid credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (checks if user is active)."""
    
    if not current_user.is_active:
        logger.warning(
            "Inactive user attempted access",
            user_id=current_user.id,
            username=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return current_user


async def authenticate_api_key(
    api_key: str,
    jwt_handler: JWTHandler,
    user_store: Dict[str, User]
) -> Optional[User]:
    """Authenticate a user via API key."""
    
    # In a real implementation, this would query a database
    # For now, we'll check against stored API key hashes
    
    for user in user_store.values():
        if user.api_key_hash:
            if jwt_handler.verify_api_key(api_key, user.api_key_hash):
                logger.debug(
                    "User authenticated via API key",
                    user_id=user.id,
                    username=user.username,
                    role=user.role.value if hasattr(user.role, 'value') else user.role
                )
                
                # Update last login
                user.last_login = datetime.utcnow()
                
                return user
    
    return None


def require_api_key(
    api_key: str = None,
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Dependency that requires a valid API key."""
    
    async def _require_api_key(request: Request) -> User:
        # Check for API key in various locations
        api_key = None
        
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        
        # Check Authorization header with API key format
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]  # Remove "ApiKey " prefix
        
        # Check query parameter (for testing)
        if not api_key:
            api_key = request.query_params.get("api_key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        user = await authenticate_api_key(api_key, jwt_handler, user_store)
        
        if not user:
            logger.warning("Invalid API key used", api_key_prefix=api_key[:8] + "...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return user
    
    return _require_api_key


def require_permission_dependency(permission: Permission):
    """Create a dependency that requires a specific permission."""
    
    async def _require_permission(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not check_permission(current_user, permission):
            logger.warning(
                "Permission denied in dependency",
                user_id=current_user.id,
                username=current_user.username,
                required_permission=permission.value if hasattr(permission, 'value') else permission,
                user_role=current_user.role.value if hasattr(current_user.role, 'value') else current_user.role
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission.value if hasattr(permission, 'value') else permission}"
            )
        
        return current_user
    
    return _require_permission


def require_role_dependency(role: UserRole):
    """Create a dependency that requires a specific role."""
    
    async def _require_role(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        from app.auth.permissions import has_role_or_higher
        
        if not has_role_or_higher(current_user.role, role):
            logger.warning(
                "Role requirement not met in dependency",
                user_id=current_user.id,
                username=current_user.username,
                user_role=current_user.role.value if hasattr(current_user.role, 'value') else current_user.role,
                required_role=role.value if hasattr(role, 'value') else role
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role requirement not met. Required: {role.value if hasattr(role, 'value') else role} or higher"
            )
        
        return current_user
    
    return _require_role


# Common dependency combinations
require_admin = require_role_dependency(UserRole.ADMIN)
require_developer = require_role_dependency(UserRole.DEVELOPER)

require_chat_permission = require_permission_dependency(Permission.CHAT_BASIC)
require_worker_permission = require_permission_dependency(Permission.WORKER_EXECUTE)
require_admin_permission = require_permission_dependency(Permission.ADMIN_USERS)


async def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract request context for logging and audit purposes."""
    
    return {
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent"),
        "method": request.method,
        "url": str(request.url),
        "timestamp": time.time()
    }


def create_user_for_testing(
    username: str,
    role: UserRole = UserRole.USER,
    permissions: list[Permission] = None,
    user_store: Dict[str, User] = Depends(get_user_store)
) -> User:
    """Create a user for testing purposes."""
    
    if permissions is None:
        permissions = []
    
    user_id = f"test-{username}-{int(time.time())}"
    
    user = User(
        id=user_id,
        username=username,
        email=f"{username}@test.local",
        full_name=f"Test User {username}",
        role=role,
        is_active=True,
        permissions=permissions,
        created_at=datetime.utcnow()
    )
    
    user_store[user_id] = user
    
    logger.info(
        "Created test user",
        user_id=user_id,
        username=username,
        role=role.value if hasattr(role, 'value') else role
    )
    
    return user


async def setup_development_users(
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: Dict[str, User] = Depends(get_user_store)
):
    """Setup development users with API keys."""
    
    # Create API key for admin user
    admin_user = user_store.get("admin-001")
    if admin_user and not admin_user.api_key_hash:
        api_key = jwt_handler.generate_api_key("admin")
        admin_user.api_key_hash = jwt_handler.hash_api_key(api_key)
        
        logger.info(
            "Generated API key for admin user",
            user_id=admin_user.id,
            api_key_prefix=api_key[:12] + "..."
        )
    
    # Create API key for developer user
    dev_user = user_store.get("dev-001")
    if dev_user and not dev_user.api_key_hash:
        api_key = jwt_handler.generate_api_key("dev")
        dev_user.api_key_hash = jwt_handler.hash_api_key(api_key)
        
        logger.info(
            "Generated API key for developer user",
            user_id=dev_user.id,
            api_key_prefix=api_key[:12] + "..."
        ) 