"""JWT token handling and validation."""

import time
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from passlib.context import CryptContext

from app.auth.models import User, TokenData, UserRole, Permission
from app.settings import Settings
from app.utils.logging_config import get_logger

logger = get_logger("jwt_handler")


class JWTHandler:
    """JWT token handler for authentication."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.secret_key = settings.jwt_secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60 * 24  # 24 hours
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # HMAC for API keys
        self.hmac_secret = settings.hmac_secret_key.encode('utf-8')
    
    def create_access_token(
        self, 
        user: User, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token for a user."""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # Get user permissions (role-based + additional)
        permissions = self._get_user_permissions(user)
        
        token_data = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value if hasattr(user.role, 'value') else user.role,
            "permissions": [p.value if hasattr(p, 'value') else p for p in permissions],
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "access"
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        
        logger.info(
            "Created access token",
            user_id=user.id,
            username=user.username,
            role=user.role.value if hasattr(user.role, 'value') else user.role,
            expires_at=expire.isoformat()
        )
        
        return token
    
    def decode_token(self, token: str) -> Optional[TokenData]:
        """Decode and validate a JWT token."""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Validate token type
            if payload.get("type") != "access":
                logger.warning("Invalid token type", token_type=payload.get("type"))
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if not exp or datetime.utcnow().timestamp() > exp:
                logger.warning("Token expired", exp=exp)
                return None
            
            # Extract token data
            role_value = payload.get("role")
            role = UserRole(role_value) if isinstance(role_value, str) else role_value
            
            permissions_list = payload.get("permissions", [])
            permissions = []
            for p in permissions_list:
                try:
                    permissions.append(Permission(p) if isinstance(p, str) else p)
                except ValueError:
                    # Skip invalid permissions
                    continue
            
            token_data = TokenData(
                user_id=payload.get("user_id"),
                username=payload.get("username"),
                role=role,
                permissions=permissions,
                exp=exp,
                iat=payload.get("iat", time.time())
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token signature expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token decode error", error=str(e))
            return None
    
    def generate_api_key(self, prefix: str = "paf") -> str:
        """Generate a secure API key."""
        # Generate 32 random bytes
        random_bytes = secrets.token_bytes(32)
        
        # Create API key with prefix
        api_key = f"{prefix}_{secrets.token_urlsafe(32)}"
        
        logger.info("Generated new API key", prefix=prefix)
        
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(
            api_key.encode('utf-8') + self.hmac_secret
        ).hexdigest()
    
    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify an API key against its stored hash."""
        computed_hash = self.hash_api_key(api_key)
        return secrets.compare_digest(computed_hash, stored_hash)
    
    def hash_password(self, password: str) -> str:
        """Hash a password for secure storage."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _get_user_permissions(self, user: User) -> list[Permission]:
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
        
        return list(permissions)
    
    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token (longer-lived)."""
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days
        
        token_data = {
            "user_id": user.id,
            "username": user.username,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "refresh"
        }
        
        return jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
    
    def decode_refresh_token(self, token: str) -> Optional[str]:
        """Decode a refresh token and return user_id if valid."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            return payload.get("user_id")
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid refresh token")
            return None
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)
    
    def verify_token_signature(self, token: str) -> bool:
        """Verify if a token has a valid signature without decoding."""
        try:
            jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Only verify signature
            )
            return True
        except jwt.InvalidTokenError:
            return False 