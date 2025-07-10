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