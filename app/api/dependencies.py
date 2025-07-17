"""API dependencies for FastAPI endpoints."""

from typing import Optional
from fastapi import Depends

from app.core.upee_engine import UPEEEngine
from app.settings import Settings

# Global UPEE engine instance
_upee_engine: Optional[UPEEEngine] = None


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


async def get_upee_engine(settings: Settings = Depends(get_settings)) -> UPEEEngine:
    """
    Get or create the UPEE engine instance.
    
    This ensures we have a singleton UPEE engine across the application.
    """
    global _upee_engine
    
    if _upee_engine is None:
        _upee_engine = UPEEEngine(settings)
        await _upee_engine.startup()
    
    return _upee_engine