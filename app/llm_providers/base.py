"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from app.utils.logging_config import get_logger


class LLMProviderType(str, Enum):
    """LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    BEDROCK = "bedrock"


@dataclass
class LLMRequest:
    """LLM request parameters."""
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = True
    system_prompt: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response chunk."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    token_count: Optional[int] = None
    is_complete: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMUsage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, provider_type: LLMProviderType):
        self.provider_type = provider_type
        self.logger = get_logger(f"llm_provider_{provider_type}")
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def validate_model(self, model: str) -> bool:
        """Validate if a model is supported by this provider."""
        pass
    
    @abstractmethod
    async def stream_completion(
        self, 
        request: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Stream completion from the LLM provider.
        
        Args:
            request: LLM request parameters
            
        Yields:
            LLMResponse chunks
        """
        pass
    
    @abstractmethod
    async def get_completion(
        self, 
        request: LLMRequest
    ) -> LLMResponse:
        """
        Get a complete response (non-streaming).
        
        Args:
            request: LLM request parameters
            
        Returns:
            Complete LLMResponse
        """
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4
    
    def _format_system_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts according to provider conventions."""
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        return user_prompt
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the provider."""
        try:
            is_available = await self.is_available()
            if not is_available:
                return {
                    "status": "unavailable",
                    "error": "Provider not configured or unavailable"
                }
            
            # Try to get models list as a basic health check
            models = await self.get_available_models()
            return {
                "status": "healthy",
                "models_count": len(models),
                "provider": self.provider_type.value
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": self.provider_type.value
            }


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMProviderNotAvailableError(LLMProviderError):
    """Raised when provider is not available or configured."""
    pass


class LLMProviderRateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    pass


class LLMProviderAuthError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class LLMProviderModelError(LLMProviderError):
    """Raised when model is not supported or invalid."""
    pass 