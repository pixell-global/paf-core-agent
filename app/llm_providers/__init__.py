"""LLM Providers package - multi-provider LLM integration."""

from app.llm_providers.base import (
    LLMProvider,
    LLMProviderType,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMProviderError,
    LLMProviderNotAvailableError,
    LLMProviderRateLimitError,
    LLMProviderAuthError,
    LLMProviderModelError
)

from app.llm_providers.manager import LLMProviderManager, ProviderModel

# Conditional imports based on availability
try:
    from app.llm_providers.openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from app.llm_providers.claude_provider import ClaudeProvider
except ImportError:
    ClaudeProvider = None

try:
    from app.llm_providers.bedrock_provider import BedrockProvider
except ImportError:
    BedrockProvider = None


__all__ = [
    # Base classes
    "LLMProvider",
    "LLMProviderType", 
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    
    # Exceptions
    "LLMProviderError",
    "LLMProviderNotAvailableError",
    "LLMProviderRateLimitError",
    "LLMProviderAuthError",
    "LLMProviderModelError",
    
    # Manager
    "LLMProviderManager",
    "ProviderModel",
    
    # Providers (conditionally available)
    "OpenAIProvider",
    "ClaudeProvider", 
    "BedrockProvider"
] 