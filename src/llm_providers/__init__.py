"""LLM Providers package - multi-provider LLM integration."""

from src.llm_providers.base import (
    LLMProvider,
    LLMProviderType,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMProviderError,
    LLMProviderNotAvailableError,
    LLMProviderRateLimitError,
    LLMProviderAuthError,
    LLMProviderModelError
)

from src.llm_providers.manager import LLMProviderManager, ProviderModel

# Conditional imports based on availability
try:
    from src.llm_providers.openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from src.llm_providers.claude_provider import ClaudeProvider
except ImportError:
    ClaudeProvider = None

try:
    from src.llm_providers.bedrock_provider import BedrockProvider
except ImportError:
    BedrockProvider = None


__all__ = [
    # Base classes
    "LLMProvider",
    "LLMProviderType", 
    "LLMMessage",
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