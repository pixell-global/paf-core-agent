"""LLM Provider Manager - orchestrates all LLM providers."""

import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass

from app.llm_providers.base import (
    LLMProvider, LLMProviderType, LLMRequest, LLMResponse,
    LLMProviderError, LLMProviderNotAvailableError
)
from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.claude_provider import ClaudeProvider
from app.llm_providers.bedrock_provider import BedrockProvider
from app.settings import Settings
from app.utils.logging_config import get_logger


@dataclass
class ProviderModel:
    """Model information with provider mapping."""
    name: str
    provider: LLMProviderType
    display_name: str
    description: str
    context_length: Optional[int] = None
    is_preferred: bool = False


class LLMProviderManager:
    """Manages all LLM providers with fallback and selection logic."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("llm_provider_manager")
        
        # Initialize providers
        self.providers: Dict[LLMProviderType, LLMProvider] = {
            LLMProviderType.OPENAI: OpenAIProvider(settings),
            LLMProviderType.ANTHROPIC: ClaudeProvider(settings),
            LLMProviderType.BEDROCK: BedrockProvider(settings)
        }
        
        # Provider priority order (for fallback)
        self.provider_priority = [
            LLMProviderType.OPENAI,
            LLMProviderType.ANTHROPIC,
            LLMProviderType.BEDROCK
        ]
        
        # Model catalog
        self.model_catalog = self._build_model_catalog()
    
    def _build_model_catalog(self) -> Dict[str, ProviderModel]:
        """Build comprehensive model catalog."""
        catalog = {}
        
        # OpenAI models
        openai_models = [
            ProviderModel("gpt-4o", LLMProviderType.OPENAI, "GPT-4o", "Latest GPT-4 optimized model", 128000, True),
            ProviderModel("gpt-4o-mini", LLMProviderType.OPENAI, "GPT-4o Mini", "Smaller, faster GPT-4 model", 128000),
            ProviderModel("gpt-4-turbo", LLMProviderType.OPENAI, "GPT-4 Turbo", "Latest GPT-4 with vision", 128000),
            ProviderModel("gpt-4", LLMProviderType.OPENAI, "GPT-4", "Standard GPT-4 model", 8192),
            ProviderModel("gpt-3.5-turbo", LLMProviderType.OPENAI, "GPT-3.5 Turbo", "Fast and efficient model", 16385),
        ]
        
        # Claude models
        claude_models = [
            ProviderModel("claude-3.5-sonnet", LLMProviderType.ANTHROPIC, "Claude 3.5 Sonnet", "Latest Claude model with improved capabilities", 200000, True),
            ProviderModel("claude-3.5-haiku", LLMProviderType.ANTHROPIC, "Claude 3.5 Haiku", "Fast and efficient Claude model", 200000),
            ProviderModel("claude-3-opus", LLMProviderType.ANTHROPIC, "Claude 3 Opus", "Most capable Claude model", 200000),
            ProviderModel("claude-3-sonnet", LLMProviderType.ANTHROPIC, "Claude 3 Sonnet", "Balanced Claude model", 200000),
            ProviderModel("claude-3-haiku", LLMProviderType.ANTHROPIC, "Claude 3 Haiku", "Fastest Claude model", 200000),
        ]
        
        # Bedrock models
        bedrock_models = [
            ProviderModel("claude-3.5-sonnet", LLMProviderType.BEDROCK, "Claude 3.5 Sonnet (Bedrock)", "Claude 3.5 Sonnet via AWS Bedrock", 200000),
            ProviderModel("claude-3-sonnet", LLMProviderType.BEDROCK, "Claude 3 Sonnet (Bedrock)", "Claude 3 Sonnet via AWS Bedrock", 200000),
            ProviderModel("claude-3-haiku", LLMProviderType.BEDROCK, "Claude 3 Haiku (Bedrock)", "Claude 3 Haiku via AWS Bedrock", 200000),
            ProviderModel("titan-text", LLMProviderType.BEDROCK, "Amazon Titan Text", "Amazon's foundation model", 32000),
            ProviderModel("llama2-70b", LLMProviderType.BEDROCK, "Llama 2 70B", "Meta's Llama 2 70B model", 4096),
            ProviderModel("llama2-13b", LLMProviderType.BEDROCK, "Llama 2 13B", "Meta's Llama 2 13B model", 4096),
        ]
        
        # Add all models to catalog
        for model_list in [openai_models, claude_models, bedrock_models]:
            for model in model_list:
                # Use provider prefix to avoid naming conflicts
                key = f"{model.provider.value}:{model.name}"
                catalog[key] = model
                
                # Also add without provider prefix for backwards compatibility
                if model.name not in catalog:
                    catalog[model.name] = model
        
        return catalog
    
    async def get_available_providers(self) -> Dict[LLMProviderType, Dict[str, Any]]:
        """Get all providers and their availability status."""
        provider_status = {}
        
        for provider_type, provider in self.providers.items():
            try:
                is_available = await provider.is_available()
                models = await provider.get_available_models() if is_available else []
                health = await provider.health_check()
                
                provider_status[provider_type] = {
                    "available": is_available,
                    "models": models,
                    "health": health
                }
            except Exception as e:
                self.logger.error(f"Error checking provider {provider_type}: {e}")
                provider_status[provider_type] = {
                    "available": False,
                    "models": [],
                    "health": {"status": "error", "error": str(e)}
                }
        
        return provider_status
    
    async def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all available models across all providers."""
        available_providers = await self.get_available_providers()
        models = []
        
        for model_key, model_info in self.model_catalog.items():
            # Skip provider-prefixed entries
            if ":" in model_key:
                continue
            
            provider_available = available_providers.get(model_info.provider, {}).get("available", False)
            provider_models = available_providers.get(model_info.provider, {}).get("models", [])
            
            is_available = provider_available and model_info.name in provider_models
            
            models.append({
                "name": model_info.name,
                "display_name": model_info.display_name,
                "description": model_info.description,
                "provider": model_info.provider.value,
                "context_length": model_info.context_length,
                "is_preferred": model_info.is_preferred,
                "available": is_available
            })
        
        # Sort by availability, then by preference, then by name
        models.sort(key=lambda m: (not m["available"], not m["is_preferred"], m["name"]))
        return models
    
    def _get_provider_for_model(self, model: str) -> Optional[LLMProvider]:
        """Get the provider for a specific model."""
        if model in self.model_catalog:
            provider_type = self.model_catalog[model].provider
            return self.providers.get(provider_type)
        
        # Try to find by provider prefix
        for provider_type in self.providers:
            prefixed_key = f"{provider_type.value}:{model}"
            if prefixed_key in self.model_catalog:
                return self.providers.get(provider_type)
        
        return None
    
    async def _get_fallback_providers(self, original_model: str) -> List[tuple[LLMProvider, str]]:
        """Get fallback providers and models when the original fails."""
        fallbacks = []
        
        # Get original provider
        original_provider = self._get_provider_for_model(original_model)
        if original_provider:
            original_provider_type = original_provider.provider_type
        else:
            original_provider_type = None
        
        # Try other providers in priority order
        for provider_type in self.provider_priority:
            if provider_type == original_provider_type:
                continue  # Skip original provider
            
            provider = self.providers[provider_type]
            if not await provider.is_available():
                continue
            
            # Get available models for this provider
            available_models = await provider.get_available_models()
            if not available_models:
                continue
            
            # Prefer similar models (e.g., if original was GPT-4, prefer Claude-3 Opus)
            fallback_model = self._select_similar_model(original_model, available_models)
            if fallback_model:
                fallbacks.append((provider, fallback_model))
        
        return fallbacks
    
    def _select_similar_model(self, original_model: str, available_models: List[str]) -> Optional[str]:
        """Select a similar model from available options."""
        # Model similarity mapping
        model_similarity = {
            "gpt-4": ["claude-3-opus", "claude-3-sonnet"],
            "gpt-4o": ["claude-3.5-sonnet", "claude-3-opus"],
            "gpt-4-turbo": ["claude-3.5-sonnet", "claude-3-opus"],
            "gpt-3.5-turbo": ["claude-3-haiku", "claude-3-sonnet"],
            "claude-3-opus": ["gpt-4", "gpt-4o"],
            "claude-3.5-sonnet": ["gpt-4o", "gpt-4-turbo"],
            "claude-3-sonnet": ["gpt-4", "gpt-3.5-turbo"],
            "claude-3-haiku": ["gpt-3.5-turbo", "gpt-4o-mini"]
        }
        
        # Try to find similar model
        if original_model in model_similarity:
            for similar_model in model_similarity[original_model]:
                if similar_model in available_models:
                    return similar_model
        
        # Fallback to first available model
        return available_models[0] if available_models else None
    
    async def stream_completion(
        self, 
        request: LLMRequest,
        enable_fallback: bool = True
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream completion with automatic fallback."""
        provider = self._get_provider_for_model(request.model)
        
        if not provider:
            raise LLMProviderError(f"No provider found for model: {request.model}")
        
        self.logger.info(
            "Starting LLM completion",
            model=request.model,
            provider=provider.provider_type.value,
            request_id=request.request_id
        )
        
        try:
            # Try primary provider
            async for response in provider.stream_completion(request):
                yield response
            return
        
        except LLMProviderNotAvailableError as e:
            self.logger.warning(f"Provider {provider.provider_type} not available: {e}")
        except Exception as e:
            self.logger.error(f"Provider {provider.provider_type} failed: {e}")
        
        # Try fallback providers if enabled
        if enable_fallback:
            fallback_providers = await self._get_fallback_providers(request.model)
            
            for fallback_provider, fallback_model in fallback_providers:
                self.logger.info(
                    "Trying fallback provider",
                    original_model=request.model,
                    fallback_provider=fallback_provider.provider_type.value,
                    fallback_model=fallback_model,
                    request_id=request.request_id
                )
                
                try:
                    # Create new request with fallback model
                    fallback_request = LLMRequest(
                        model=fallback_model,
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=request.stream,
                        system_prompt=request.system_prompt,
                        request_id=request.request_id
                    )
                    
                    async for response in fallback_provider.stream_completion(fallback_request):
                        yield response
                    return
                
                except Exception as e:
                    self.logger.warning(f"Fallback provider {fallback_provider.provider_type} failed: {e}")
                    continue
        
        # If all providers failed
        raise LLMProviderError("All LLM providers failed")
    
    async def get_completion(
        self, 
        request: LLMRequest,
        enable_fallback: bool = True
    ) -> LLMResponse:
        """Get completion with automatic fallback."""
        provider = self._get_provider_for_model(request.model)
        
        if not provider:
            raise LLMProviderError(f"No provider found for model: {request.model}")
        
        self.logger.info(
            "Starting LLM completion (non-streaming)",
            model=request.model,
            provider=provider.provider_type.value,
            request_id=request.request_id
        )
        
        try:
            # Try primary provider
            return await provider.get_completion(request)
        
        except LLMProviderNotAvailableError as e:
            self.logger.warning(f"Provider {provider.provider_type} not available: {e}")
        except Exception as e:
            self.logger.error(f"Provider {provider.provider_type} failed: {e}")
        
        # Try fallback providers if enabled
        if enable_fallback:
            fallback_providers = await self._get_fallback_providers(request.model)
            
            for fallback_provider, fallback_model in fallback_providers:
                self.logger.info(
                    "Trying fallback provider",
                    original_model=request.model,
                    fallback_provider=fallback_provider.provider_type.value,
                    fallback_model=fallback_model,
                    request_id=request.request_id
                )
                
                try:
                    # Create new request with fallback model
                    fallback_request = LLMRequest(
                        model=fallback_model,
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=request.stream,
                        system_prompt=request.system_prompt,
                        request_id=request.request_id
                    )
                    
                    return await fallback_provider.get_completion(fallback_request)
                
                except Exception as e:
                    self.logger.warning(f"Fallback provider {fallback_provider.provider_type} failed: {e}")
                    continue
        
        # If all providers failed
        raise LLMProviderError("All LLM providers failed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        provider_status = await self.get_available_providers()
        
        total_providers = len(self.providers)
        healthy_providers = sum(
            1 for status in provider_status.values() 
            if status.get("available", False)
        )
        
        overall_status = "healthy" if healthy_providers > 0 else "unhealthy"
        
        return {
            "status": overall_status,
            "providers": provider_status,
            "summary": {
                "total_providers": total_providers,
                "healthy_providers": healthy_providers,
                "unhealthy_providers": total_providers - healthy_providers
            }
        } 