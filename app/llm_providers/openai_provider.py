"""OpenAI LLM provider implementation."""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.llm_providers.base import (
    LLMProvider, LLMProviderType, LLMRequest, LLMResponse, LLMUsage,
    LLMProviderError, LLMProviderNotAvailableError, LLMProviderRateLimitError,
    LLMProviderAuthError, LLMProviderModelError
)
from app.settings import Settings


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(LLMProviderType.OPENAI)
        self.settings = settings
        self.client: Optional[AsyncOpenAI] = None
        
        # OpenAI model mappings
        self.model_map = {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini"
        }
        
        if self.settings.openai_api_key and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
    
    async def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return (
            OPENAI_AVAILABLE and 
            self.settings.openai_api_key is not None and 
            self.client is not None
        )
    
    async def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        if not await self.is_available():
            return []
        
        return list(self.model_map.keys())
    
    async def validate_model(self, model: str) -> bool:
        """Validate if a model is supported."""
        return model in self.model_map
    
    def _get_openai_model(self, model: str) -> str:
        """Get the actual OpenAI model name."""
        return self.model_map.get(model, model)
    
    def _format_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Format request into OpenAI messages format."""
        messages = []
        
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": request.prompt
        })
        
        return messages
    
    async def stream_completion(
        self, 
        request: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream completion from OpenAI."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("OpenAI provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by OpenAI")
        
        openai_model = self._get_openai_model(request.model)
        messages = self._format_messages(request)
        
        self.logger.info(
            "Starting OpenAI streaming completion",
            model=openai_model,
            request_id=request.request_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        try:
            # Create streaming completion
            stream = await self.client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            content_buffer = ""
            total_tokens = 0
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    if choice.delta and choice.delta.content:
                        content = choice.delta.content
                        content_buffer += content
                        
                        # Yield content chunk
                        yield LLMResponse(
                            content=content,
                            model=request.model,
                            provider=self.provider_type.value,
                            token_count=self._estimate_tokens(content),
                            is_complete=False,
                            metadata={
                                "openai_model": openai_model,
                                "request_id": request.request_id
                            }
                        )
                    
                    # Handle completion
                    if choice.finish_reason:
                        # Check for usage information in the final chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            total_tokens = chunk.usage.total_tokens
                        
                        # Yield final completion chunk
                        yield LLMResponse(
                            content="",
                            model=request.model,
                            provider=self.provider_type.value,
                            finish_reason=choice.finish_reason,
                            token_count=total_tokens or self._estimate_tokens(content_buffer),
                            is_complete=True,
                            metadata={
                                "openai_model": openai_model,
                                "request_id": request.request_id,
                                "total_content_length": len(content_buffer)
                            }
                        )
                        break
        
        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            raise LLMProviderRateLimitError(f"OpenAI rate limit: {str(e)}")
        
        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication failed: {e}")
            raise LLMProviderAuthError(f"OpenAI auth error: {str(e)}")
        
        except openai.BadRequestError as e:
            self.logger.error(f"OpenAI bad request: {e}")
            raise LLMProviderModelError(f"OpenAI request error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"OpenAI error: {str(e)}")
    
    async def get_completion(
        self, 
        request: LLMRequest
    ) -> LLMResponse:
        """Get complete response from OpenAI (non-streaming)."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("OpenAI provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by OpenAI")
        
        openai_model = self._get_openai_model(request.model)
        messages = self._format_messages(request)
        
        self.logger.info(
            "Starting OpenAI completion",
            model=openai_model,
            request_id=request.request_id
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider_type.value,
                finish_reason=choice.finish_reason,
                token_count=response.usage.total_tokens if response.usage else self._estimate_tokens(content),
                is_complete=True,
                metadata={
                    "openai_model": openai_model,
                    "request_id": request.request_id,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0
                }
            )
        
        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            raise LLMProviderRateLimitError(f"OpenAI rate limit: {str(e)}")
        
        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication failed: {e}")
            raise LLMProviderAuthError(f"OpenAI auth error: {str(e)}")
        
        except openai.BadRequestError as e:
            self.logger.error(f"OpenAI bad request: {e}")
            raise LLMProviderModelError(f"OpenAI request error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"OpenAI error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform OpenAI-specific health check."""
        base_health = await super().health_check()
        
        if not OPENAI_AVAILABLE:
            return {
                **base_health,
                "status": "unavailable",
                "error": "OpenAI package not installed"
            }
        
        if not self.settings.openai_api_key:
            return {
                **base_health,
                "status": "unavailable", 
                "error": "OpenAI API key not configured"
            }
        
        try:
            # Test with a minimal API call
            models = await self.get_available_models()
            return {
                **base_health,
                "available_models": models,
                "api_key_configured": bool(self.settings.openai_api_key)
            }
        except Exception as e:
            return {
                **base_health,
                "status": "unhealthy",
                "error": str(e)
            } 