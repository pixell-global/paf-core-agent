"""Anthropic Claude LLM provider implementation."""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from app.llm_providers.base import (
    LLMProvider, LLMProviderType, LLMRequest, LLMResponse, LLMUsage,
    LLMProviderError, LLMProviderNotAvailableError, LLMProviderRateLimitError,
    LLMProviderAuthError, LLMProviderModelError
)
from app.settings import Settings


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(LLMProviderType.ANTHROPIC)
        self.settings = settings
        self.client: Optional[AsyncAnthropic] = None
        
        # Claude model mappings
        self.model_map = {
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3.5-haiku": "claude-3-5-haiku-20241022"
        }
        
        if self.settings.anthropic_api_key and ANTHROPIC_AVAILABLE:
            self.client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
    
    async def is_available(self) -> bool:
        """Check if Claude provider is available."""
        return (
            ANTHROPIC_AVAILABLE and 
            self.settings.anthropic_api_key is not None and 
            self.client is not None
        )
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Claude models."""
        if not await self.is_available():
            return []
        
        return list(self.model_map.keys())
    
    async def validate_model(self, model: str) -> bool:
        """Validate if a model is supported."""
        return model in self.model_map
    
    def _get_claude_model(self, model: str) -> str:
        """Get the actual Claude model name."""
        return self.model_map.get(model, model)
    
    def _format_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Format request into Claude messages format."""
        messages = []
        
        # Claude doesn't use system messages in the messages array
        # System prompts are handled separately
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        
        return messages
    
    async def stream_completion(
        self, 
        request: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream completion from Claude."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("Claude provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by Claude")
        
        claude_model = self._get_claude_model(request.model)
        messages = self._format_messages(request)
        
        self.logger.info(
            "Starting Claude streaming completion",
            model=claude_model,
            request_id=request.request_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        try:
            # Prepare kwargs for the API call
            kwargs = {
                "model": claude_model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens or 50000,  # Much more aggressive fallback
                "stream": True
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            
            # Create streaming completion
            stream = await self.client.messages.create(**kwargs)
            
            content_buffer = ""
            input_tokens = 0
            output_tokens = 0
            
            async for event in stream:
                if event.type == "message_start":
                    # Get token usage from message start
                    if hasattr(event.message, 'usage'):
                        input_tokens = event.message.usage.input_tokens
                
                elif event.type == "content_block_delta":
                    # Handle content delta
                    if hasattr(event.delta, 'text'):
                        content = event.delta.text
                        content_buffer += content
                        
                        # Yield content chunk
                        yield LLMResponse(
                            content=content,
                            model=request.model,
                            provider=self.provider_type.value,
                            token_count=self._estimate_tokens(content),
                            is_complete=False,
                            metadata={
                                "claude_model": claude_model,
                                "request_id": request.request_id
                            }
                        )
                
                elif event.type == "message_delta":
                    # Handle usage updates
                    if hasattr(event.usage, 'output_tokens'):
                        output_tokens = event.usage.output_tokens
                
                elif event.type == "message_stop":
                    # Handle completion
                    total_tokens = input_tokens + output_tokens
                    
                    # Yield final completion chunk
                    yield LLMResponse(
                        content="",
                        model=request.model,
                        provider=self.provider_type.value,
                        finish_reason="stop",
                        token_count=total_tokens,
                        is_complete=True,
                        metadata={
                            "claude_model": claude_model,
                            "request_id": request.request_id,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_content_length": len(content_buffer)
                        }
                    )
                    break
        
        except anthropic.RateLimitError as e:
            self.logger.error(f"Claude rate limit exceeded: {e}")
            raise LLMProviderRateLimitError(f"Claude rate limit: {str(e)}")
        
        except anthropic.AuthenticationError as e:
            self.logger.error(f"Claude authentication failed: {e}")
            raise LLMProviderAuthError(f"Claude auth error: {str(e)}")
        
        except anthropic.BadRequestError as e:
            self.logger.error(f"Claude bad request: {e}")
            raise LLMProviderModelError(f"Claude request error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Claude completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"Claude error: {str(e)}")
    
    async def get_completion(
        self, 
        request: LLMRequest
    ) -> LLMResponse:
        """Get complete response from Claude (non-streaming)."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("Claude provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by Claude")
        
        claude_model = self._get_claude_model(request.model)
        messages = self._format_messages(request)
        
        self.logger.info(
            "Starting Claude completion",
            model=claude_model,
            request_id=request.request_id
        )
        
        try:
            # Prepare kwargs for the API call
            kwargs = {
                "model": claude_model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens or 50000  # Much more aggressive fallback
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            
            response = await self.client.messages.create(**kwargs)
            
            # Extract content from response
            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text
            
            total_tokens = 0
            if response.usage:
                total_tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider_type.value,
                finish_reason="stop",
                token_count=total_tokens or self._estimate_tokens(content),
                is_complete=True,
                metadata={
                    "claude_model": claude_model,
                    "request_id": request.request_id,
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0
                }
            )
        
        except anthropic.RateLimitError as e:
            self.logger.error(f"Claude rate limit exceeded: {e}")
            raise LLMProviderRateLimitError(f"Claude rate limit: {str(e)}")
        
        except anthropic.AuthenticationError as e:
            self.logger.error(f"Claude authentication failed: {e}")
            raise LLMProviderAuthError(f"Claude auth error: {str(e)}")
        
        except anthropic.BadRequestError as e:
            self.logger.error(f"Claude bad request: {e}")
            raise LLMProviderModelError(f"Claude request error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Claude completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"Claude error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Claude-specific health check."""
        base_health = await super().health_check()
        
        if not ANTHROPIC_AVAILABLE:
            return {
                **base_health,
                "status": "unavailable",
                "error": "Anthropic package not installed"
            }
        
        if not self.settings.anthropic_api_key:
            return {
                **base_health,
                "status": "unavailable",
                "error": "Anthropic API key not configured"
            }
        
        try:
            # Test with available models (doesn't require API call)
            models = await self.get_available_models()
            return {
                **base_health,
                "available_models": models,
                "api_key_configured": bool(self.settings.anthropic_api_key)
            }
        except Exception as e:
            return {
                **base_health,
                "status": "unhealthy",
                "error": str(e)
            } 