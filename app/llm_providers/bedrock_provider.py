"""AWS Bedrock LLM provider implementation."""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

try:
    import boto3
    import aioboto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from app.llm_providers.base import (
    LLMProvider, LLMProviderType, LLMRequest, LLMResponse, LLMUsage,
    LLMProviderError, LLMProviderNotAvailableError, LLMProviderRateLimitError,
    LLMProviderAuthError, LLMProviderModelError
)
from app.settings import Settings


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider implementation."""
    
    def __init__(self, settings: Settings):
        super().__init__(LLMProviderType.BEDROCK)
        self.settings = settings
        
        # Bedrock model mappings
        self.model_map = {
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "titan-text": "amazon.titan-text-express-v1",
            "llama2-13b": "meta.llama2-13b-chat-v1",
            "llama2-70b": "meta.llama2-70b-chat-v1"
        }
        
        # Set up AWS session parameters
        self.aws_config = {
            "region_name": settings.aws_region
        }
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.aws_config.update({
                "aws_access_key_id": settings.aws_access_key_id,
                "aws_secret_access_key": settings.aws_secret_access_key
            })
    
    async def is_available(self) -> bool:
        """Check if Bedrock provider is available."""
        if not BOTO3_AVAILABLE:
            return False
        
        try:
            # Test AWS credentials and region
            session = aioboto3.Session(**self.aws_config)
            async with session.client('bedrock-runtime') as client:
                # This will fail if credentials are invalid
                await client.list_foundation_models()
                return True
        except Exception:
            return False
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Bedrock models."""
        if not await self.is_available():
            return []
        
        return list(self.model_map.keys())
    
    async def validate_model(self, model: str) -> bool:
        """Validate if a model is supported."""
        return model in self.model_map
    
    def _get_bedrock_model(self, model: str) -> str:
        """Get the actual Bedrock model ID."""
        return self.model_map.get(model, model)
    
    def _format_claude_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for Claude models on Bedrock."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        }
        
        if request.system_prompt:
            body["system"] = request.system_prompt
        
        return body
    
    def _format_titan_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for Amazon Titan models."""
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\nHuman: {request.prompt}\n\nAssistant:"
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens or 1000,
                "temperature": request.temperature,
                "stopSequences": []
            }
        }
    
    def _format_llama_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Format request for Llama models."""
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"<s>[INST] <<SYS>>\n{request.system_prompt}\n<</SYS>>\n\n{request.prompt} [/INST]"
        
        return {
            "prompt": prompt,
            "max_gen_len": request.max_tokens or 1000,
            "temperature": request.temperature,
            "top_p": 0.9
        }
    
    def _format_request_body(self, request: LLMRequest, bedrock_model: str) -> Dict[str, Any]:
        """Format request body based on model type."""
        if "anthropic.claude" in bedrock_model:
            return self._format_claude_request(request)
        elif "amazon.titan" in bedrock_model:
            return self._format_titan_request(request)
        elif "meta.llama" in bedrock_model:
            return self._format_llama_request(request)
        else:
            # Default to Claude format
            return self._format_claude_request(request)
    
    async def stream_completion(
        self, 
        request: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream completion from Bedrock."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("Bedrock provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by Bedrock")
        
        bedrock_model = self._get_bedrock_model(request.model)
        body = self._format_request_body(request, bedrock_model)
        
        self.logger.info(
            "Starting Bedrock streaming completion",
            model=bedrock_model,
            request_id=request.request_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        try:
            session = aioboto3.Session(**self.aws_config)
            async with session.client('bedrock-runtime') as client:
                
                # Note: Not all Bedrock models support streaming
                # For now, we'll use invoke_model and simulate streaming
                response = await client.invoke_model(
                    modelId=bedrock_model,
                    body=json.dumps(body),
                    contentType='application/json'
                )
                
                response_body = json.loads(await response['body'].read())
                
                # Parse response based on model type
                content = ""
                if "anthropic.claude" in bedrock_model:
                    content = response_body.get('content', [{}])[0].get('text', '')
                elif "amazon.titan" in bedrock_model:
                    content = response_body.get('outputText', '')
                elif "meta.llama" in bedrock_model:
                    content = response_body.get('generation', '')
                
                # Simulate streaming by chunking the response
                words = content.split()
                chunk_size = 3  # Words per chunk
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if i + chunk_size < len(words):
                        chunk += " "
                    
                    yield LLMResponse(
                        content=chunk,
                        model=request.model,
                        provider=self.provider_type.value,
                        token_count=self._estimate_tokens(chunk),
                        is_complete=False,
                        metadata={
                            "bedrock_model": bedrock_model,
                            "request_id": request.request_id
                        }
                    )
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)
                
                # Final completion chunk
                total_tokens = self._estimate_tokens(content)
                yield LLMResponse(
                    content="",
                    model=request.model,
                    provider=self.provider_type.value,
                    finish_reason="stop",
                    token_count=total_tokens,
                    is_complete=True,
                    metadata={
                        "bedrock_model": bedrock_model,
                        "request_id": request.request_id,
                        "total_content_length": len(content)
                    }
                )
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ThrottlingException':
                self.logger.error(f"Bedrock rate limit exceeded: {e}")
                raise LLMProviderRateLimitError(f"Bedrock rate limit: {str(e)}")
            elif error_code == 'AccessDeniedException':
                self.logger.error(f"Bedrock access denied: {e}")
                raise LLMProviderAuthError(f"Bedrock auth error: {str(e)}")
            else:
                self.logger.error(f"Bedrock client error: {e}")
                raise LLMProviderError(f"Bedrock error: {str(e)}")
        
        except NoCredentialsError as e:
            self.logger.error(f"Bedrock credentials not found: {e}")
            raise LLMProviderAuthError(f"Bedrock credentials error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Bedrock completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"Bedrock error: {str(e)}")
    
    async def get_completion(
        self, 
        request: LLMRequest
    ) -> LLMResponse:
        """Get complete response from Bedrock (non-streaming)."""
        if not await self.is_available():
            raise LLMProviderNotAvailableError("Bedrock provider not available")
        
        if not await self.validate_model(request.model):
            raise LLMProviderModelError(f"Model {request.model} not supported by Bedrock")
        
        bedrock_model = self._get_bedrock_model(request.model)
        body = self._format_request_body(request, bedrock_model)
        
        self.logger.info(
            "Starting Bedrock completion",
            model=bedrock_model,
            request_id=request.request_id
        )
        
        try:
            session = aioboto3.Session(**self.aws_config)
            async with session.client('bedrock-runtime') as client:
                
                response = await client.invoke_model(
                    modelId=bedrock_model,
                    body=json.dumps(body),
                    contentType='application/json'
                )
                
                response_body = json.loads(await response['body'].read())
                
                # Parse response based on model type
                content = ""
                if "anthropic.claude" in bedrock_model:
                    content = response_body.get('content', [{}])[0].get('text', '')
                elif "amazon.titan" in bedrock_model:
                    content = response_body.get('outputText', '')
                elif "meta.llama" in bedrock_model:
                    content = response_body.get('generation', '')
                
                return LLMResponse(
                    content=content,
                    model=request.model,
                    provider=self.provider_type.value,
                    finish_reason="stop",
                    token_count=self._estimate_tokens(content),
                    is_complete=True,
                    metadata={
                        "bedrock_model": bedrock_model,
                        "request_id": request.request_id
                    }
                )
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ThrottlingException':
                self.logger.error(f"Bedrock rate limit exceeded: {e}")
                raise LLMProviderRateLimitError(f"Bedrock rate limit: {str(e)}")
            elif error_code == 'AccessDeniedException':
                self.logger.error(f"Bedrock access denied: {e}")
                raise LLMProviderAuthError(f"Bedrock auth error: {str(e)}")
            else:
                self.logger.error(f"Bedrock client error: {e}")
                raise LLMProviderError(f"Bedrock error: {str(e)}")
        
        except NoCredentialsError as e:
            self.logger.error(f"Bedrock credentials not found: {e}")
            raise LLMProviderAuthError(f"Bedrock credentials error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Bedrock completion failed: {e}", exc_info=True)
            raise LLMProviderError(f"Bedrock error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Bedrock-specific health check."""
        base_health = await super().health_check()
        
        if not BOTO3_AVAILABLE:
            return {
                **base_health,
                "status": "unavailable",
                "error": "Boto3 package not installed"
            }
        
        try:
            is_available = await self.is_available()
            models = await self.get_available_models()
            
            return {
                **base_health,
                "status": "healthy" if is_available else "unavailable",
                "available_models": models,
                "aws_region": self.settings.aws_region,
                "credentials_configured": bool(self.settings.aws_access_key_id)
            }
        except Exception as e:
            return {
                **base_health,
                "status": "unhealthy",
                "error": str(e)
            } 