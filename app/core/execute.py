"""Execution phase - Generate response using LLM providers and external calls."""

import asyncio
import time
from typing import Dict, Any, Optional, AsyncGenerator

from app.schemas import ChatRequest, UPEEResult, UPEEPhase, EventType, ContentEvent
from app.llm_providers import LLMProviderManager, LLMRequest, LLMProviderError
from app.grpc_clients.base import ServiceUnavailableError
from app.utils.logging_config import get_logger
from app.settings import Settings
# 추가: A2A 클라이언트 임포트
from app.utils.agent_client import AgentClient


class ExecutePhase:
    """
    Execution phase of the UPEE loop.
    
    Responsible for:
    - Executing the planned response strategy
    - Making LLM API calls with streaming
    - Coordinating external gRPC calls if needed
    - Streaming content chunks to the client
    """
    
    def __init__(self, settings: Settings, grpc_manager: Optional['GRPCClientManager'] = None):
        self.settings = settings
        self.logger = get_logger("execute_phase")
        self.llm_manager = LLMProviderManager(settings)
        self.grpc_manager = grpc_manager
        # A2A 클라이언트 초기화 (Plan 단계와 동일한 설정 사용)
        self.a2a_client = (
            AgentClient(settings.a2a_server_url)
            if settings.a2a_enabled else None
        )
    
    async def process(
        self, 
        request: ChatRequest, 
        request_id: str,
        understanding_result: Optional[UPEEResult] = None,
        plan_result: Optional[UPEEResult] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process the execution phase with streaming.
        
        Args:
            request: The chat request to execute
            request_id: Request tracking ID
            understanding_result: Result from understanding phase
            plan_result: Result from planning phase
            
        Yields:
            Content events and final phase result
        """
        self.logger.info(
            "Starting execution phase",
            request_id=request_id,
            has_understanding=understanding_result is not None,
            has_plan=plan_result is not None
        )
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # Extract plan metadata
            plan_meta = plan_result.metadata if plan_result else {}
            understanding_meta = understanding_result.metadata if understanding_result else {}
            
            # Get execution parameters
            model_to_use = plan_meta.get("model_recommendation", request.model or self.settings.resolved_default_model)
            temperature = plan_meta.get("temperature", request.temperature)
            max_tokens = plan_meta.get("max_tokens", request.max_tokens)
            
            # Log execution parameters for debugging
            self.logger.info(
                "Execution parameters configured",
                request_id=request_id,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                planned_max_tokens=plan_meta.get("max_tokens"),
                request_max_tokens=request.max_tokens,
                estimated_output_tokens=plan_meta.get("estimated_output_tokens")
            )
            
            # Build the prompt
            prompt = await self._build_prompt(request, understanding_result, plan_result)
            
            # Debug: Log the exact prompt being sent to LLM
            self.logger.info(
                "LLM prompt built",
                request_id=request_id,
                prompt_length=len(prompt),
                prompt_preview=prompt[:500] + "..." if len(prompt) > 500 else prompt
            )
            
            # Execute external calls if needed
            external_results = await self._execute_external_calls(
                request, request_id, plan_meta
            )
            
            # Execute LLM call with streaming
            async for event in self._execute_llm_streaming(
                prompt, model_to_use, temperature, max_tokens, request_id, external_results
            ):
                if event["event"] == EventType.CONTENT:
                    # Track tokens (rough estimation)
                    content_data = event.get("data", {})
                    if isinstance(content_data, str):
                        import json
                        content_data = json.loads(content_data)
                    
                    chunk = content_data.get("content", "")
                    total_tokens += len(chunk.split())  # Rough token count
                    
                    yield event
                elif event["event"] == "execution_complete":
                    total_tokens = event.get("total_tokens", total_tokens)
            
            # Create final execution result
            duration = time.time() - start_time
            
            result = UPEEResult(
                phase=UPEEPhase.EXECUTE,
                content="Execution completed successfully",
                metadata={
                    "model_used": model_to_use,
                    "tokens_generated": total_tokens,
                    "duration_ms": duration * 1000,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "external_calls_made": len(external_results),
                    "external_results": external_results
                },
                completed=True
            )
            
            self.logger.info(
                "Execution phase completed",
                request_id=request_id,
                model_used=model_to_use,
                tokens_generated=total_tokens,
                duration_ms=duration * 1000
            )
            
            # Yield the final result
            yield {
                "event": "phase_complete",
                "result": result
            }
            
        except Exception as e:
            self.logger.error(
                "Execution phase failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            error_result = UPEEResult(
                phase=UPEEPhase.EXECUTE,
                content=f"Execution failed: {str(e)}",
                metadata={"error": str(e)},
                completed=False,
                error=str(e)
            )
            
            yield {
                "event": "phase_complete",
                "result": error_result
            }
    
    async def _build_prompt(
        self,
        request: ChatRequest,
        understanding_result: Optional[UPEEResult] = None,
        plan_result: Optional[UPEEResult] = None
    ) -> str:
        """Build the prompt for the LLM call."""
        
        # Get context from understanding and planning
        understanding_meta = understanding_result.metadata if understanding_result else {}
        plan_meta = plan_result.metadata if plan_result else {}
        
        intent = understanding_meta.get("intent", "general")
        strategy = plan_meta.get("strategy", "direct_response")
        approach = plan_meta.get("response_structure", "structured")
        
        # Build system prompt based on strategy
        system_prompt = self._build_system_prompt(intent, strategy, approach)
        
        # Build user prompt with file context
        user_prompt = await self._build_user_prompt(request, understanding_meta)
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        return full_prompt
    
    def _build_system_prompt(self, intent: str, strategy: str, approach: str) -> str:
        """Build the system prompt based on the execution plan."""
        
        base_prompt = "You are PAF Core Agent, a helpful AI assistant designed to provide intelligent responses through a structured UPEE (Understand → Plan → Execute → Evaluate) process."
        
        # Customize based on intent and strategy
        intent_prompts = {
            "conversation": "Engage in natural, friendly conversation while being helpful and informative.",
            "question": "Provide clear, accurate, and comprehensive answers to questions. Use examples when helpful.",
            "request": "Focus on understanding the user's needs and provide practical, actionable solutions.",
            "task": "Break down complex tasks into clear steps. Be methodical and thorough in your approach.",
            "analysis": "Provide detailed, analytical responses. Consider multiple perspectives and provide evidence-based insights."
        }
        
        approach_prompts = {
            "conversational": "Keep your tone natural and engaging.",
            "explanatory": "Focus on clarity and educational value.",
            "step_by_step": "Structure your response with clear steps and actionable guidance.",
            "analytical_with_context": "Reference the provided context and build your analysis upon it.",
            "structured": "Organize your response logically with clear sections."
        }
        
        intent_guidance = intent_prompts.get(intent, "Provide a helpful and informative response.")
        approach_guidance = approach_prompts.get(approach, "Structure your response clearly.")
        
        return f"{base_prompt}\n\n{intent_guidance} {approach_guidance}"
    
    async def _build_user_prompt(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any]
    ) -> str:
        """Build the user prompt including file context."""
        
        prompt_parts = []
        
        # Add file context if provided
        if request.files:
            prompt_parts.append("Context files:")
            for i, file_item in enumerate(request.files, 1):
                # Handle both FileContent and FileContext schemas
                if hasattr(file_item, 'file_name'):
                    # New FileContent schema
                    file_name = file_item.file_name
                    file_content = file_item.content or ""
                else:
                    # Legacy FileContext schema
                    file_name = getattr(file_item, 'path', f'file_{i}')
                    file_content = getattr(file_item, 'content', "")
                
                # Get processed content from understanding phase if available
                processed_files = understanding_meta.get("processed_files", [])
                if i <= len(processed_files):
                    processed_file = processed_files[i-1]
                    if processed_file.get("processed") and processed_file.get("content"):
                        # Use the processed content (from agentic workflow)
                        file_content = processed_file["content"]
                        file_name = processed_file.get("file_name", file_name)
                
                # Truncate if too long (simple implementation)
                if file_content and len(file_content) > 2000:
                    file_content = file_content[:2000] + "... [truncated]"
                
                prompt_parts.append(f"\n--- File {i}: {file_name} ---")
                prompt_parts.append(file_content or "[No content available]")
                prompt_parts.append("--- End of file ---\n")
        
        # Add the main user message
        prompt_parts.append(request.message)
        
        return "\n".join(prompt_parts)
    
    async def _execute_external_calls(
        self,
        request: ChatRequest,
        request_id: str,
        plan_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute external gRPC calls if needed."""
        
        external_results = {}
        
        if not plan_meta.get("needs_external_calls", False):
            return external_results
        
        call_types = plan_meta.get("external_call_types", [])
        
        for call_type in call_types:
            try:
                self.logger.info(
                    "Executing external call",
                    request_id=request_id,
                    call_type=call_type
                )

                # 1) A2A 에이전트 호출 처리
                if call_type == "a2a_agent":
                    result = await self._execute_a2a_agent(request, request_id, plan_meta)
                    external_results[call_type] = result
                # 2) gRPC 워커 작업 처리
                elif self.grpc_manager and call_type in [
                    "worker_task",
                    "code_analysis",
                    "file_processing",
                    "data_extraction",
                ]:
                    result = await self._execute_worker_task(
                        request, request_id, call_type, plan_meta
                    )
                    external_results[call_type] = result
                # 3) 기타 - 모의 응답
                else:
                    await asyncio.sleep(0.1)  # Simulate network delay
                    external_results[call_type] = {
                        "status": "success",
                        "data": f"Mock result for {call_type}",
                        "timestamp": time.time(),
                        "source": "simulation",
                    }
                
            except Exception as e:
                self.logger.error(
                    "External call failed",
                    request_id=request_id,
                    call_type=call_type,
                    error=str(e)
                )
                external_results[call_type] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        return external_results
    
    async def _execute_worker_task(
        self,
        request: ChatRequest,
        request_id: str,
        task_type: str,
        plan_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task using the worker agent gRPC client."""
        
        if not self.grpc_manager:
            raise ValueError("gRPC manager not available")
        
        try:
            # Prepare task payload based on request and task type
            payload = self._prepare_task_payload(request, task_type, plan_meta)
            
            # Generate task ID
            task_id = f"{request_id}-{task_type}-{int(time.time())}"
            
            # Execute the task
            response = await self.grpc_manager.execute_worker_task(
                task_id=task_id,
                task_type=task_type,
                payload=payload,
                context={
                    "request_id": request_id,
                    "user_message": request.message,
                    "intent": plan_meta.get("intent", "unknown")
                },
                timeout=30.0
            )
            
            return {
                "status": "success" if response.success else "error",
                "data": response.result,
                "error": response.error,
                "execution_time": response.execution_time,
                "task_id": response.task_id,
                "metadata": response.metadata,
                "timestamp": time.time(),
                "source": "worker_agent"
            }
            
        except ServiceUnavailableError as e:
            self.logger.warning(f"Worker service unavailable for {task_type}: {e}")
            return {
                "status": "error",
                "error": f"Worker service unavailable: {str(e)}",
                "timestamp": time.time(),
                "source": "worker_agent"
            }
        except Exception as e:
            self.logger.error(f"Worker task execution failed for {task_type}: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": time.time(),
                "source": "worker_agent"
            }

    async def _execute_a2a_agent(
        self,
        request: ChatRequest,
        request_id: str,
        plan_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """실제 A2A 에이전트(스킬)를 호출합니다."""

        if not self.a2a_client:
            return {
                "status": "error",
                "error": "A2A client not configured",
                "timestamp": time.time(),
                "source": "a2a_agent",
            }

        # Plan 단계에서 찾은 매칭 스킬 정보 사용
        match_info: Dict[str, Any] = plan_meta.get("a2a_agent_match", {})
        if not match_info or not match_info.get("matched"):
            return {
                "status": "error",
                "error": "No matching A2A skill found in plan metadata",
                "timestamp": time.time(),
                "source": "a2a_agent",
            }

        message_payload = {
            "type": "skill_request",
            "skill_id": match_info.get("skill_id"),
            "skill_name": match_info.get("skill_name"),
            # 사용자가 입력한 전체 메시지를 그대로 전달 (파라미터가 필요한 경우 추후 확장)
            "parameters": match_info.get("parameters", {}),
            "user_message": request.message,
        }

        try:
            result = await self.a2a_client.send_message(message_payload)

            return {
                "status": result.get("status", "success"),
                # result 내부 구조: {status:..., response:...}
                "data": result.get("response", result),
                "timestamp": time.time(),
                "source": "a2a_agent",
            }

        except Exception as e:
            self.logger.error(
                "A2A agent call failed",
                request_id=request_id,
                error=str(e),
                exc_info=True,
            )
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
                "source": "a2a_agent",
            }
    
    def _prepare_task_payload(
        self,
        request: ChatRequest,
        task_type: str,
        plan_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare payload for worker task based on task type."""
        
        base_payload = {
            "user_message": request.message,
            "complexity": plan_meta.get("complexity", "medium"),
            "intent": plan_meta.get("intent", "unknown")
        }
        
        # Add file context if available
        if request.files:
            base_payload["files"] = [
                {
                    "path": file_ctx.path,
                    "content": file_ctx.content[:1000],  # Truncate for payload
                    "summary": file_ctx.summary
                }
                for file_ctx in request.files
            ]
        
        # Task-specific payload preparation
        if task_type == "code_analysis":
            base_payload.update({
                "analysis_type": "comprehensive",
                "include_metrics": True,
                "check_security": True
            })
        
        elif task_type == "file_processing":
            base_payload.update({
                "processing_mode": "intelligent",
                "extract_structure": True,
                "generate_summary": True
            })
        
        elif task_type == "data_extraction":
            base_payload.update({
                "extraction_format": "json",
                "include_metadata": True,
                "validate_schema": True
            })
        
        return base_payload
    
    async def _execute_llm_streaming(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        request_id: str,
        external_results: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute LLM call with streaming response."""
        
        self.logger.info(
            "Starting LLM streaming",
            request_id=request_id,
            model=model,
            prompt_length=len(prompt),
            temperature=temperature
        )
        
        try:
            # Extract system prompt and user prompt
            system_prompt, user_prompt = self._extract_prompts(prompt)
            
            # Add external results to the prompt if available
            if external_results:
                external_context = self._format_external_results(external_results)
                user_prompt = f"{user_prompt}\n\nAdditional context from external services:\n{external_context}"
            
            # Create LLM request
            llm_request = LLMRequest(
                model=model,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                request_id=request_id
            )
            
            total_tokens = 0
            content_buffer = ""
            last_provider = None
            
            # Stream from LLM provider
            async for llm_response in self.llm_manager.stream_completion(llm_request):
                last_provider = llm_response.provider
                
                if llm_response.content:
                    # Stream content chunk
                    content_buffer += llm_response.content
                    
                    content_event = ContentEvent(
                        content=llm_response.content,
                        timestamp=time.time()
                    )
                    
                    yield {
                        "event": EventType.CONTENT,
                        "data": content_event.model_dump_json(),
                        "id": f"{request_id}-content-{len(content_buffer)}"
                    }
                
                if llm_response.is_complete:
                    # Explicit completion signal from LLM provider
                    total_tokens = llm_response.token_count or self._estimate_tokens(content_buffer)
                    
                    self.logger.info(
                        "LLM streaming completed (explicit completion)",
                        request_id=request_id,
                        model=model,
                        total_tokens=total_tokens,
                        content_length=len(content_buffer),
                        provider=llm_response.provider
                    )
                    
                    # Signal completion
                    yield {
                        "event": "execution_complete",
                        "total_tokens": total_tokens,
                        "provider": llm_response.provider,
                        "finish_reason": llm_response.finish_reason
                    }
                    return
            
            # If we exit the async loop without explicit completion, handle implicit completion
            total_tokens = self._estimate_tokens(content_buffer)
            
            self.logger.info(
                "LLM streaming completed (implicit completion)",
                request_id=request_id,
                model=model,
                total_tokens=total_tokens,
                content_length=len(content_buffer),
                provider=last_provider
            )
            
            # Signal completion
            yield {
                "event": "execution_complete", 
                "total_tokens": total_tokens,
                "provider": last_provider,
                "finish_reason": "stream_ended"
            }
        
        except LLMProviderError as e:
            self.logger.error(
                "LLM provider error during streaming",
                request_id=request_id,
                model=model,
                error=str(e)
            )
            
            # Fallback to error response
            error_message = f"I apologize, but I encountered an issue with the LLM service: {str(e)}. Please try again later."
            
            content_event = ContentEvent(
                content=error_message,
                timestamp=time.time()
            )
            
            yield {
                "event": EventType.CONTENT,
                "data": content_event.model_dump_json(),
                "id": f"{request_id}-error"
            }
            
            yield {
                "event": "execution_complete",
                "total_tokens": self._estimate_tokens(error_message),
                "error": str(e)
            }
        
        except Exception as e:
            self.logger.error(
                "Unexpected error during LLM streaming",
                request_id=request_id,
                model=model,
                error=str(e),
                exc_info=True
            )
            
            # Fallback to error response
            error_message = f"I apologize, but I encountered an unexpected error. Please try again later."
            
            content_event = ContentEvent(
                content=error_message,
                timestamp=time.time()
            )
            
            yield {
                "event": EventType.CONTENT,
                "data": content_event.model_dump_json(),
                "id": f"{request_id}-error"
            }
            
            yield {
                "event": "execution_complete",
                "total_tokens": self._estimate_tokens(error_message),
                "error": str(e)
            }
    
    def _extract_prompts(self, full_prompt: str) -> tuple[str, str]:
        """Extract system and user prompts from full prompt."""
        # Split on the first occurrence of "User:"
        if "User:" in full_prompt:
            parts = full_prompt.split("User:", 1)
            system_prompt = parts[0].strip()
            user_prompt = parts[1].strip()
        else:
            # If no clear separation, treat entire prompt as user prompt
            system_prompt = "You are PAF Core Agent, a helpful AI assistant."
            user_prompt = full_prompt.strip()
        
        return system_prompt, user_prompt
    
    def _format_external_results(self, external_results: Dict[str, Any]) -> str:
        """Format external results for inclusion in prompt."""
        formatted_results = []
        
        for service, result in external_results.items():
            if result.get("status") == "success":
                data = result.get("data", "No data")
                formatted_results.append(f"- {service}: {data}")
            else:
                error = result.get("error", "Unknown error")
                formatted_results.append(f"- {service}: Error - {error}")
        
        return "\n".join(formatted_results)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters per token average)."""
        return len(text) // 4
    
 