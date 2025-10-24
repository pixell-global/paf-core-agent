"""UPEE Engine - Main orchestrator for the cognitive loop."""

import asyncio
import time
import uuid
import json
from typing import AsyncGenerator, Dict, Any, Optional, List

from src.schemas import (
    ChatRequest, UPEEPhase, UPEEResult, EventType, ThinkingEvent, 
    ContentEvent, CompleteEvent, FileContext
)
from src.core.understand import UnderstandPhase
from src.core.plan import PlanPhase
from src.core.execute import ExecutePhase
from src.core.evaluate import EvaluatePhase
from src.agents.manager import AgentManager
from src.utils.logging_config import get_logger, log_upee_phase
from src.settings import Settings
from src.utils.agent_client import AgentClient
from src.utils.a2a_util import run
from src.langgraph_upee import execute_upee_graph, UPEEInput
from src.llm_providers import LLMProviderManager

class UPEEEngine:
    """
    UPEE (Understand → Plan → Execute → Evaluate) cognitive loop engine.
    
    Orchestrates the four-phase process:
    1. Understand: Parse and analyze input with context
    2. Plan: Develop response strategy and identify resources
    3. Execute: Generate response using LLM providers
    4. Evaluate: Assess quality and refine if needed
    """
    
    def __init__(
        self,
        settings: Settings,
        grpc_manager=None,
        a2a_client=None,
        registry=None,
        selector=None,
        client_pool=None
    ):
        """
        Initialize UPEE Engine.

        Args:
            settings: Application settings
            grpc_manager: Optional gRPC manager for worker agents
            a2a_client: Optional legacy A2A client
            registry: Optional agent registry (for multi-agent mode)
            selector: Optional agent selector (for multi-agent mode)
            client_pool: Optional client pool (for multi-agent mode)
        """
        self.settings = settings
        self.logger = get_logger("upee_engine")
        self.grpc_manager = grpc_manager

        # Multi-agent components (passed from par_adapter or main.py)
        self.registry = registry
        self.selector = selector
        self.client_pool = client_pool

        # Determine mode: multi-agent (with components) or legacy (with agent_manager)
        self._use_multi_agent_mode = bool(registry and selector)

        # Initialize agent manager only if NOT using multi-agent components
        if not self._use_multi_agent_mode:
            self.logger.info("Using legacy agent manager mode")
            self.agent_manager = AgentManager(settings)
            self._agent_manager_started = False
        else:
            self.logger.info(
                "Using multi-agent mode",
                has_registry=bool(registry),
                has_selector=bool(selector),
                has_client_pool=bool(client_pool)
            )
            self.agent_manager = None
            self._agent_manager_started = False

        # Initialize phases with multi-agent components
        self.understand_phase = UnderstandPhase(settings)
        self.plan_phase = PlanPhase(settings, registry=registry, selector=selector)
        self.execute_phase = ExecutePhase(settings, grpc_manager, client_pool=client_pool)
        self.evaluate_phase = EvaluatePhase(settings)

        # Tracking
        self.current_request_id: Optional[str] = None
        self.phase_results: Dict[UPEEPhase, UPEEResult] = {}
        self.retry_count: int = 0
        self.max_retries: int = 3
        self.retry_attempts: List[Dict[str, Any]] = []

        #Legacy A2A Client
        self.a2a_client = a2a_client or AgentClient(settings.a2a_server_url)

        # LangGraph UPEE (new AI-native routing)
        self._use_langgraph = settings.use_langgraph_upee if hasattr(settings, 'use_langgraph_upee') else False
        if self._use_langgraph:
            self.llm_manager = LLMProviderManager(settings)
            self.logger.info("UPEE Engine configured to use LangGraph AI-native routing")

    async def startup(self):
        """Start the UPEE engine and its components."""
        # Only start agent_manager if using legacy mode (not multi-agent mode)
        if self.agent_manager and not self._agent_manager_started:
            await self.agent_manager.startup()
            self._agent_manager_started = True
            self.logger.info("UPEE Engine started with legacy agent manager")
        elif self._use_multi_agent_mode:
            self.logger.info("UPEE Engine using multi-agent components (no startup needed)")

    async def shutdown(self):
        """Shutdown the UPEE engine and its components."""
        # Only shutdown agent_manager if using legacy mode
        if self.agent_manager and self._agent_manager_started:
            await self.agent_manager.shutdown()
            self._agent_manager_started = False
            self.logger.info("UPEE Engine shutdown complete")
        elif self._use_multi_agent_mode:
            self.logger.info("UPEE Engine using multi-agent components (no shutdown needed)")
    
    async def process_request(
        self, 
        request: ChatRequest,
        request_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a chat request through the UPEE loop with retry mechanism.
        
        Args:
            request: The chat request to process
            request_id: Optional request ID for tracking
            
        Yields:
            Events (thinking, content, complete, error)
        """
        # Ensure agent manager is started
        if not self._agent_manager_started:
            await self.startup()
            
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        self.current_request_id = request_id
        self.retry_count = 0
        self.retry_attempts = []
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting UPEE processing",
                request_id=request_id,
                message_preview=request.message[:100],
                show_thinking=request.show_thinking,
                use_langgraph=self._use_langgraph
            )

            # Route to LangGraph or legacy UPEE loop
            if self._use_langgraph:
                async for event in self._process_with_langgraph(request, request_id):
                    yield event
            else:
                # Run legacy UPEE loop with retry mechanism
                async for event in self._run_upee_loop_with_retries(request):
                    yield event
            
            # Generate final completion event
            duration = time.time() - start_time
            completion_event = await self._create_completion_event(
                request, duration
            )
            yield completion_event
            
        except Exception as e:
            self.logger.error(
                "UPEE processing failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            error_event = {
                "event": EventType.ERROR,
                "data": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time(),
                    "request_id": request_id
                }
            }
            yield error_event

    async def _process_with_langgraph(
        self,
        request: ChatRequest,
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process request using LangGraph AI-native routing.

        This is the new AI-powered UPEE loop that uses LangGraph for intelligent routing.
        """
        try:
            # Show thinking event if requested
            if request.show_thinking:
                yield {
                    "event": EventType.THINKING,
                    "data": {
                        "phase": "upee",
                        "thinking": "🧠 Starting AI-native UPEE processing with LangGraph...",
                        "timestamp": time.time()
                    }
                }

            # Convert ChatRequest to UPEEInput
            upee_input: UPEEInput = {
                "user_message": request.message,
                "request_id": request_id,
                "files": [f.dict() for f in request.files] if request.files else None,
                "conversation_history": [h.dict() for h in request.history] if request.history else None,
                "model": request.model,
                "show_thinking": request.show_thinking,
                "temperature": request.temperature
            }

            # Execute LangGraph
            output = await execute_upee_graph(upee_input, self.settings, self.llm_manager)

            # Show routing decision if requested
            if request.show_thinking and output.get("routing_decision"):
                routing_msg = f"🎯 AI Routing Decision: {output['routing_decision']}"
                if output.get("selected_agent_name"):
                    routing_msg += f" → {output['selected_agent_name']}"
                routing_msg += f" (confidence: {output.get('quality_score', 0):.2f})"

                yield {
                    "event": EventType.THINKING,
                    "data": {
                        "phase": "routing",
                        "thinking": routing_msg,
                        "timestamp": time.time()
                    }
                }

            # Yield content event with response
            if output.get("response"):
                yield {
                    "event": EventType.CONTENT,
                    "data": {
                        "content": output["response"],
                        "timestamp": time.time(),
                        "metadata": {
                            "routing": output.get("routing_decision"),
                            "agent": output.get("selected_agent_name"),
                            "quality": output.get("quality_score")
                        }
                    }
                }

            # Store result for completion event
            self.phase_results[UPEEPhase.EVALUATE] = UPEEResult(
                phase=UPEEPhase.EVALUATE,
                content=output.get("response", ""),
                metadata={
                    "quality_score": output.get("quality_score", 0.0),
                    "routing_decision": output.get("routing_decision"),
                    "selected_agent": output.get("selected_agent_name"),
                    "error": output.get("error")
                },
                completed=True
            )

        except Exception as e:
            self.logger.error(
                "LangGraph processing failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )

            # Return error event
            yield {
                "event": EventType.ERROR,
                "data": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time(),
                    "request_id": request_id
                }
            }

    async def _run_upee_loop_with_retries(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the UPEE loop with retry mechanism based on evaluation quality score.
        
        Args:
            request: The chat request to process
            
        Yields:
            Events (thinking, content, complete, error)
        """
        best_result_quality = 0.0
        best_phase_results = None
        
        while self.retry_count <= self.max_retries:
            attempt_start = time.time()
            self.phase_results = {}
            
            # Show retry thinking event
            if self.retry_count > 0 and request.show_thinking:
                yield await self._create_thinking_event(
                    UPEEPhase.EVALUATE,
                    f"Starting retry attempt {self.retry_count}/{self.max_retries}. "
                    f"Previous attempt quality: {self.retry_attempts[-1]['quality_score']:.3f}"
                )
            
            try:
                # Run single UPEE cycle
                content_generated = False
                stored_content_events = []
                async for event in self._run_single_upee_cycle(request):
                    if event.get("event") == EventType.CONTENT:
                        # Store content events for potential retry, but also yield them for streaming
                        stored_content_events.append(event)
                        content_generated = True
                        # For simple conversations, always yield content immediately for better UX
                        yield event
                    else:
                        yield event
                
                # Check if this attempt meets quality threshold
                evaluate_result = self.phase_results.get(UPEEPhase.EVALUATE)
                if evaluate_result:
                    quality_score = evaluate_result.metadata.get("quality_score", 0.0)
                    needs_refinement = evaluate_result.metadata.get("needs_refinement", False)
                    
                    # Store this attempt
                    attempt_data = {
                        "attempt": self.retry_count + 1,
                        "quality_score": quality_score,
                        "needs_refinement": needs_refinement,
                        "duration": time.time() - attempt_start,
                        "phase_results": self.phase_results.copy()
                    }
                    self.retry_attempts.append(attempt_data)
                    
                    # Track best result
                    if quality_score > best_result_quality:
                        best_result_quality = quality_score
                        best_phase_results = self.phase_results.copy()
                    
                    # Log attempt result
                    self.logger.info(
                        "UPEE attempt completed",
                        request_id=self.current_request_id,
                        attempt=self.retry_count + 1,
                        quality_score=quality_score,
                        needs_refinement=needs_refinement,
                        will_retry=needs_refinement and self.retry_count < self.max_retries
                    )
                    
                    # Check if we should retry
                    if not needs_refinement or self.retry_count >= self.max_retries:
                        # Use current results if quality is acceptable, or best results if max retries reached
                        if needs_refinement and self.retry_count >= self.max_retries:
                            self.phase_results = best_phase_results
                            if request.show_thinking:
                                yield await self._create_thinking_event(
                                    UPEEPhase.EVALUATE,
                                    f"Max retries reached. Using best attempt with quality: {best_result_quality:.3f}"
                                )
                        
                        # Content already yielded during execution phase for better streaming UX
                        
                        return
                    
                    # Prepare for retry
                    self.retry_count += 1
                    if request.show_thinking:
                        yield await self._create_thinking_event(
                            UPEEPhase.EVALUATE,
                            f"Quality score {quality_score:.3f} below threshold (0.5). "
                            f"Preparing retry {self.retry_count}/{self.max_retries}"
                        )
                else:
                    # No evaluation result - this shouldn't happen but handle gracefully
                    self.logger.warning(
                        "No evaluation result available",
                        request_id=self.current_request_id,
                        attempt=self.retry_count + 1
                    )
                    return
                    
            except Exception as e:
                self.logger.error(
                    "UPEE cycle failed",
                    request_id=self.current_request_id,
                    attempt=self.retry_count + 1,
                    error=str(e)
                )
                
                # If this is the last attempt, re-raise the exception
                if self.retry_count >= self.max_retries:
                    raise
                
                # Otherwise, try again
                self.retry_count += 1
                if request.show_thinking:
                    yield await self._create_thinking_event(
                        UPEEPhase.EVALUATE,
                        f"Attempt {self.retry_count} failed with error: {str(e)}. Retrying..."
                    )

    async def _run_single_upee_cycle(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run a single UPEE cycle (Understand → Plan → Execute → Evaluate).
        
        Args:
            request: The chat request to process
            
        Yields:
            Events (thinking, content, complete, error)
        """
        # Phase 1: Understand
        async for event in self._run_understand_phase(request):
            self.logger.debug(f"Understand phase event: {event}")
            yield event
        
        # Phase 2: Plan
        async for event in self._run_plan_phase(request):
            self.logger.debug(f"Plan phase event: {event}")
            yield event
        
        # Phase 3: Execute
        async for event in self._run_execute_phase(request):
            self.logger.debug(f"Execute phase event: {event}")
            yield event
        
        # Phase 4: Evaluate
        async for event in self._run_evaluate_phase(request):
            self.logger.debug(f"Evaluate phase event: {event}")
            yield event
    
    async def _run_understand_phase(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the Understanding phase."""
        phase_start = time.time()
        
        if request.show_thinking:
            yield await self._create_thinking_event(
                UPEEPhase.UNDERSTAND,
                f"Analyzing user message and context. Message length: {len(request.message)} chars. "
                f"Files provided: {len(request.files) if request.files else 0}"
            )
        
        try:
            # Check if we should use an external agent
            context = {
                "message": request.message,
                "files": [f.model_dump() for f in request.files] if request.files else [],
                "model": request.model,
                "context_window": request.context_window_size
            }
            # Pass through client UI capabilities to agent apps (if provided)
            if request.metadata and isinstance(request.metadata, dict):
                ui_caps = request.metadata.get("ui.capabilities")
                if ui_caps is not None:
                    context["ui.capabilities"] = ui_caps
            
            agent_decision = await self.agent_manager.should_use_agent(
                UPEEPhase.UNDERSTAND, context
            )
            
            if agent_decision:
                if request.show_thinking:
                    yield await self._create_thinking_event(
                        UPEEPhase.UNDERSTAND,
                        f"Using external agent '{agent_decision.agent_id}' for {agent_decision.capability}. "
                        f"Reason: {agent_decision.reasoning}"
                    )
                
                # Execute agent request
                agent_response = await self.agent_manager.execute_agent_request(
                    agent_decision,
                    payload={"request": request.model_dump()},
                    context=context
                )
                
                # Forward UI envelopes (ui.render / ui.patch) from agent to client
                try:
                    if agent_response and agent_response.status == "success":
                        ui_envelopes = []
                        # metadata["ui"] form (treated as ui.render)
                        if agent_response.metadata and isinstance(agent_response.metadata, dict):
                            ui_spec = agent_response.metadata.get("ui")
                            if isinstance(ui_spec, dict) and ui_spec.get("manifest") and ui_spec.get("view"):
                                ui_envelopes.append({"type": "ui.render", **ui_spec})
                        # envelope form in result
                        if agent_response.result and isinstance(agent_response.result, dict):
                            result_type = agent_response.result.get("type")
                            if result_type in ("ui.render", "ui.patch"):
                                ui_envelopes.append(agent_response.result)
                        # Emit each envelope as a content event for the client to render
                        for env in ui_envelopes:
                            yield {
                                "event": EventType.CONTENT,
                                "data": json.dumps({
                                    "ui": env,
                                    "agent_id": agent_decision.agent_id,
                                    "a2a_request_id": agent_response.request_id
                                })
                            }
                        # Forward action.result if present
                        if agent_response.result and isinstance(agent_response.result, dict):
                            if agent_response.result.get("type") == "action.result":
                                yield {
                                    "event": EventType.CONTENT,
                                    "data": json.dumps({
                                        "action_result": agent_response.result,
                                        "agent_id": agent_decision.agent_id,
                                        "a2a_request_id": agent_response.request_id
                                    })
                                }
                except Exception as e:
                    self.logger.warning(
                        "Failed to forward UI envelope",
                        request_id=self.current_request_id,
                        error=str(e)
                    )

                if agent_response.status == "success":
                    # Merge agent results with local processing
                    result = await self.understand_phase.process(
                        request, self.current_request_id, 
                        agent_enhancement=agent_response.result
                    )
                else:
                    # Fallback to local processing
                    self.logger.warning(
                        f"Agent request failed: {agent_response.error}",
                        agent_id=agent_decision.agent_id
                    )
                    result = await self.understand_phase.process(
                        request, self.current_request_id
                    )
            else:
                # Normal local processing
                result = await self.understand_phase.process(
                    request, self.current_request_id
                )
            
            self.phase_results[UPEEPhase.UNDERSTAND] = result
            
            duration = time.time() - phase_start
            log_upee_phase(
                "understand", 
                self.current_request_id, 
                "Phase completed successfully",
                {
                    "duration_ms": duration * 1000, 
                    "tokens_analyzed": result.metadata.get("tokens", 0),
                    "used_agent": agent_decision is not None
                }
            )
            
            if request.show_thinking:
                yield await self._create_thinking_event(
                    UPEEPhase.UNDERSTAND,
                    f"Understanding complete. Identified intent: {result.metadata.get('intent', 'general')}. "
                    f"Context tokens: {result.metadata.get('context_tokens', 0)}"
                )
                
        except Exception as e:
            self.logger.error(
                "Understand phase failed",
                request_id=self.current_request_id,
                error=str(e)
            )
            raise
    
    async def _run_plan_phase(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the Planning phase."""
        phase_start = time.time()
        
        if request.show_thinking:
            understanding_result = self.phase_results.get(UPEEPhase.UNDERSTAND)
            yield await self._create_thinking_event(
                UPEEPhase.PLAN,
                f"Planning response strategy. Intent: {understanding_result.metadata.get('intent', 'unknown')}. "
                f"Considering model: {request.model or self.settings.resolved_default_model}"
            )
        
        try:
            # Check if we should use an external agent
            understanding_result = self.phase_results.get(UPEEPhase.UNDERSTAND)
            context = {
                "understanding_result": understanding_result.model_dump() if understanding_result else {},
                "message": request.message,
                "model": request.model
            }
            # Pass through client UI capabilities to agent apps (if provided)
            if request.metadata and isinstance(request.metadata, dict):
                ui_caps = request.metadata.get("ui.capabilities")
                if ui_caps is not None:
                    context["ui.capabilities"] = ui_caps
            
            agent_decision = await self.agent_manager.should_use_agent(
                UPEEPhase.PLAN, context
            )
            
            if agent_decision:
                if request.show_thinking:
                    yield await self._create_thinking_event(
                        UPEEPhase.PLAN,
                        f"Using external agent '{agent_decision.agent_id}' for {agent_decision.capability}. "
                        f"Reason: {agent_decision.reasoning}"
                    )
                
                # Execute agent request
                agent_response = await self.agent_manager.execute_agent_request(
                    agent_decision,
                    payload={
                        "request": request.model_dump(),
                        "understanding_result": understanding_result.model_dump() if understanding_result else {}
                    },
                    context=context
                )
                
                # Forward UI envelopes (ui.render / ui.patch) from agent to client
                try:
                    if agent_response and agent_response.status == "success":
                        ui_envelopes = []
                        # metadata["ui"] form (treated as ui.render)
                        if agent_response.metadata and isinstance(agent_response.metadata, dict):
                            ui_spec = agent_response.metadata.get("ui")
                            if isinstance(ui_spec, dict) and ui_spec.get("manifest") and ui_spec.get("view"):
                                ui_envelopes.append({"type": "ui.render", **ui_spec})
                        # envelope form in result
                        if agent_response.result and isinstance(agent_response.result, dict):
                            result_type = agent_response.result.get("type")
                            if result_type in ("ui.render", "ui.patch"):
                                ui_envelopes.append(agent_response.result)
                        # Emit each envelope as a content event for the client to render
                        for env in ui_envelopes:
                            yield {
                                "event": EventType.CONTENT,
                                "data": json.dumps({
                                    "ui": env,
                                    "agent_id": agent_decision.agent_id,
                                    "a2a_request_id": agent_response.request_id
                                })
                            }
                        # Forward action.result if present
                        if agent_response.result and isinstance(agent_response.result, dict):
                            if agent_response.result.get("type") == "action.result":
                                yield {
                                    "event": EventType.CONTENT,
                                    "data": json.dumps({
                                        "action_result": agent_response.result,
                                        "agent_id": agent_decision.agent_id,
                                        "a2a_request_id": agent_response.request_id
                                    })
                                }
                except Exception as e:
                    self.logger.warning(
                        "Failed to forward UI envelope",
                        request_id=self.current_request_id,
                        error=str(e)
                    )

                if agent_response.status == "success":
                    # Merge agent results with local processing
                    result = await self.plan_phase.process(
                        request,
                        self.current_request_id,
                        understanding_result,
                        agent_enhancement=agent_response.result
                    )
                else:
                    # Fallback to local processing
                    self.logger.warning(
                        f"Agent request failed: {agent_response.error}",
                        agent_id=agent_decision.agent_id
                    )
                    result = await self.plan_phase.process(
                        request, 
                        self.current_request_id,
                        understanding_result
                    )
            else:
                # Normal local processing when no external agent is used
                result = await self.plan_phase.process(
                    request,
                    self.current_request_id,
                    understanding_result
                )
            
            self.phase_results[UPEEPhase.PLAN] = result
            
            duration = time.time() - phase_start
            log_upee_phase(
                "plan", 
                self.current_request_id, 
                "Phase completed successfully",
                {
                    "duration_ms": duration * 1000, 
                    "plan_complexity": result.metadata.get("complexity", "simple"),
                    "used_agent": agent_decision is not None
                }
            )
            
            if request.show_thinking:
                yield await self._create_thinking_event(
                    UPEEPhase.PLAN,
                    f"Plan created. Strategy: {result.metadata.get('strategy', 'direct_response')}. "
                    f"External calls needed: {result.metadata.get('needs_external_calls', False)}"
                )
                
        except Exception as e:
            self.logger.error(
                "Plan phase failed",
                request_id=self.current_request_id,
                error=str(e)
            )
            raise
    
    async def _run_execute_phase(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the Execution phase."""
        phase_start = time.time()
        
        if request.show_thinking:
            plan_result = self.phase_results.get(UPEEPhase.PLAN)
            yield await self._create_thinking_event(
                UPEEPhase.EXECUTE,
                f"Executing plan using {request.model or self.settings.resolved_default_model}. "
                f"Strategy: {plan_result.metadata.get('strategy', 'direct_response')}"
            )
        
        try:
            plan_result = self.phase_results.get(UPEEPhase.PLAN)
            # 실행 단계에서는 ExecutePhase 내부에서 외부 호출이 처리되므로
            # 여기서 A2A 호출을 다시 수행하지 않는다 (중복 방지).
            async for event in self.execute_phase.process(
                request,
                self.current_request_id,
                self.phase_results.get(UPEEPhase.UNDERSTAND),
                self.phase_results.get(UPEEPhase.PLAN)
            ):
                # Pass through content events
                if event.get("event") == EventType.CONTENT:
                    yield event
                # Store the final result
                elif event.get("event") == "phase_complete":
                    self.phase_results[UPEEPhase.EXECUTE] = event["result"]
                    
        except Exception as e:
            self.logger.error(
                "Execute phase failed",
                request_id=self.current_request_id,
                error=str(e)
            )
            raise
    
    async def _run_evaluate_phase(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the Evaluation phase."""
        phase_start = time.time()
        
        if request.show_thinking:
            execute_result = self.phase_results.get(UPEEPhase.EXECUTE)
            yield await self._create_thinking_event(
                UPEEPhase.EVALUATE,
                f"Evaluating response quality. Content length: {len(execute_result.content) if execute_result else 0} chars"
            )
        
        try:
            result = await self.evaluate_phase.process(
                request,
                self.current_request_id,
                self.phase_results.get(UPEEPhase.UNDERSTAND),
                self.phase_results.get(UPEEPhase.PLAN),
                self.phase_results.get(UPEEPhase.EXECUTE)
            )
            self.phase_results[UPEEPhase.EVALUATE] = result
            
            duration = time.time() - phase_start
            log_upee_phase(
                "evaluate", 
                self.current_request_id, 
                "Phase completed successfully",
                {"duration_ms": duration * 1000, "quality_score": result.metadata.get("quality_score", 0.0)}
            )
            
            if request.show_thinking:
                yield await self._create_thinking_event(
                    UPEEPhase.EVALUATE,
                    f"Evaluation complete. Quality score: {result.metadata.get('quality_score', 0.0):.2f}"
                )
                
        except Exception as e:
            self.logger.error(
                "Evaluate phase failed",
                request_id=self.current_request_id,
                error=str(e)
            )
            raise
    
    async def _create_thinking_event(
        self, 
        phase: UPEEPhase, 
        message: str
    ) -> Dict[str, Any]:
        """Create a thinking event for streaming."""
        thinking_event = ThinkingEvent(
            phase=phase.value,  # phase.value로 문자열 변환
            content=message,    # message -> content 필드명 수정
            timestamp=time.time()
        )
        
        return {
            "event": EventType.THINKING,
            "data": thinking_event.model_dump_json(),
            "id": f"{self.current_request_id}-thinking-{phase.value}"
        }
    
    async def _create_completion_event(
        self,
        request: ChatRequest,
        duration: float
    ) -> Dict[str, Any]:
        """Create the final completion event with retry information and A2A agent attribution."""
        execute_result = self.phase_results.get(UPEEPhase.EXECUTE)
        evaluate_result = self.phase_results.get(UPEEPhase.EVALUATE)
        plan_result = self.phase_results.get(UPEEPhase.PLAN)

        # Build retry summary
        retry_summary = {
            "total_attempts": len(self.retry_attempts) + 1,
            "max_retries_reached": self.retry_count >= self.max_retries,
            "final_quality_score": evaluate_result.metadata.get("quality_score", 0.0) if evaluate_result else 0.0,
            "attempt_scores": [attempt.get("quality_score", 0.0) for attempt in self.retry_attempts]
        }

        # Include retry information in completion data
        retry_summary = {
            "total_attempts": len(self.retry_attempts),
            "final_quality_score": evaluate_result.metadata.get("quality_score", 0.0) if evaluate_result else 0.0,
            "retry_triggered": len(self.retry_attempts) > 1,
            "max_retries_reached": self.retry_count >= self.max_retries,
            "attempt_scores": [attempt["quality_score"] for attempt in self.retry_attempts]
        }

        # Extract A2A agent attribution info
        agent_used = "PAF Core Agent"  # Default to core agent
        agent_app_id = None
        skill_used = None
        skill_id = None
        routing_source = "core_agent"

        # Check if A2A agent was used
        if execute_result:
            external_results = execute_result.metadata.get("external_results", {})
            a2a_result = external_results.get("a2a_agent", {})

            if a2a_result and a2a_result.get("status") == "success":
                # A2A agent was used successfully
                agent_app_id = a2a_result.get("agent_app_id")
                agent_used = a2a_result.get("agent_name", "External Agent")
                routing_source = a2a_result.get("source", "a2a_agent")

                # Get skill info from plan metadata
                if plan_result:
                    a2a_match = plan_result.metadata.get("a2a_agent_match", {})
                    if a2a_match.get("matched"):
                        skill_used = a2a_match.get("skill_name")
                        skill_id = a2a_match.get("skill_id")

        complete_data = CompleteEvent(
            total_tokens=execute_result.metadata.get("tokens_used", 0) if execute_result else 0,
            duration=duration,
            model=execute_result.metadata.get("model_used", request.model or self.settings.resolved_default_model) if execute_result else (request.model or self.settings.resolved_default_model),
            timestamp=time.time(),
            agent_used=agent_used,
            agent_app_id=agent_app_id,
            skill_used=skill_used,
            skill_id=skill_id,
            routing_source=routing_source
        )
        
        # Add retry information to the completion event
        completion_event = {
            "event": EventType.COMPLETE,
            "data": complete_data.model_dump_json(),
            "id": f"{self.current_request_id}-complete",
            "retry_summary": retry_summary
        }
        
        # Log retry summary
        if len(self.retry_attempts) > 1:
            self.logger.info(
                "UPEE completed with retries",
                request_id=self.current_request_id,
                total_attempts=retry_summary["total_attempts"],
                final_quality_score=retry_summary["final_quality_score"],
                max_retries_reached=retry_summary["max_retries_reached"],
                attempt_scores=retry_summary["attempt_scores"]
            )
        
        return completion_event
    
    async def call_a2a_agent(self, request: ChatRequest, a2a_agent_match: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 에이전트를 스킬 정보와 함께 호출합니다."""
        
        if not self.settings.a2a_enabled:
            self.logger.info("A2A functionality is disabled")
            return {"status": "disabled", "message": "A2A functionality is disabled"}
        
        skill_data = a2a_agent_match.get("skill_data", {})
        skill_name = a2a_agent_match.get("skill_name", "")
        skill_id = a2a_agent_match.get("skill_id", "")
        
        # 스킬 정보에서 파라미터 추출
        skill_description = skill_data.get("description", "")
        
        # 사용자 메시지에서 파라미터 추출 (간단한 예시)
        parameters = self._extract_parameters_from_message(request.message, skill_description)
        
        # 메시지를 보낼 대상 에이전트 URL 결정 (skill 이 정의된 카드 URL 우선)
        agent_url = a2a_agent_match.get("agent_url")

        # 에이전트 URL 이 settings 와 다르면 임시 클라이언트 이용
        target_client = self.a2a_client
        if agent_url and agent_url.strip() and agent_url.rstrip('/') != self.settings.a2a_server_url.rstrip('/'):
            from src.utils.agent_client import AgentClient
            target_client = AgentClient(agent_url)

        # A2A 메시지 구성
        message = {
            "type": "skill_request",
            "skill_id": skill_id,
            "skill_name": skill_name,
            "parameters": parameters,
            "user_message": request.message,
            "payload": {
                "message": request.message,
                "skill_request": {
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "parameters": parameters
                }
            }
        }
        
        if agent_url:
            message["url"] = agent_url
        elif self.settings.a2a_server_url:
            message["url"] = self.settings.a2a_server_url
        
        try:
            response = await run(self, parameters.get("user_id"),
                                skill_id,
                                parameters.get("name"),
                                agent_url)
            
            return response
            
        except Exception as e:
            self.logger.error(f"A2A agent call failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _extract_parameters_from_message(self, message: str, skill_description: str) -> Dict[str, Any]:
        """사용자 메시지에서 스킬 파라미터를 추출합니다."""
        import re
        
        parameters = {}
        message_lower = message.lower()
        
        # 스킬 설명에서 파라미터 정보 추출
        # 예: "create new productparameters: user_id[str]-user id, name[str]-product name, description[str]-product description"
        if "parameters:" in skill_description:
            param_section = skill_description.split("parameters:")[1].strip()
            param_matches = re.findall(r'(\w+)\[(\w+)\]', param_section)
            
            for param_name, param_type in param_matches:
                # 메시지에서 파라미터 값 추출
                if param_name == "user_id":
                    # "userid: 1" 패턴 찾기
                    user_id_match = re.search(r'userid?\s*:\s*(\d+)', message_lower)
                    if user_id_match:
                        parameters["user_id"] = user_id_match.group(1)
                
                elif param_name == "name":
                    # "제품명:test22" 패턴 찾기
                    name_match = re.search(r'제품명?\s*:\s*([^,\s]+)', message_lower)
                    if name_match:
                        parameters["name"] = name_match.group(1)
                
                elif param_name == "description":
                    # "제품셜명: 굿!!" 패턴 찾기
                    desc_match = re.search(r'제품?설명?\s*:\s*([^,]+)', message_lower)
                    if desc_match:
                        parameters["description"] = desc_match.group(1).strip()
        
        # 기본값 설정
        if not parameters.get("user_id"):
            # userid가 없으면 기본값 설정
            user_id_match = re.search(r'userid?\s*:\s*(\d+)', message_lower)
            if user_id_match:
                parameters["user_id"] = user_id_match.group(1)
        
        self.logger.info(
            "Extracted parameters from message",
            parameters=parameters,
            message_preview=message[:100]
        )
        
        return parameters
    
    def call_external_agent(self, message_type: str, payload: dict) -> dict:
        """외부 A2A 서버에 메시지를 보내고 응답을 반환합니다."""
        
        # A2A 기능이 비활성화된 경우
        if not self.settings.a2a_enabled:
            self.logger.info("A2A functionality is disabled")
            return {"status": "disabled", "message": "A2A functionality is disabled"}
        
        # A2A 메시지 구성
        message = {
            "type": message_type,
            "payload": payload
        }
        
        if self.settings.a2a_server_url:
            message["url"] = self.settings.a2a_server_url
        else:
            # 기본값 설정
            message["agent_card"] = "paf-core-agent"
            self.logger.warning("No agent_card or url configured, using default: paf-core-agent")
        
        try:
            response = self.a2a_client.send_message(message)
            return response
        except Exception as e:
            self.logger.error(f"A2A call failed: {e}")
            return {"status": "error", "error": str(e)}
