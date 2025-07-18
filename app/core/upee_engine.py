"""UPEE Engine - Main orchestrator for the cognitive loop."""

import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List

from app.schemas import (
    ChatRequest, UPEEPhase, UPEEResult, EventType, ThinkingEvent, 
    ContentEvent, CompleteEvent, FileContext
)
from app.core.understand import UnderstandPhase
from app.core.plan import PlanPhase
from app.core.execute import ExecutePhase
from app.core.evaluate import EvaluatePhase
from app.agents.manager import AgentManager
from app.utils.logging_config import get_logger, log_upee_phase
from app.settings import Settings


class UPEEEngine:
    """
    UPEE (Understand → Plan → Execute → Evaluate) cognitive loop engine.
    
    Orchestrates the four-phase process:
    1. Understand: Parse and analyze input with context
    2. Plan: Develop response strategy and identify resources
    3. Execute: Generate response using LLM providers
    4. Evaluate: Assess quality and refine if needed
    """
    
    def __init__(self, settings: Settings, grpc_manager=None):
        self.settings = settings
        self.logger = get_logger("upee_engine")
        self.grpc_manager = grpc_manager
        
        # Initialize agent manager
        self.agent_manager = AgentManager(settings)
        
        # Initialize phases
        self.understand_phase = UnderstandPhase(settings)
        self.plan_phase = PlanPhase(settings)
        self.execute_phase = ExecutePhase(settings, grpc_manager)
        self.evaluate_phase = EvaluatePhase(settings)
        
        # Tracking
        self.current_request_id: Optional[str] = None
        self.phase_results: Dict[UPEEPhase, UPEEResult] = {}
        self.retry_count: int = 0
        self.max_retries: int = 3
        self.retry_attempts: List[Dict[str, Any]] = []
        self._agent_manager_started = False
    
    async def startup(self):
        """Start the UPEE engine and its components."""
        if not self._agent_manager_started:
            await self.agent_manager.startup()
            self._agent_manager_started = True
            self.logger.info("UPEE Engine started with agent manager")
    
    async def shutdown(self):
        """Shutdown the UPEE engine and its components."""
        if self._agent_manager_started:
            await self.agent_manager.shutdown()
            self._agent_manager_started = False
            self.logger.info("UPEE Engine shutdown complete")
    
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
                "Starting UPEE processing with retry support",
                request_id=request_id,
                message_preview=request.message[:100],
                show_thinking=request.show_thinking,
                max_retries=self.max_retries
            )
            
            # Run UPEE loop with retry mechanism
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
            yield event
        
        # Phase 2: Plan
        async for event in self._run_plan_phase(request):
            yield event
        
        # Phase 3: Execute
        async for event in self._run_execute_phase(request):
            yield event
        
        # Phase 4: Evaluate
        async for event in self._run_evaluate_phase(request):
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
                "context_window": request.context_window
            }
            
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
                # Normal local processing
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
            # Execute phase yields content events directly
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
            
            duration = time.time() - phase_start
            execute_result = self.phase_results.get(UPEEPhase.EXECUTE)
            log_upee_phase(
                "execute", 
                self.current_request_id, 
                "Phase completed successfully",
                {
                    "duration_ms": duration * 1000, 
                    "tokens_generated": execute_result.metadata.get("tokens_generated", 0),
                    "model_used": execute_result.metadata.get("model_used", "unknown")
                }
            )
            
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
            yield await self._create_thinking_event(
                UPEEPhase.EVALUATE,
                "Evaluating response quality and completeness"
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
                {
                    "duration_ms": duration * 1000, 
                    "quality_score": result.metadata.get("quality_score", 0.0),
                    "needs_refinement": result.metadata.get("needs_refinement", False)
                }
            )
            
            if request.show_thinking:
                quality_score = result.metadata.get("quality_score", 0.0)
                yield await self._create_thinking_event(
                    UPEEPhase.EVALUATE,
                    f"Evaluation complete. Quality score: {quality_score:.2f}. "
                    f"Response approved: {not result.metadata.get('needs_refinement', False)}"
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
        content: str
    ) -> Dict[str, Any]:
        """Create a thinking event for the given phase."""
        thinking_data = ThinkingEvent(
            phase=phase.value,
            content=content,
            timestamp=time.time()
        )
        
        return {
            "event": EventType.THINKING,
            "data": thinking_data.model_dump_json(),
            "id": f"{self.current_request_id}-thinking-{phase.value}"
        }
    
    async def _create_completion_event(
        self, 
        request: ChatRequest, 
        duration: float
    ) -> Dict[str, Any]:
        """Create the final completion event with retry information."""
        execute_result = self.phase_results.get(UPEEPhase.EXECUTE)
        evaluate_result = self.phase_results.get(UPEEPhase.EVALUATE)
        
        total_tokens = 0
        if execute_result:
            total_tokens = execute_result.metadata.get("tokens_generated", 0)
        
        # Include retry information in completion data
        retry_summary = {
            "total_attempts": len(self.retry_attempts),
            "final_quality_score": evaluate_result.metadata.get("quality_score", 0.0) if evaluate_result else 0.0,
            "retry_triggered": len(self.retry_attempts) > 1,
            "max_retries_reached": self.retry_count >= self.max_retries,
            "attempt_scores": [attempt["quality_score"] for attempt in self.retry_attempts]
        }
        
        complete_data = CompleteEvent(
            total_tokens=total_tokens,
            duration=duration,
            model=execute_result.metadata.get("model_used", request.model or self.settings.resolved_default_model) if execute_result else (request.model or self.settings.resolved_default_model),
            timestamp=time.time()
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