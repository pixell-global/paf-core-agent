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
        
        # Initialize phases
        self.understand_phase = UnderstandPhase(settings)
        self.plan_phase = PlanPhase(settings)
        self.execute_phase = ExecutePhase(settings, grpc_manager)
        self.evaluate_phase = EvaluatePhase(settings)
        
        # Tracking
        self.current_request_id: Optional[str] = None
        self.phase_results: Dict[UPEEPhase, UPEEResult] = {}
    
    async def process_request(
        self, 
        request: ChatRequest,
        request_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a chat request through the UPEE loop.
        
        Args:
            request: The chat request to process
            request_id: Optional request ID for tracking
            
        Yields:
            Events (thinking, content, complete, error)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        self.current_request_id = request_id
        self.phase_results = {}
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting UPEE processing",
                request_id=request_id,
                message_preview=request.message[:100],
                show_thinking=request.show_thinking
            )
            
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
            
            # Generate completion event
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
            result = await self.understand_phase.process(
                request, self.current_request_id
            )
            self.phase_results[UPEEPhase.UNDERSTAND] = result
            
            duration = time.time() - phase_start
            log_upee_phase(
                "understand", 
                self.current_request_id, 
                "Phase completed successfully",
                {"duration_ms": duration * 1000, "tokens_analyzed": result.metadata.get("tokens", 0)}
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
            result = await self.plan_phase.process(
                request, 
                self.current_request_id,
                self.phase_results.get(UPEEPhase.UNDERSTAND)
            )
            self.phase_results[UPEEPhase.PLAN] = result
            
            duration = time.time() - phase_start
            log_upee_phase(
                "plan", 
                self.current_request_id, 
                "Phase completed successfully",
                {"duration_ms": duration * 1000, "plan_complexity": result.metadata.get("complexity", "simple")}
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
        """Create the final completion event."""
        execute_result = self.phase_results.get(UPEEPhase.EXECUTE)
        evaluate_result = self.phase_results.get(UPEEPhase.EVALUATE)
        
        total_tokens = 0
        if execute_result:
            total_tokens = execute_result.metadata.get("tokens_generated", 0)
        
        complete_data = CompleteEvent(
            total_tokens=total_tokens,
            duration=duration,
                            model=execute_result.metadata.get("model_used", request.model or self.settings.resolved_default_model) if execute_result else (request.model or self.settings.resolved_default_model),
            timestamp=time.time()
        )
        
        return {
            "event": EventType.COMPLETE,
            "data": complete_data.model_dump_json(),
            "id": f"{self.current_request_id}-complete"
        } 