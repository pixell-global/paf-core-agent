"""Planning phase - Develop response strategy and identify required resources."""

import time
from typing import Dict, Any, Optional

from app.schemas import ChatRequest, UPEEResult, UPEEPhase
from app.settings import Settings
from app.utils.logging_config import get_logger


class PlanPhase:
    """
    Planning phase of the UPEE loop.
    
    Responsible for:
    - Developing response strategy based on understanding
    - Identifying required resources (LLM, external calls)
    - Planning response structure and approach
    - Determining execution parameters
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("plan_phase")
    
    async def process(
        self, 
        request: ChatRequest, 
        request_id: str,
        understanding_result: Optional[UPEEResult] = None,
        agent_enhancement: Optional[Dict[str, Any]] = None
    ) -> UPEEResult:
        """
        Process the planning phase.
        
        Args:
            request: The chat request to plan for
            request_id: Request tracking ID
            understanding_result: Result from understanding phase
            
        Returns:
            UPEEResult with planning strategy
        """
        self.logger.info(
            "Starting planning phase",
            request_id=request_id,
            has_understanding=understanding_result is not None
        )
        
        try:
            # Extract understanding metadata
            understanding_meta = understanding_result.metadata if understanding_result else {}
            
            # Determine response strategy
            strategy = await self._determine_strategy(request, understanding_meta)
            
            # Plan model selection
            model_plan = await self._plan_model_selection(request, understanding_meta, strategy)
            
            # Plan external calls
            external_calls_plan = await self._plan_external_calls(request, understanding_meta, strategy)
            
            # Plan file processing approach
            file_processing_plan = await self._plan_file_processing(request, understanding_meta, strategy)
            
            # Plan memory context usage
            memory_plan = await self._plan_memory_usage(request, understanding_meta, strategy)
            
            # Plan response structure
            structure_plan = await self._plan_response_structure(request, understanding_meta, strategy)
            
            # Estimate execution parameters
            execution_params = await self._plan_execution_parameters(request, understanding_meta, strategy)
            
            # Build plan summary
            content = self._build_plan_summary(
                strategy, model_plan, external_calls_plan, file_processing_plan, 
                memory_plan, structure_plan, execution_params
            )
            
            metadata = {
                "strategy": strategy["type"],
                "confidence": strategy["confidence"],
                "complexity": strategy["complexity"],
                "model_recommendation": model_plan["recommended_model"],
                "needs_external_calls": external_calls_plan["needs_calls"],
                "external_call_types": external_calls_plan["call_types"],
                "file_processing": file_processing_plan,
                "memory_usage": memory_plan,
                "response_structure": structure_plan["type"],
                "estimated_tokens": execution_params["estimated_output_tokens"],
                "temperature": execution_params["temperature"],
                "max_tokens": execution_params["max_tokens"],
                "streaming_recommended": execution_params["streaming_recommended"]
            }
            
            result = UPEEResult(
                phase=UPEEPhase.PLAN,
                content=content,
                metadata=metadata,
                completed=True
            )
            
            self.logger.info(
                "Planning phase completed",
                request_id=request_id,
                strategy=metadata["strategy"],
                model=metadata["model_recommendation"],
                needs_external_calls=metadata["needs_external_calls"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Planning phase failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            return UPEEResult(
                phase=UPEEPhase.PLAN,
                content=f"Planning failed: {str(e)}",
                metadata={"error": str(e)},
                completed=False,
                error=str(e)
            )
    
    async def _determine_strategy(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the overall response strategy."""
        
        intent = understanding_meta.get("intent", "general")
        complexity = understanding_meta.get("complexity", "simple")
        file_count = understanding_meta.get("file_count", 0)
        
        # Strategy mapping based on intent and complexity
        strategy_map = {
            "conversation": {
                "type": "direct_response",
                "confidence": 0.9,
                "complexity": "simple",
                "approach": "conversational"
            },
            "question": {
                "type": "informative_response",
                "confidence": 0.8,
                "complexity": complexity,
                "approach": "explanatory"
            },
            "request": {
                "type": "helpful_response",
                "confidence": 0.8,
                "complexity": complexity,
                "approach": "solution_oriented"
            },
            "task": {
                "type": "structured_execution",
                "confidence": 0.7,
                "complexity": "moderate" if complexity == "simple" else "complex",
                "approach": "step_by_step"
            },
            "analysis": {
                "type": "analytical_response",
                "confidence": 0.7,
                "complexity": "moderate" if complexity == "simple" else "complex",
                "approach": "detailed_analysis"
            }
        }
        
        base_strategy = strategy_map.get(intent, strategy_map["question"])
        
        # Adjust strategy based on file context
        if file_count > 0:
            base_strategy["approach"] = "context_aware_" + base_strategy["approach"]
            if file_count > 3:
                base_strategy["complexity"] = "complex"
                base_strategy["confidence"] *= 0.9
        
        # Adjust strategy based on conversation history
        history_count = understanding_meta.get("conversation_history", {}).get("message_count", 0)
        if history_count > 0:
            base_strategy["approach"] = "memory_aware_" + base_strategy["approach"]
            if history_count > 5:
                base_strategy["complexity"] = "moderate" if base_strategy["complexity"] == "simple" else "complex"
        
        return base_strategy
    
    async def _plan_model_selection(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan which model to use for execution."""
        
        # Use explicitly requested model if provided
        if request.model:
            return {
                "recommended_model": request.model,
                "reason": "explicitly_requested",
                "confidence": 1.0
            }
        
        # Model selection logic based on complexity and intent
        complexity = strategy["complexity"]
        intent = understanding_meta.get("intent", "general")
        
        # Default to settings default
        recommended_model = self.settings.resolved_default_model
        reason = "default"
        confidence = 0.7
        
        # Upgrade for complex tasks
        if complexity == "complex" or intent in ["task", "analysis"]:
            if self.settings.openai_api_key:
                recommended_model = "gpt-4"
                reason = "complexity_upgrade"
                confidence = 0.8
        
        # Consider file context
        file_count = understanding_meta.get("file_count", 0)
        if file_count > 0:
            if self.settings.openai_api_key:
                recommended_model = "gpt-4"
                reason = "file_context_handling"
                confidence = 0.9
        
        return {
            "recommended_model": recommended_model,
            "reason": reason,
            "confidence": confidence
        }
    
    async def _plan_external_calls(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan external API calls (gRPC worker agents)."""
        
        intent = understanding_meta.get("intent", "general")
        complexity = strategy["complexity"]
        requires_external = understanding_meta.get("requires_external_calls", False)
        file_count = understanding_meta.get("file_count", 0)
        
        needs_calls = False
        call_types = []
        
        # Check for specific keywords that indicate external processing needs
        message_lower = request.message.lower()
        
        # Code analysis needs
        if any(keyword in message_lower for keyword in [
            "analyze code", "code review", "security", "complexity", "refactor", 
            "optimization", "bug", "vulnerability", "lint"
        ]):
            needs_calls = True
            call_types.append("code_analysis")
        
        # File processing needs
        if file_count > 0 and any(keyword in message_lower for keyword in [
            "process", "extract", "convert", "transform", "parse", "summary", "structure"
        ]):
            needs_calls = True
            call_types.append("file_processing")
        
        # Data extraction needs
        if any(keyword in message_lower for keyword in [
            "extract data", "parse data", "data analysis", "statistics", "metrics",
            "csv", "json", "database", "query"
        ]):
            needs_calls = True
            call_types.append("data_extraction")
        
        # Heavy computation needs
        if any(keyword in message_lower for keyword in [
            "calculate", "compute", "algorithm", "processing", "batch", "large dataset"
        ]) and complexity in ["moderate", "complex"]:
            needs_calls = True
            call_types.append("computation")
        
        # Fallback: complex tasks that could benefit from worker processing
        if intent in ["task", "analysis"] and complexity == "complex" and not call_types:
            needs_calls = True
            call_types.append("worker_task")
        
        # Multiple files suggest need for distributed processing
        if file_count > 3:
            needs_calls = True
            if "file_processing" not in call_types:
                call_types.append("file_processing")
        
        return {
            "needs_calls": needs_calls,
            "call_types": call_types,
            "estimated_calls": len(call_types),
            "parallel_execution": len(call_types) > 1,
            "reasoning": self._get_external_call_reasoning(call_types, intent, complexity, file_count)
        }
    
    def _get_external_call_reasoning(
        self, 
        call_types: list, 
        intent: str, 
        complexity: str, 
        file_count: int
    ) -> str:
        """Generate reasoning for external call decisions."""
        if not call_types:
            return "No external calls needed for this request"
        
        reasons = []
        
        for call_type in call_types:
            if call_type == "code_analysis":
                reasons.append("specialized code analysis required")
            elif call_type == "file_processing":
                reasons.append(f"processing {file_count} files efficiently")
            elif call_type == "data_extraction":
                reasons.append("data extraction and analysis needed")
            elif call_type == "computation":
                reasons.append("heavy computation required")
            elif call_type == "worker_task":
                reasons.append("complex task benefits from distributed processing")
        
        return f"External calls planned: {', '.join(reasons)}"
    
    async def _plan_file_processing(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan how to handle file processing during execution."""
        
        file_processing = understanding_meta.get("file_processing", {})
        processed_files = understanding_meta.get("processed_files", [])
        
        if not processed_files:
            return {
                "approach": "no_files",
                "strategy": "direct_response",
                "context_inclusion": "none"
            }
        
        files_with_content = file_processing.get("files_with_content", 0)
        files_with_signed_urls = file_processing.get("files_with_signed_urls", 0)
        total_file_size = file_processing.get("total_file_size", 0)
        
        # Determine processing approach
        if total_file_size > 50000:  # 50KB threshold
            approach = "selective_content"
            strategy = "summarize_and_highlight"
        elif files_with_signed_urls > 0:
            approach = "mixed_content"
            strategy = "inline_plus_references"
        else:
            approach = "full_content"
            strategy = "complete_context"
        
        # Determine context inclusion strategy
        if len(processed_files) > 5:
            context_inclusion = "summarized"
        elif total_file_size > 20000:
            context_inclusion = "selective"
        else:
            context_inclusion = "full"
        
        return {
            "approach": approach,
            "strategy": strategy,
            "context_inclusion": context_inclusion,
            "file_count": len(processed_files),
            "files_with_content": files_with_content,
            "files_with_signed_urls": files_with_signed_urls,
            "total_size": total_file_size
        }
    
    async def _plan_memory_usage(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan how to use conversation history (short-term memory)."""
        
        history_info = understanding_meta.get("conversation_history", {})
        message_count = history_info.get("message_count", 0)
        
        if message_count == 0:
            return {
                "approach": "no_history",
                "strategy": "fresh_context",
                "context_inclusion": "none"
            }
        
        history_tokens = history_info.get("total_tokens_estimate", 0)
        was_truncated = history_info.get("history_truncated", False)
        files_in_history = history_info.get("files_in_history", 0)
        
        # Determine memory approach
        if history_tokens > 2000:
            approach = "selective_memory"
            strategy = "recent_focus"
        elif message_count > 8:
            approach = "contextual_memory"
            strategy = "relevant_extraction"
        else:
            approach = "full_memory"
            strategy = "complete_context"
        
        # Determine context inclusion
        if was_truncated:
            context_inclusion = "truncated_with_summary"
        elif files_in_history > 0:
            context_inclusion = "history_with_file_refs"
        else:
            context_inclusion = "full_history"
        
        return {
            "approach": approach,
            "strategy": strategy,
            "context_inclusion": context_inclusion,
            "message_count": message_count,
            "history_tokens": history_tokens,
            "was_truncated": was_truncated,
            "files_in_history": files_in_history
        }
    
    async def _plan_response_structure(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan the structure of the response."""
        
        intent = understanding_meta.get("intent", "general")
        approach = strategy.get("approach", "direct")
        file_count = understanding_meta.get("file_count", 0)
        
        # Determine response structure
        if intent == "conversation":
            structure_type = "conversational"
        elif intent == "question":
            structure_type = "explanatory"
        elif intent == "task":
            structure_type = "step_by_step"
        elif intent == "analysis" and file_count > 0:
            structure_type = "analytical_with_context"
        else:
            structure_type = "structured"
        
        # Plan sections
        sections = []
        if structure_type == "step_by_step":
            sections = ["introduction", "steps", "conclusion"]
        elif structure_type == "analytical_with_context":
            sections = ["context_summary", "analysis", "insights", "recommendations"]
        elif structure_type == "explanatory":
            sections = ["explanation", "examples", "summary"]
        else:
            sections = ["response"]
        
        return {
            "type": structure_type,
            "sections": sections,
            "estimated_length": self._estimate_response_length(intent, approach, file_count)
        }
    
    async def _plan_execution_parameters(
        self, 
        request: ChatRequest, 
        understanding_meta: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan execution parameters for the LLM call."""
        
        intent = understanding_meta.get("intent", "general")
        complexity = strategy["complexity"]
        
        # Base parameters
        temperature = request.temperature if hasattr(request, 'temperature') else 0.7
        max_tokens = request.max_tokens if hasattr(request, 'max_tokens') else None
        
        # Adjust temperature based on intent
        if intent == "analysis":
            temperature = max(0.3, temperature - 0.2)  # More focused
        elif intent == "task":
            temperature = max(0.4, temperature - 0.1)  # Slightly more focused
        elif intent == "conversation":
            temperature = min(0.9, temperature + 0.1)  # More creative
        
        # Estimate output tokens
        estimated_tokens = self._estimate_output_tokens(intent, complexity, understanding_meta)
        
        # Set max_tokens if not specified
        if max_tokens is None:
            max_tokens = min(estimated_tokens * 2, 100000)  # 2x estimate, capped at 100000
        
        # Debug logging for token planning
        self.logger.info(
            "Token planning details",
            intent=intent,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            calculated_max_tokens=estimated_tokens * 2,
            final_max_tokens=max_tokens,
            request_max_tokens=request.max_tokens if hasattr(request, 'max_tokens') else None,
            file_count=understanding_meta.get("file_count", 0)
        )
        
        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "estimated_output_tokens": estimated_tokens,
            "streaming_recommended": True,  # Always recommend streaming for chat
            "stop_sequences": []
        }
    
    def _estimate_response_length(
        self, 
        intent: str, 
        approach: str, 
        file_count: int
    ) -> str:
        """Estimate response length category."""
        
        base_length = {
            "conversation": "short",
            "question": "medium",
            "request": "medium",
            "task": "long",
            "analysis": "long"
        }
        
        length = base_length.get(intent, "medium")
        
        # Adjust for file context
        if file_count > 0:
            if length == "short":
                length = "medium"
            elif length == "medium":
                length = "long"
        
        return length
    
    def _estimate_output_tokens(
        self, 
        intent: str, 
        complexity: str, 
        understanding_meta: Dict[str, Any]
    ) -> int:
        """Estimate number of output tokens needed."""
        
        base_tokens = {
            "conversation": 3000,     # Increased from 50
            "question": 8000,         # Increased from 200
            "request": 10000,         # Increased from 300
            "task": 15000,            # Increased from 500
            "analysis": 20000         # Increased from 600
        }
        
        tokens = base_tokens.get(intent, 8000)  # Better default
        
        # Adjust for complexity
        complexity_multiplier = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.5  # Increased from 2.0
        }
        tokens = int(tokens * complexity_multiplier.get(complexity, 1.0))
        
        # Adjust for file context
        file_count = understanding_meta.get("file_count", 0)
        if file_count > 0:
            tokens += file_count * 200  # Increased from 100 tokens per file
        
        return tokens
    
    def _build_plan_summary(
        self,
        strategy: Dict[str, Any],
        model_plan: Dict[str, Any],
        external_calls_plan: Dict[str, Any],
        file_processing_plan: Dict[str, Any],
        memory_plan: Dict[str, Any],
        structure_plan: Dict[str, Any],
        execution_params: Dict[str, Any]
    ) -> str:
        """Build a human-readable summary of the plan."""
        
        summary_parts = [
            f"Strategy: {strategy['type']} ({strategy['approach']})",
            f"Model: {model_plan['recommended_model']} ({model_plan['reason']})",
            f"Structure: {structure_plan['type']}"
        ]
        
        # Add file processing info
        if file_processing_plan["approach"] != "no_files":
            summary_parts.append(f"Files: {file_processing_plan['approach']} ({file_processing_plan['file_count']} files)")
        
        # Add memory usage info
        if memory_plan["approach"] != "no_history":
            summary_parts.append(f"Memory: {memory_plan['approach']} ({memory_plan['message_count']} msgs)")
        
        if external_calls_plan["needs_calls"]:
            summary_parts.append(f"External calls: {', '.join(external_calls_plan['call_types'])}")
        
        summary_parts.append(f"Est. tokens: {execution_params['estimated_output_tokens']}")
        
        return " | ".join(summary_parts) 