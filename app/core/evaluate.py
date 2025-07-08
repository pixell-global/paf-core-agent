"""Evaluation phase - Assess response quality and determine if refinement is needed."""

import time
from typing import Dict, Any, Optional

from app.schemas import ChatRequest, UPEEResult, UPEEPhase
from app.utils.logging_config import get_logger
from app.settings import Settings


class EvaluatePhase:
    """
    Evaluation phase of the UPEE loop.
    
    Responsible for:
    - Assessing response quality and completeness
    - Checking alignment with user intent and plan
    - Determining if refinement is needed
    - Collecting quality metrics
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("evaluate_phase")
    
    async def process(
        self, 
        request: ChatRequest, 
        request_id: str,
        understanding_result: Optional[UPEEResult] = None,
        plan_result: Optional[UPEEResult] = None,
        execute_result: Optional[UPEEResult] = None
    ) -> UPEEResult:
        """
        Process the evaluation phase.
        
        Args:
            request: The original chat request
            request_id: Request tracking ID
            understanding_result: Result from understanding phase
            plan_result: Result from planning phase
            execute_result: Result from execution phase
            
        Returns:
            UPEEResult with evaluation assessment
        """
        self.logger.info(
            "Starting evaluation phase",
            request_id=request_id,
            has_understanding=understanding_result is not None,
            has_plan=plan_result is not None,
            has_execute=execute_result is not None
        )
        
        try:
            # Extract metadata from previous phases
            understanding_meta = understanding_result.metadata if understanding_result else {}
            plan_meta = plan_result.metadata if plan_result else {}
            execute_meta = execute_result.metadata if execute_result else {}
            
            # Perform quality assessments
            completeness_score = await self._assess_completeness(
                request, understanding_meta, plan_meta, execute_meta
            )
            
            accuracy_score = await self._assess_accuracy(
                request, understanding_meta, plan_meta, execute_meta
            )
            
            relevance_score = await self._assess_relevance(
                request, understanding_meta, plan_meta, execute_meta
            )
            
            coherence_score = await self._assess_coherence(
                request, understanding_meta, plan_meta, execute_meta
            )
            
            # Calculate overall quality score
            quality_score = await self._calculate_overall_quality(
                completeness_score, accuracy_score, relevance_score, coherence_score
            )
            
            # Determine if refinement is needed
            needs_refinement = await self._needs_refinement(
                quality_score, understanding_meta, plan_meta, execute_meta
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                quality_score, completeness_score, accuracy_score, 
                relevance_score, coherence_score, needs_refinement
            )
            
            # Build evaluation summary
            content = self._build_evaluation_summary(
                quality_score, completeness_score, accuracy_score,
                relevance_score, coherence_score, needs_refinement, recommendations
            )
            
            metadata = {
                "quality_score": quality_score,
                "completeness_score": completeness_score,
                "accuracy_score": accuracy_score,
                "relevance_score": relevance_score,
                "coherence_score": coherence_score,
                "needs_refinement": needs_refinement,
                "recommendations": recommendations,
                "evaluation_criteria": {
                    "intent_alignment": self._check_intent_alignment(understanding_meta, execute_meta),
                    "plan_execution": self._check_plan_execution(plan_meta, execute_meta),
                    "response_structure": self._check_response_structure(plan_meta, execute_meta)
                }
            }
            
            result = UPEEResult(
                phase=UPEEPhase.EVALUATE,
                content=content,
                metadata=metadata,
                completed=True
            )
            
            self.logger.info(
                "Evaluation phase completed",
                request_id=request_id,
                quality_score=quality_score,
                needs_refinement=needs_refinement,
                recommendations_count=len(recommendations)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Evaluation phase failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            return UPEEResult(
                phase=UPEEPhase.EVALUATE,
                content=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)},
                completed=False,
                error=str(e)
            )
    
    async def _assess_completeness(
        self,
        request: ChatRequest,
        understanding_meta: Dict[str, Any],
        plan_meta: Dict[str, Any],
        execute_meta: Dict[str, Any]
    ) -> float:
        """Assess how complete the response is."""
        
        score = 0.8  # Base score
        
        # Check if execution was successful
        if execute_meta.get("tokens_generated", 0) > 0:
            score += 0.1
        
        # Check if planned tokens were roughly met
        planned_tokens = plan_meta.get("estimated_tokens", 0)
        actual_tokens = execute_meta.get("tokens_generated", 0)
        
        if planned_tokens > 0:
            token_ratio = actual_tokens / planned_tokens
            if 0.5 <= token_ratio <= 2.0:  # Reasonable range
                score += 0.1
        
        # Check if external calls were made when planned
        planned_external = plan_meta.get("needs_external_calls", False)
        actual_external = execute_meta.get("external_calls_made", 0)
        
        if planned_external and actual_external > 0:
            score += 0.05
        elif not planned_external:
            score += 0.05  # Good if no external calls were needed
        
        return min(score, 1.0)
    
    async def _assess_accuracy(
        self,
        request: ChatRequest,
        understanding_meta: Dict[str, Any],
        plan_meta: Dict[str, Any],
        execute_meta: Dict[str, Any]
    ) -> float:
        """Assess the accuracy of the response."""
        
        # For dummy implementation, we'll use heuristics
        # In production, this might involve fact-checking or validation
        
        score = 0.7  # Base score
        
        # Check if execution completed without errors
        if not execute_meta.get("error"):
            score += 0.2
        
        # Check model appropriateness
        model_used = execute_meta.get("model_used", "")
        complexity = understanding_meta.get("complexity", "simple")
        
        if complexity == "complex" and "gpt-4" in model_used.lower():
            score += 0.1  # Appropriate model for complex tasks
        elif complexity == "simple":
            score += 0.1  # Any model is fine for simple tasks
        
        return min(score, 1.0)
    
    async def _assess_relevance(
        self,
        request: ChatRequest,
        understanding_meta: Dict[str, Any],
        plan_meta: Dict[str, Any],
        execute_meta: Dict[str, Any]
    ) -> float:
        """Assess how relevant the response is to the user's intent."""
        
        score = 0.8  # Base score
        
        # Check intent alignment
        intent = understanding_meta.get("intent", "general")
        strategy = plan_meta.get("strategy", "")
        
        # Map intents to expected strategies
        intent_strategy_map = {
            "question": "informative_response",
            "task": "structured_execution",
            "analysis": "analytical_response",
            "conversation": "direct_response",
            "request": "helpful_response"
        }
        
        expected_strategy = intent_strategy_map.get(intent)
        if expected_strategy and expected_strategy in strategy:
            score += 0.1
        
        # Check if file context was considered when provided
        if request.files and execute_meta.get("external_calls_made", 0) >= 0:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _assess_coherence(
        self,
        request: ChatRequest,
        understanding_meta: Dict[str, Any],
        plan_meta: Dict[str, Any],
        execute_meta: Dict[str, Any]
    ) -> float:
        """Assess the coherence and structure of the response."""
        
        score = 0.8  # Base score
        
        # Check if planned structure was followed
        planned_structure = plan_meta.get("response_structure", "")
        if planned_structure:
            score += 0.1  # Good if structure was planned
        
        # Check token generation consistency
        tokens_generated = execute_meta.get("tokens_generated", 0)
        if tokens_generated > 10:  # Minimum viable response
            score += 0.1
        
        return min(score, 1.0)
    
    async def _calculate_overall_quality(
        self,
        completeness: float,
        accuracy: float,
        relevance: float,
        coherence: float
    ) -> float:
        """Calculate overall quality score using weighted average."""
        
        # Weighted average of quality dimensions
        weights = {
            "completeness": 0.25,
            "accuracy": 0.30,
            "relevance": 0.30,
            "coherence": 0.15
        }
        
        overall = (
            completeness * weights["completeness"] +
            accuracy * weights["accuracy"] +
            relevance * weights["relevance"] +
            coherence * weights["coherence"]
        )
        
        return round(overall, 3)
    
    async def _needs_refinement(
        self,
        quality_score: float,
        understanding_meta: Dict[str, Any],
        plan_meta: Dict[str, Any],
        execute_meta: Dict[str, Any]
    ) -> bool:
        """Determine if the response needs refinement."""
        
        # Quality threshold
        quality_threshold = 0.7
        
        if quality_score < quality_threshold:
            return True
        
        # Check for execution errors
        if execute_meta.get("error"):
            return True
        
        # Check for very short responses to complex requests
        complexity = understanding_meta.get("complexity", "simple")
        tokens_generated = execute_meta.get("tokens_generated", 0)
        
        if complexity == "complex" and tokens_generated < 50:
            return True
        
        return False
    
    async def _generate_recommendations(
        self,
        quality_score: float,
        completeness: float,
        accuracy: float,
        relevance: float,
        coherence: float,
        needs_refinement: bool
    ) -> list:
        """Generate recommendations for improvement."""
        
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Overall quality below threshold - consider refinement")
        
        if completeness < 0.7:
            recommendations.append("Response may be incomplete - ensure all aspects are addressed")
        
        if accuracy < 0.7:
            recommendations.append("Accuracy concerns detected - verify information quality")
        
        if relevance < 0.7:
            recommendations.append("Response may not fully address user intent")
        
        if coherence < 0.7:
            recommendations.append("Response structure could be improved")
        
        if not recommendations and not needs_refinement:
            recommendations.append("Response quality is satisfactory")
        
        return recommendations
    
    def _check_intent_alignment(
        self, 
        understanding_meta: Dict[str, Any], 
        execute_meta: Dict[str, Any]
    ) -> bool:
        """Check if execution aligned with understood intent."""
        intent = understanding_meta.get("intent", "general")
        execution_success = not execute_meta.get("error", False)
        return execution_success  # Simple check for now
    
    def _check_plan_execution(
        self, 
        plan_meta: Dict[str, Any], 
        execute_meta: Dict[str, Any]
    ) -> bool:
        """Check if execution followed the plan."""
        planned_model = plan_meta.get("model_recommendation", "")
        used_model = execute_meta.get("model_used", "")
        return planned_model == used_model if planned_model else True
    
    def _check_response_structure(
        self, 
        plan_meta: Dict[str, Any], 
        execute_meta: Dict[str, Any]
    ) -> bool:
        """Check if response followed planned structure."""
        # For now, assume structure was followed if execution succeeded
        return not execute_meta.get("error", False)
    
    def _build_evaluation_summary(
        self,
        quality_score: float,
        completeness: float,
        accuracy: float,
        relevance: float,
        coherence: float,
        needs_refinement: bool,
        recommendations: list
    ) -> str:
        """Build human-readable evaluation summary."""
        
        summary_parts = [
            f"Quality Score: {quality_score:.3f}",
            f"Completeness: {completeness:.3f}",
            f"Accuracy: {accuracy:.3f}",
            f"Relevance: {relevance:.3f}",
            f"Coherence: {coherence:.3f}"
        ]
        
        if needs_refinement:
            summary_parts.append("⚠️ Refinement needed")
        else:
            summary_parts.append("✅ Quality acceptable")
        
        if recommendations:
            summary_parts.append(f"Recommendations: {len(recommendations)} items")
        
        return " | ".join(summary_parts) 