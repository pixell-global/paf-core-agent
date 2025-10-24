"""Evaluate Node - Assess response quality and determine if refinement needed.

This node evaluates the quality of responses (from core or agent) and provides
feedback on accuracy, completeness, and relevance.
"""

import json
from datetime import datetime
from typing import Dict, Any
import structlog

from src.langgraph_upee.state import UPEEState
from src.llm_providers import LLMProviderManager, LLMRequest
from src.settings import Settings

logger = structlog.get_logger()


async def evaluate_node(state: UPEEState, settings: Settings, llm_manager: LLMProviderManager) -> UPEEState:
    """
    Evaluate Node: Assess response quality.

    Uses an LLM to evaluate the response against the user's original query,
    providing quality metrics and feedback.

    Args:
        state: Current UPEE state with response
        settings: Application settings
        llm_manager: LLM provider manager for quality assessment

    Returns:
        Updated state with quality_score, quality_feedback, needs_refinement
    """
    request_id = state.get("request_id", "unknown")
    user_message = state.get("user_message", "")
    response = state.get("response", "")

    logger.info(
        "Starting evaluation phase",
        request_id=request_id,
        response_length=len(response)
    )

    # Track execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("evaluate")

    # Track timestamps
    timestamps = state.get("timestamps", {})
    timestamps["evaluate_start"] = datetime.utcnow().isoformat()

    try:
        # Use LLM to evaluate response quality
        evaluation = await _evaluate_response_quality(
            user_message,
            response,
            state,
            llm_manager,
            settings
        )

        timestamps["evaluate_end"] = datetime.utcnow().isoformat()

        # Update state with evaluation
        state["quality_score"] = evaluation.get("quality_score", 0.75)
        state["quality_feedback"] = evaluation.get("feedback", "")
        state["needs_refinement"] = evaluation.get("needs_refinement", False)
        state["refinement_suggestions"] = evaluation.get("suggestions", "")
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        logger.info(
            "Evaluation completed",
            request_id=request_id,
            quality_score=state["quality_score"],
            needs_refinement=state["needs_refinement"]
        )

        return state

    except Exception as e:
        logger.error(
            "Evaluation failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )

        timestamps["evaluate_end"] = datetime.utcnow().isoformat()

        # Provide default evaluation if LLM evaluation fails
        state["quality_score"] = 0.7  # Assume reasonable quality
        state["quality_feedback"] = "Evaluation not available"
        state["needs_refinement"] = False
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        return state


async def _evaluate_response_quality(
    user_message: str,
    response: str,
    state: UPEEState,
    llm_manager: LLMProviderManager,
    settings: Settings
) -> Dict[str, Any]:
    """
    Use LLM to evaluate response quality.

    Assesses accuracy, completeness, and relevance of the response.
    """

    understanding = state.get("understanding", {})
    routing_decision = state.get("routing_decision", "core")
    selected_agent = state.get("selected_agent")

    # Build evaluation prompt
    prompt = f"""Evaluate the quality of this AI response.

User Query: "{user_message}"
User Intent: {understanding.get("intent_summary", "Not analyzed")}

Response Source: {"Agent: " + selected_agent.name if selected_agent else "Core AI System"}

Response:
{response}

Evaluate the response on these criteria:
1. **Accuracy**: Is the information correct and factual?
2. **Completeness**: Does it fully address the user's query?
3. **Relevance**: Is it relevant to what the user asked?
4. **Clarity**: Is it well-organized and easy to understand?

Provide your evaluation in JSON format:
{{
    "quality_score": 0.0 to 1.0 (overall quality rating),
    "feedback": "Brief explanation of the quality assessment",
    "needs_refinement": true/false (whether response should be improved),
    "suggestions": "Specific suggestions for improvement if needed"
}}

Scoring guide:
- 0.9-1.0: Excellent - Complete, accurate, highly relevant
- 0.7-0.89: Good - Mostly complete and accurate
- 0.5-0.69: Fair - Partially addresses query, may need improvement
- Below 0.5: Poor - Incomplete or inaccurate, needs refinement

Respond ONLY with valid JSON, no additional text."""

    try:
        llm_request = LLMRequest(
            model=settings.default_model,
            prompt=prompt,
            temperature=0.2,  # Low temperature for consistent evaluation
            max_tokens=300,
            stream=False
        )

        eval_response = await llm_manager.get_completion(llm_request)

        if eval_response.content:
            # Parse JSON response
            content = eval_response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            evaluation = json.loads(content)

            logger.debug(
                "Quality evaluation completed",
                quality_score=evaluation.get("quality_score"),
                needs_refinement=evaluation.get("needs_refinement")
            )

            return evaluation

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse evaluation response as JSON",
            error=str(e),
            response_content=eval_response.content[:200] if eval_response.content else None
        )
    except Exception as e:
        logger.error(
            "Quality evaluation failed",
            error=str(e),
            exc_info=True
        )

    # Fallback evaluation if LLM fails
    return _fallback_evaluation(response)


def _fallback_evaluation(response: str) -> Dict[str, Any]:
    """
    Simple rule-based fallback evaluation if LLM evaluation fails.

    Provides basic quality metrics based on response characteristics.
    """
    # Basic quality heuristics
    response_length = len(response)

    if response_length < 20:
        quality_score = 0.4  # Very short response, likely incomplete
        feedback = "Response seems too brief"
        needs_refinement = True
        suggestions = "Provide more detailed information"
    elif response_length < 100:
        quality_score = 0.65  # Short but may be sufficient for simple queries
        feedback = "Response is concise"
        needs_refinement = False
        suggestions = ""
    elif response_length > 2000:
        quality_score = 0.75  # Long response, likely comprehensive
        feedback = "Response is detailed and comprehensive"
        needs_refinement = False
        suggestions = ""
    else:
        quality_score = 0.7  # Moderate length, likely reasonable
        feedback = "Response appears reasonable"
        needs_refinement = False
        suggestions = ""

    # Check for error indicators
    if "apologize" in response.lower() or "error" in response.lower():
        quality_score *= 0.7  # Reduce score if there are errors
        needs_refinement = True
        suggestions = "Address the error mentioned in the response"

    return {
        "quality_score": quality_score,
        "feedback": feedback,
        "needs_refinement": needs_refinement,
        "suggestions": suggestions
    }
