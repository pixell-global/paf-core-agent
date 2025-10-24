"""Understand Node - AI-powered analysis of user intent and context.

This node analyzes the user's message to understand:
- Intent (what the user wants to accomplish)
- Topics and entities mentioned
- Complexity level
- Required context

The understanding feeds into the AI routing decision.
"""

import json
from datetime import datetime
from typing import Dict, Any
import structlog

from src.langgraph_upee.state import UPEEState
from src.llm_providers import LLMProviderManager, LLMRequest
from src.settings import Settings

logger = structlog.get_logger()


async def understand_node(state: UPEEState, settings: Settings, llm_manager: LLMProviderManager) -> UPEEState:
    """
    Understand Node: Analyze user message to extract intent and context.

    This is an AI-native node that uses an LLM to deeply understand what the user wants,
    preparing the context for intelligent routing decisions.

    Args:
        state: Current UPEE state containing user_message
        settings: Application settings
        llm_manager: LLM provider manager for AI analysis

    Returns:
        Updated state with understanding metadata
    """
    request_id = state.get("request_id", "unknown")
    user_message = state.get("user_message", "")

    logger.info(
        "Starting understand phase",
        request_id=request_id,
        message_length=len(user_message)
    )

    # Track execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("understand")

    # Track timestamps
    timestamps = state.get("timestamps", {})
    timestamps["understand_start"] = datetime.utcnow().isoformat()

    try:
        # Use LLM to analyze user intent
        understanding = await _analyze_user_intent_with_llm(
            user_message,
            state.get("conversation_history"),
            llm_manager,
            settings
        )

        timestamps["understand_end"] = datetime.utcnow().isoformat()

        # Update state with understanding
        state["understanding"] = understanding
        state["intent_summary"] = understanding.get("intent_summary", "")
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        logger.info(
            "Understanding phase completed",
            request_id=request_id,
            intent=understanding.get("primary_intent"),
            complexity=understanding.get("complexity"),
            topics=understanding.get("topics", [])
        )

        return state

    except Exception as e:
        logger.error(
            "Understanding phase failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )

        timestamps["understand_end"] = datetime.utcnow().isoformat()

        # Update state with error
        state["error"] = f"Understanding failed: {str(e)}"
        state["error_stage"] = "understand"
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        return state


async def _analyze_user_intent_with_llm(
    user_message: str,
    conversation_history: Any,
    llm_manager: LLMProviderManager,
    settings: Settings
) -> Dict[str, Any]:
    """
    Use LLM to analyze user intent in a structured way.

    This provides rich context for the AI routing decision that follows.
    """

    # Build prompt for intent analysis
    prompt = f"""Analyze the following user message and extract structured information about their intent.

User Message: "{user_message}"

Provide your analysis in JSON format with these fields:
{{
    "intent_summary": "One sentence summary of what the user wants",
    "primary_intent": "The main goal (choose one: question, request, task, analysis, conversation, search, creative)",
    "topics": ["list", "of", "main", "topics", "mentioned"],
    "entities": ["specific", "entities", "like", "names", "products", "technologies"],
    "complexity": "simple|moderate|complex (based on the task difficulty)",
    "requires_specialized_knowledge": true/false,
    "domain": "The domain area (e.g., 'general', 'reddit', 'marketing', 'technical', 'creative')",
    "keywords": ["important", "keywords", "for", "routing"]
}}

Respond ONLY with valid JSON, no additional text."""

    try:
        llm_request = LLMRequest(
            model=settings.default_model,
            prompt=prompt,
            temperature=0.3,  # Low temperature for structured output
            max_tokens=500,
            stream=False
        )

        response = await llm_manager.get_completion(llm_request)

        if response.content:
            # Parse JSON response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            understanding = json.loads(content)

            logger.debug(
                "LLM intent analysis completed",
                intent=understanding.get("primary_intent"),
                domain=understanding.get("domain")
            )

            return understanding

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse LLM response as JSON",
            error=str(e),
            response_content=response.content[:200] if response.content else None
        )
    except Exception as e:
        logger.error(
            "Intent analysis failed",
            error=str(e),
            exc_info=True
        )

    # Fallback to simple analysis if LLM fails
    return _fallback_intent_analysis(user_message)


def _fallback_intent_analysis(user_message: str) -> Dict[str, Any]:
    """
    Simple rule-based fallback if LLM analysis fails.
    """
    message_lower = user_message.lower()

    # Simple intent detection
    if any(word in message_lower for word in ["find", "search", "discover", "show me", "list"]):
        primary_intent = "search"
    elif "?" in user_message:
        primary_intent = "question"
    elif any(word in message_lower for word in ["create", "generate", "make", "write"]):
        primary_intent = "creative"
    else:
        primary_intent = "general"

    # Extract basic topics
    topics = []
    if "reddit" in message_lower or "subreddit" in message_lower:
        topics.append("reddit")
    if "ai" in message_lower or "machine learning" in message_lower:
        topics.append("ai")

    return {
        "intent_summary": user_message[:100],
        "primary_intent": primary_intent,
        "topics": topics,
        "entities": [],
        "complexity": "moderate",
        "requires_specialized_knowledge": False,
        "domain": "general",
        "keywords": user_message.split()[:10]
    }
