"""Execute PAF Core Node - Handle requests with PAF Core's LLM.

This node executes when routing_decision == "core".
Uses PAF Core's LLM providers to generate responses for general queries.
"""

from datetime import datetime
from typing import Dict, Any
import structlog

from src.langgraph_upee.state import UPEEState
from src.llm_providers import LLMProviderManager, LLMRequest
from src.settings import Settings

logger = structlog.get_logger()


async def execute_core_node(state: UPEEState, settings: Settings, llm_manager: LLMProviderManager) -> UPEEState:
    """
    Execute PAF Core Node: Generate response using PAF Core's LLM.

    This node is called when the routing decision is "core" - meaning the
    request should be handled by PAF Core's general intelligence rather than
    a specialized agent.

    Args:
        state: Current UPEE state with user_message and understanding
        settings: Application settings
        llm_manager: LLM provider manager

    Returns:
        Updated state with response
    """
    request_id = state.get("request_id", "unknown")
    user_message = state.get("user_message", "")
    model = state.get("model") or settings.default_model
    temperature = state.get("temperature", 0.7)

    logger.info(
        "Executing with PAF Core LLM",
        request_id=request_id,
        model=model
    )

    # Track execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("execute_core")

    # Track timestamps
    timestamps = state.get("timestamps", {})
    timestamps["execute_core_start"] = datetime.utcnow().isoformat()

    try:
        # Build prompt for LLM
        prompt = _build_core_prompt(user_message, state)

        # Call LLM
        llm_request = LLMRequest(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=2000,
            stream=False
        )

        response = await llm_manager.get_completion(llm_request)

        timestamps["execute_core_end"] = datetime.utcnow().isoformat()

        if response.content:
            # Update state with response
            state["response"] = response.content
            state["response_metadata"] = {
                "model": response.model,
                "provider": response.provider,
                "execution_type": "core",
                "finish_reason": response.finish_reason,
                "token_count": response.token_count
            }
            state["timestamps"] = timestamps
            state["execution_path"] = execution_path

            logger.info(
                "PAF Core execution completed",
                request_id=request_id,
                response_length=len(response.content),
                model=response.model
            )

            return state
        else:
            raise Exception("LLM returned empty response")

    except Exception as e:
        logger.error(
            "PAF Core execution failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )

        timestamps["execute_core_end"] = datetime.utcnow().isoformat()

        # Update state with error
        state["error"] = f"Core execution failed: {str(e)}"
        state["error_stage"] = "execute_core"
        state["response"] = f"I apologize, but I encountered an error processing your request: {str(e)}"
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        return state


def _build_core_prompt(user_message: str, state: UPEEState) -> str:
    """
    Build prompt for PAF Core LLM.

    Includes user message, conversation history if available, and any file context.
    """
    prompt_parts = []

    # Add system context
    understanding = state.get("understanding", {})
    if understanding:
        intent_summary = understanding.get("intent_summary", "")
        if intent_summary:
            prompt_parts.append(f"User Intent: {intent_summary}")

    # Add conversation history if available
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        prompt_parts.append("\nConversation History:")
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("")

    # Add file context if available
    files = state.get("files", [])
    if files:
        prompt_parts.append(f"\n[User has attached {len(files)} file(s) for context]")

    # Add the current user message
    prompt_parts.append(f"\nUser Query: {user_message}")
    prompt_parts.append("\nProvide a helpful, accurate, and comprehensive response:")

    return "\n".join(prompt_parts)
