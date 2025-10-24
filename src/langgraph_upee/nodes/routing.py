"""AI-Powered Routing Node - Intelligently routes requests to PAF Core or specialized agents.

This is the CRITICAL node that makes intelligent routing decisions using an LLM.
Instead of keyword matching, the LLM analyzes:
- User intent and requirements
- Available agent capabilities
- Best match for the task

The LLM decides:
- "core": Handle with PAF Core's general intelligence
- "agent": Route to specialized A2A agent via gRPC
"""

import json
from datetime import datetime
from typing import Dict, Any, List
import structlog

from src.langgraph_upee.state import UPEEState
from src.config.agent_loader import load_agents, AgentConfig
from src.llm_providers import LLMProviderManager, LLMRequest
from src.settings import Settings

logger = structlog.get_logger()


async def routing_node(state: UPEEState, settings: Settings, llm_manager: LLMProviderManager) -> UPEEState:
    """
    AI-Powered Routing Node: Intelligently decide where to route the request.

    This node uses an LLM to analyze the user's intent against available agent capabilities
    and make an intelligent routing decision.

    Args:
        state: Current UPEE state with understanding
        settings: Application settings
        llm_manager: LLM provider manager for AI routing decision

    Returns:
        Updated state with routing_decision, selected_agent, routing_reasoning
    """
    request_id = state.get("request_id", "unknown")
    user_message = state.get("user_message", "")
    understanding = state.get("understanding", {})

    logger.info(
        "Starting AI-powered routing phase",
        request_id=request_id,
        intent=understanding.get("primary_intent"),
        domain=understanding.get("domain")
    )

    # Track execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("routing")

    # Track timestamps
    timestamps = state.get("timestamps", {})
    timestamps["routing_start"] = datetime.utcnow().isoformat()

    try:
        # Load available agents
        available_agents = load_agents()

        logger.debug(
            "Loaded available agents",
            agent_count=len(available_agents),
            agent_names=[a.name for a in available_agents]
        )

        # Use LLM to make intelligent routing decision
        routing_result = await _ai_routing_decision(
            user_message,
            understanding,
            available_agents,
            llm_manager,
            settings
        )

        timestamps["routing_end"] = datetime.utcnow().isoformat()

        # Update state with routing decision
        state["routing_decision"] = routing_result["decision"]
        state["selected_agent"] = routing_result.get("selected_agent")
        state["routing_reasoning"] = routing_result["reasoning"]
        state["routing_confidence"] = routing_result["confidence"]
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        logger.info(
            "Routing decision completed",
            request_id=request_id,
            decision=routing_result["decision"],
            selected_agent_name=routing_result.get("selected_agent").name if routing_result.get("selected_agent") else None,
            confidence=routing_result["confidence"],
            reasoning=routing_result["reasoning"][:100]  # Log first 100 chars
        )

        return state

    except Exception as e:
        logger.error(
            "Routing phase failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )

        timestamps["routing_end"] = datetime.utcnow().isoformat()

        # Fallback to core if routing fails
        state["routing_decision"] = "core"
        state["selected_agent"] = None
        state["routing_reasoning"] = f"Routing failed, defaulting to core: {str(e)}"
        state["routing_confidence"] = 0.5
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        return state


async def _ai_routing_decision(
    user_message: str,
    understanding: Dict[str, Any],
    available_agents: List[AgentConfig],
    llm_manager: LLMProviderManager,
    settings: Settings
) -> Dict[str, Any]:
    """
    Use LLM to make intelligent routing decision.

    This is the core AI-native routing logic that replaces keyword matching.
    The LLM reads agent capabilities and example queries to understand what each agent does.
    """

    # Build prompt with agent information
    agents_info = _format_agents_for_llm(available_agents)

    prompt = f"""You are an intelligent request router. Analyze the user's request and decide whether to handle it with the core AI system or route it to a specialized agent.

User Message: "{user_message}"

User Intent Analysis:
- Primary Intent: {understanding.get("primary_intent", "unknown")}
- Domain: {understanding.get("domain", "general")}
- Topics: {", ".join(understanding.get("topics", []))}
- Keywords: {", ".join(understanding.get("keywords", [])[:10])}
- Complexity: {understanding.get("complexity", "moderate")}
- Summary: {understanding.get("intent_summary", "")}

Available Specialized Agents:
{agents_info}

Routing Decision Rules:
1. If the user's request CLOSELY MATCHES a specialized agent's capabilities and example queries, route to that agent
2. If the request requires general knowledge or doesn't match any agent well, use core
3. Consider the agent's domain expertise and example queries carefully
4. Be confident in your decision - only route to agent if it's clearly a good match

Respond in JSON format:
{{
    "decision": "core" or "agent",
    "agent_id": "agent_app_id if decision is agent, otherwise null",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this routing decision was made"
}}

Respond ONLY with valid JSON, no additional text."""

    try:
        llm_request = LLMRequest(
            model=settings.default_model,
            prompt=prompt,
            temperature=0.2,  # Low temperature for consistent routing decisions
            max_tokens=300,
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

            routing_data = json.loads(content)

            decision = routing_data.get("decision", "core")
            agent_id = routing_data.get("agent_id")
            confidence = routing_data.get("confidence", 0.5)
            reasoning = routing_data.get("reasoning", "No reasoning provided")

            # Find the selected agent if routing to agent
            selected_agent = None
            if decision == "agent" and agent_id:
                selected_agent = next(
                    (agent for agent in available_agents if agent.agent_app_id == agent_id),
                    None
                )

                if not selected_agent:
                    logger.warning(
                        "LLM selected non-existent agent",
                        agent_id=agent_id
                    )
                    # Fallback to core if agent not found
                    decision = "core"
                    reasoning = f"Selected agent {agent_id} not found, falling back to core"
                    confidence = 0.5

            logger.debug(
                "AI routing decision made",
                decision=decision,
                agent_id=agent_id,
                confidence=confidence
            )

            return {
                "decision": decision,
                "selected_agent": selected_agent,
                "confidence": confidence,
                "reasoning": reasoning
            }

    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse LLM routing response as JSON",
            error=str(e),
            response_content=response.content[:200] if response.content else None
        )
    except Exception as e:
        logger.error(
            "AI routing decision failed",
            error=str(e),
            exc_info=True
        )

    # Fallback to core if LLM routing fails
    return _fallback_routing_decision(user_message, understanding, available_agents)


def _format_agents_for_llm(agents: List[AgentConfig]) -> str:
    """
    Format agent information for LLM prompt.

    Includes agent name, description, capabilities, and example queries.
    """
    if not agents:
        return "No specialized agents available."

    formatted = []
    for agent in agents:
        agent_info = f"""
Agent: {agent.name} (ID: {agent.agent_app_id})
Description: {agent.description}
Capabilities:
{chr(10).join(f"  - {cap}" for cap in agent.capabilities)}
Example Queries:
{chr(10).join(f"  - {query}" for query in agent.example_queries)}
"""
        formatted.append(agent_info.strip())

    return "\n\n".join(formatted)


def _fallback_routing_decision(
    user_message: str,
    understanding: Dict[str, Any],
    available_agents: List[AgentConfig]
) -> Dict[str, Any]:
    """
    Simple rule-based fallback routing if AI routing fails.

    Checks for basic keyword matches in user message against agent capabilities.
    """
    message_lower = user_message.lower()

    # Check each agent for keyword matches
    for agent in available_agents:
        # Check if agent keywords appear in message
        agent_keywords = []
        agent_keywords.extend([cap.lower() for cap in agent.capabilities])
        agent_keywords.extend([query.lower() for query in agent.example_queries])
        agent_keywords.append(agent.name.lower())

        # Check for reddit-specific keywords for Vivid Commenter
        if "reddit" in agent.name.lower() or "reddit" in agent.description.lower():
            reddit_keywords = ["reddit", "subreddit", "r/", "karma", "post", "comment"]
            if any(keyword in message_lower for keyword in reddit_keywords):
                return {
                    "decision": "agent",
                    "selected_agent": agent,
                    "confidence": 0.7,
                    "reasoning": f"Fallback: Detected Reddit-related keywords, routing to {agent.name}"
                }

    # Default to core if no agent matches
    return {
        "decision": "core",
        "selected_agent": None,
        "confidence": 0.6,
        "reasoning": "Fallback: No specialized agent matches, using core system"
    }
