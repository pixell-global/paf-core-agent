"""Execute Agent Node - Route requests to specialized A2A agents via gRPC.

This node executes when routing_decision == "agent".
Makes gRPC A2A calls to specialized agents like Vivid Commenter.
"""

from datetime import datetime
from typing import Dict, Any
import structlog

from src.langgraph_upee.state import UPEEState
from src.agents.grpc_a2a_client import GrpcA2AClient
from src.settings import Settings

logger = structlog.get_logger()


async def execute_agent_node(state: UPEEState, settings: Settings) -> UPEEState:
    """
    Execute Agent Node: Route request to specialized A2A agent via gRPC.

    This node is called when the routing decision is "agent" - meaning the
    request should be handled by a specialized agent like Vivid Commenter.

    Args:
        state: Current UPEE state with selected_agent
        settings: Application settings

    Returns:
        Updated state with agent response
    """
    request_id = state.get("request_id", "unknown")
    user_message = state.get("user_message", "")
    selected_agent = state.get("selected_agent")

    if not selected_agent:
        logger.error(
            "Execute agent called but no agent selected",
            request_id=request_id
        )
        # Fallback error
        state["error"] = "No agent selected for execution"
        state["error_stage"] = "execute_agent"
        state["response"] = "I apologize, but I couldn't route your request to the appropriate agent."
        return state

    logger.info(
        "Executing with specialized agent via gRPC A2A",
        request_id=request_id,
        agent_name=selected_agent.name,
        agent_id=selected_agent.agent_app_id,
        endpoint=selected_agent.endpoint
    )

    # Track execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("execute_agent")

    # Track timestamps
    timestamps = state.get("timestamps", {})
    timestamps["execute_agent_start"] = datetime.utcnow().isoformat()

    try:
        # Create gRPC A2A client for this agent
        client = GrpcA2AClient(
            agent_app_id=selected_agent.agent_app_id,
            endpoint_url=selected_agent.endpoint,
            timeout=settings.a2a_timeout
        )

        # Build A2A message payload
        message_payload = _build_a2a_message(user_message, state)

        # Make gRPC call to agent
        response = await client.send_message(message_payload)

        timestamps["execute_agent_end"] = datetime.utcnow().isoformat()

        # Extract response content
        response_content = _extract_response_content(response)

        # Update state with agent response
        state["response"] = response_content
        state["agent_response"] = response  # Store full response for debugging
        state["response_metadata"] = {
            "agent_name": selected_agent.name,
            "agent_id": selected_agent.agent_app_id,
            "execution_type": "agent",
            "protocol": selected_agent.protocol,
            "success": response.get("success", False)
        }
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        logger.info(
            "Agent execution completed",
            request_id=request_id,
            agent_name=selected_agent.name,
            response_length=len(response_content),
            success=response.get("success", False)
        )

        return state

    except Exception as e:
        logger.error(
            "Agent execution failed",
            request_id=request_id,
            agent_name=selected_agent.name,
            agent_id=selected_agent.agent_app_id,
            error=str(e),
            exc_info=True
        )

        timestamps["execute_agent_end"] = datetime.utcnow().isoformat()

        # Update state with error
        state["error"] = f"Agent execution failed: {str(e)}"
        state["error_stage"] = "execute_agent"
        state["response"] = f"I apologize, but I encountered an error communicating with the {selected_agent.name} agent: {str(e)}"
        state["timestamps"] = timestamps
        state["execution_path"] = execution_path

        return state


def _build_a2a_message(user_message: str, state: UPEEState) -> Dict[str, Any]:
    """
    Build A2A message payload for agent invocation.

    This creates the message format expected by A2A agents.
    """
    # Build parameters with user message
    parameters = {
        "message": user_message,
        "request_id": state.get("request_id", "unknown")
    }

    # Add conversation history if available
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        parameters["conversation_history"] = conversation_history

    # Add file context if available
    files = state.get("files", [])
    if files:
        parameters["files"] = files

    # Build A2A message
    message = {
        "type": "chat",  # Message type
        "skill_id": "chat",  # Most agents use "chat" skill
        "parameters": parameters,
        "metadata": {
            "routing_source": "langgraph_ai_routing",
            "routing_reasoning": state.get("routing_reasoning", ""),
            "routing_confidence": state.get("routing_confidence", 0.0)
        }
    }

    return message


def _extract_response_content(response: Dict[str, Any]) -> str:
    """
    Extract response content from A2A response.

    A2A responses can have various formats. This extracts the text content.
    """
    # Try various response formats
    if "data" in response:
        data = response["data"]

        # Check for direct content
        if isinstance(data, str):
            return data

        # Check for structured data with content field
        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"])
            if "message" in data:
                return str(data["message"])
            if "response" in data:
                return str(data["response"])
            if "text" in data:
                return str(data["text"])

            # Try to convert dict to readable format
            import json
            return json.dumps(data, indent=2)

    # Check for direct message field
    if "message" in response:
        return str(response["message"])

    # Check for result field
    if "result" in response:
        result = response["result"]
        if isinstance(result, str):
            return result
        import json
        return json.dumps(result, indent=2)

    # Fallback: return the whole response as JSON
    import json
    return json.dumps(response, indent=2)
