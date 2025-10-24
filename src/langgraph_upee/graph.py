"""LangGraph UPEE Graph Assembly - Wires all nodes together.

This module assembles the complete UPEE cognitive loop using LangGraph:
Understand → Routing → (Core | Agent) → Evaluate

The graph uses conditional edges to route based on AI routing decisions.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
import structlog

from src.langgraph_upee.state import UPEEState, UPEEInput, UPEEOutput
from src.langgraph_upee.nodes.understand import understand_node
from src.langgraph_upee.nodes.routing import routing_node
from src.langgraph_upee.nodes.execute_core import execute_core_node
from src.langgraph_upee.nodes.execute_agent import execute_agent_node
from src.langgraph_upee.nodes.evaluate import evaluate_node
from src.llm_providers import LLMProviderManager
from src.settings import Settings

logger = structlog.get_logger()


def route_execution(state: UPEEState) -> Literal["execute_core", "execute_agent"]:
    """
    Conditional edge function: Route to core or agent based on routing decision.

    This is where the AI routing decision gets acted upon.
    """
    routing_decision = state.get("routing_decision", "core")

    if routing_decision == "agent":
        logger.debug("Routing to agent execution", agent_name=state.get("selected_agent").name if state.get("selected_agent") else None)
        return "execute_agent"
    else:
        logger.debug("Routing to core execution")
        return "execute_core"


def build_upee_graph(settings: Settings, llm_manager: LLMProviderManager) -> StateGraph:
    """
    Build the complete UPEE LangGraph.

    Graph structure:
    START → understand → routing → [core|agent] → evaluate → END

    Args:
        settings: Application settings
        llm_manager: LLM provider manager

    Returns:
        Compiled LangGraph ready for execution
    """
    logger.info("Building UPEE LangGraph")

    # Create state graph
    graph = StateGraph(UPEEState)

    # Define async wrapper functions for nodes
    async def understand_wrapper(state):
        return await understand_node(state, settings, llm_manager)

    async def routing_wrapper(state):
        return await routing_node(state, settings, llm_manager)

    async def execute_core_wrapper(state):
        return await execute_core_node(state, settings, llm_manager)

    async def execute_agent_wrapper(state):
        return await execute_agent_node(state, settings)

    async def evaluate_wrapper(state):
        return await evaluate_node(state, settings, llm_manager)

    # Add nodes
    graph.add_node("understand", understand_wrapper)
    graph.add_node("routing", routing_wrapper)
    graph.add_node("execute_core", execute_core_wrapper)
    graph.add_node("execute_agent", execute_agent_wrapper)
    graph.add_node("evaluate", evaluate_wrapper)

    # Define edges
    # START → understand
    graph.set_entry_point("understand")

    # understand → routing
    graph.add_edge("understand", "routing")

    # routing → conditional routing based on decision
    graph.add_conditional_edges(
        "routing",
        route_execution,
        {
            "execute_core": "execute_core",
            "execute_agent": "execute_agent"
        }
    )

    # execute_core → evaluate
    graph.add_edge("execute_core", "evaluate")

    # execute_agent → evaluate
    graph.add_edge("execute_agent", "evaluate")

    # evaluate → END
    graph.add_edge("evaluate", END)

    # Compile graph
    compiled_graph = graph.compile()

    logger.info("UPEE LangGraph built successfully")

    return compiled_graph


async def execute_upee_graph(
    user_input: UPEEInput,
    settings: Settings,
    llm_manager: LLMProviderManager
) -> UPEEOutput:
    """
    Execute the UPEE graph with user input.

    This is the main entry point for running the complete UPEE loop.

    Args:
        user_input: User input with message, request_id, etc.
        settings: Application settings
        llm_manager: LLM provider manager

    Returns:
        UPEEOutput with final response and metadata
    """
    request_id = user_input.get("request_id", "unknown")

    logger.info(
        "Starting UPEE graph execution",
        request_id=request_id,
        user_message=user_input.get("user_message", "")[:100]
    )

    try:
        # Build graph
        graph = build_upee_graph(settings, llm_manager)

        # Initialize state from input
        initial_state: UPEEState = {
            "user_message": user_input["user_message"],
            "request_id": user_input["request_id"],
            "files": user_input.get("files"),
            "conversation_history": user_input.get("conversation_history"),
            "model": user_input.get("model"),
            "show_thinking": user_input.get("show_thinking", False),
            "temperature": user_input.get("temperature"),
            "needs_refinement": False,
            "execution_path": [],
            "timestamps": {}
        }

        # Execute graph
        final_state = await graph.ainvoke(initial_state)

        # Build output
        output: UPEEOutput = {
            "response": final_state.get("response", ""),
            "routing_decision": final_state.get("routing_decision", "core"),
            "selected_agent_name": final_state.get("selected_agent").name if final_state.get("selected_agent") else None,
            "quality_score": final_state.get("quality_score", 0.0),
            "request_id": request_id,
            "error": final_state.get("error")
        }

        logger.info(
            "UPEE graph execution completed",
            request_id=request_id,
            routing=output["routing_decision"],
            agent=output["selected_agent_name"],
            quality=output["quality_score"]
        )

        return output

    except Exception as e:
        logger.error(
            "UPEE graph execution failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )

        # Return error output
        return {
            "response": f"I apologize, but I encountered an error: {str(e)}",
            "routing_decision": "core",
            "selected_agent_name": None,
            "quality_score": 0.0,
            "request_id": request_id,
            "error": str(e)
        }
