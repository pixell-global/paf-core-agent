"""LangGraph state definition for UPEE cognitive loop."""

from typing import TypedDict, Optional, List, Dict, Any
from src.config.agent_loader import AgentConfig


class UPEEState(TypedDict, total=False):
    """State shared across all UPEE graph nodes.

    This state flows through the LangGraph:
    Understand -> AI Routing -> (Execute Core | Execute Agent) -> Evaluate
    """

    # ===== User Input =====
    user_message: str  # Required: The user's query
    files: Optional[List[Dict[str, Any]]]  # Optional file attachments
    conversation_history: Optional[List[Dict[str, Any]]]  # Chat history

    # ===== Understanding Phase =====
    understanding: Optional[Dict[str, Any]]  # Analysis of user intent, context, complexity
    intent_summary: Optional[str]  # Concise summary of what user wants

    # ===== AI Routing Phase (CRITICAL) =====
    routing_decision: Optional[str]  # "core" = handle internally, "agent" = route to A2A agent
    selected_agent: Optional[AgentConfig]  # If routing_decision="agent", which agent to use
    routing_reasoning: Optional[str]  # LLM's explanation of routing decision
    routing_confidence: Optional[float]  # Confidence score 0.0-1.0

    # ===== Execution Phase =====
    response: Optional[str]  # Final response text (from core or agent)
    response_metadata: Optional[Dict[str, Any]]  # Metadata about the response
    agent_response: Optional[Dict[str, Any]]  # Raw A2A agent response if routed

    # ===== Evaluation Phase =====
    quality_score: Optional[float]  # 0.0-1.0 quality assessment
    quality_feedback: Optional[str]  # LLM's quality analysis
    needs_refinement: bool  # Whether to loop back and refine (default: False)
    refinement_suggestions: Optional[str]  # What to improve if needs_refinement=True

    # ===== Request Metadata =====
    request_id: str  # Required: Unique request identifier
    model: Optional[str]  # LLM model to use (e.g., "gpt-4o")
    show_thinking: bool  # Whether to stream thinking events (default: False)
    temperature: Optional[float]  # LLM temperature setting

    # ===== Error Handling =====
    error: Optional[str]  # Error message if something fails
    error_stage: Optional[str]  # Which stage failed (understand/route/execute/evaluate)

    # ===== Performance Tracking =====
    timestamps: Optional[Dict[str, str]]  # ISO timestamps for each phase
    execution_path: Optional[List[str]]  # Track which nodes were executed


class UPEEInput(TypedDict, total=False):
    """Input to start UPEE graph execution."""
    user_message: str  # Required
    request_id: str  # Required
    files: Optional[List[Dict[str, Any]]]
    conversation_history: Optional[List[Dict[str, Any]]]
    model: Optional[str]
    show_thinking: bool
    temperature: Optional[float]


class UPEEOutput(TypedDict, total=False):
    """Output from UPEE graph execution."""
    response: str  # Final response to user
    routing_decision: str  # "core" or "agent"
    selected_agent_name: Optional[str]  # Name of agent if routed
    selected_agent_id: Optional[str]  # Agent app ID if routed
    quality_score: float  # Final quality score
    request_id: str  # Request identifier
    error: Optional[str]  # Error if failed
