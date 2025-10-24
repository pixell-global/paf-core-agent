"""Tests for UPEE LangGraph Assembly - Integration Tests."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.langgraph_upee.graph import build_upee_graph, execute_upee_graph, route_execution
from src.langgraph_upee.state import UPEEState, UPEEInput
from src.config.agent_loader import AgentConfig
from src.llm_providers import LLMResponse


@pytest.fixture
def mock_settings():
    """Mock Settings object."""
    settings = Mock()
    settings.default_model = "gpt-4o"
    settings.a2a_timeout = 30
    return settings


@pytest.fixture
def mock_llm_manager():
    """Mock LLM Provider Manager."""
    manager = Mock()
    manager.get_completion = AsyncMock()
    return manager


@pytest.fixture
def vivid_agent():
    """Vivid Commenter agent config."""
    return AgentConfig(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        endpoint="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="grpc",
        description="Reddit marketing agent",
        capabilities=["Find subreddits"],
        example_queries=["find subreddits about AI"]
    )


def test_route_execution_to_core():
    """Test route_execution selects core execution."""
    state: UPEEState = {
        "user_message": "test",
        "request_id": "test",
        "routing_decision": "core",
        "needs_refinement": False
    }

    result = route_execution(state)
    assert result == "execute_core"


def test_route_execution_to_agent(vivid_agent):
    """Test route_execution selects agent execution."""
    state: UPEEState = {
        "user_message": "test",
        "request_id": "test",
        "routing_decision": "agent",
        "selected_agent": vivid_agent,
        "needs_refinement": False
    }

    result = route_execution(state)
    assert result == "execute_agent"


def test_build_upee_graph(mock_settings, mock_llm_manager):
    """Test building the UPEE graph."""
    graph = build_upee_graph(mock_settings, mock_llm_manager)

    # Graph should be compiled and ready
    assert graph is not None


@pytest.mark.asyncio
async def test_execute_upee_graph_core_path(mock_settings, mock_llm_manager):
    """Test complete graph execution via core path (general query)."""
    # Mock LLM responses for understand, routing, core execution, evaluate
    mock_llm_manager.get_completion.side_effect = [
        # Understand node
        LLMResponse(
            content='{"intent_summary": "User wants to know capital of France", "primary_intent": "question", "topics": ["geography"], "entities": [], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": ["capital", "france"]}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Routing node
        LLMResponse(
            content='{"decision": "core", "agent_id": null, "confidence": 0.92, "reasoning": "General knowledge query"}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Execute core node
        LLMResponse(
            content="Paris is the capital of France.",
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Evaluate node
        LLMResponse(
            content='{"quality_score": 0.95, "feedback": "Excellent response", "needs_refinement": false, "suggestions": ""}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        )
    ]

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[]):
        user_input: UPEEInput = {
            "user_message": "What is the capital of France?",
            "request_id": "test-001"
        }

        output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    # Verify output
    assert output["response"] == "Paris is the capital of France."
    assert output["routing_decision"] == "core"
    assert output["selected_agent_name"] is None
    assert output["quality_score"] == 0.95
    assert output["error"] is None


@pytest.mark.asyncio
async def test_execute_upee_graph_agent_path(mock_settings, mock_llm_manager, vivid_agent):
    """Test complete graph execution via agent path (Reddit query)."""
    # Mock LLM responses
    mock_llm_manager.get_completion.side_effect = [
        # Understand node
        LLMResponse(
            content='{"intent_summary": "User wants to find AI subreddits", "primary_intent": "search", "topics": ["reddit", "ai"], "entities": [], "complexity": "simple", "requires_specialized_knowledge": true, "domain": "reddit", "keywords": ["find", "subreddit", "ai"]}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Routing node
        LLMResponse(
            content='{"decision": "agent", "agent_id": "4906eeb7-9959-414e-84c6-f2445822ebe4", "confidence": 0.95, "reasoning": "Reddit query matches Vivid Commenter"}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Evaluate node
        LLMResponse(
            content='{"quality_score": 0.88, "feedback": "Good subreddit list", "needs_refinement": false, "suggestions": ""}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        )
    ]

    # Mock gRPC client for execute_agent node
    mock_grpc_client = Mock()
    mock_grpc_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {
            "content": "Here are 10 subreddits about AI:\n1. r/MachineLearning\n2. r/artificial..."
        }
    })

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_grpc_client):
            user_input: UPEEInput = {
                "user_message": "find me 10 subreddits related to ai",
                "request_id": "test-002"
            }

            output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    # Verify output
    assert "MachineLearning" in output["response"]
    assert output["routing_decision"] == "agent"
    assert output["selected_agent_name"] == "Vivid Commenter"
    assert output["quality_score"] == 0.88
    assert output["error"] is None


@pytest.mark.asyncio
async def test_execute_upee_graph_with_conversation_history(mock_settings, mock_llm_manager):
    """Test graph execution with conversation history."""
    mock_llm_manager.get_completion.side_effect = [
        # Understand
        LLMResponse(
            content='{"intent_summary": "Follow-up question", "primary_intent": "question", "topics": [], "entities": [], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": ["more"]}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Routing
        LLMResponse(
            content='{"decision": "core", "agent_id": null, "confidence": 0.85, "reasoning": "General follow-up"}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Execute core
        LLMResponse(
            content="Paris is beautiful in spring with blooming gardens.",
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Evaluate
        LLMResponse(
            content='{"quality_score": 0.80, "feedback": "Good contextual response", "needs_refinement": false, "suggestions": ""}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        )
    ]

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[]):
        user_input: UPEEInput = {
            "user_message": "Tell me more",
            "request_id": "test-003",
            "conversation_history": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"}
            ]
        }

        output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    assert "Paris" in output["response"]
    assert output["error"] is None


@pytest.mark.asyncio
async def test_execute_upee_graph_error_handling(mock_settings, mock_llm_manager):
    """Test graph handles errors gracefully."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("LLM API error")

    user_input: UPEEInput = {
        "user_message": "test",
        "request_id": "test-004"
    }

    output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    # Should handle error gracefully with fallback mechanisms
    # Graph continues with fallbacks, so response exists (not just error message)
    assert "error" in output["response"].lower() or "apologize" in output["response"].lower()
    # Error is tracked but graph completes
    assert output["routing_decision"] == "core"
    # Quality score is reduced for error responses (fallback evaluation)
    assert output["quality_score"] < 0.5  # Fallback gives lower score for errors
