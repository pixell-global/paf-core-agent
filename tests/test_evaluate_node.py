"""Tests for Evaluate Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from src.langgraph_upee.nodes.evaluate import evaluate_node, _fallback_evaluation
from src.langgraph_upee.state import UPEEState
from src.llm_providers import LLMResponse


@pytest.fixture
def mock_settings():
    """Mock Settings object."""
    settings = Mock()
    settings.default_model = "gpt-4o"
    return settings


@pytest.fixture
def mock_llm_manager():
    """Mock LLM Provider Manager."""
    manager = Mock()
    manager.get_completion = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_evaluate_node_high_quality(mock_settings, mock_llm_manager):
    """Test evaluate node rates high quality response."""
    # Mock LLM to rate highly
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"quality_score": 0.95, "feedback": "Excellent response, complete and accurate", "needs_refinement": false, "suggestions": ""}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "What is the capital of France?",
        "request_id": "test-001",
        "response": "Paris is the capital of France. It has been the capital since 987 AD.",
        "understanding": {
            "intent_summary": "User wants to know the capital of France"
        },
        "routing_decision": "core",
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    # Verify evaluation results
    assert result["quality_score"] == 0.95
    assert "excellent" in result["quality_feedback"].lower()
    assert result["needs_refinement"] is False
    assert "evaluate" in result["execution_path"]
    assert "evaluate_start" in result["timestamps"]
    assert "evaluate_end" in result["timestamps"]


@pytest.mark.asyncio
async def test_evaluate_node_low_quality(mock_settings, mock_llm_manager):
    """Test evaluate node rates low quality response."""
    # Mock LLM to rate poorly
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"quality_score": 0.45, "feedback": "Response is incomplete and lacks detail", "needs_refinement": true, "suggestions": "Provide more comprehensive information"}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "Explain quantum computing",
        "request_id": "test-002",
        "response": "It's complicated.",
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    assert result["quality_score"] == 0.45
    assert result["needs_refinement"] is True
    assert "comprehensive" in result["refinement_suggestions"].lower()


@pytest.mark.asyncio
async def test_evaluate_node_with_markdown_json(mock_settings, mock_llm_manager):
    """Test evaluate node handles JSON in markdown code blocks."""
    # Mock LLM response with markdown
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='```json\n{"quality_score": 0.8, "feedback": "Good response", "needs_refinement": false, "suggestions": ""}\n```',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-003",
        "response": "Test response",
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    assert result["quality_score"] == 0.8
    assert "good" in result["quality_feedback"].lower()


@pytest.mark.asyncio
async def test_evaluate_node_fallback_on_llm_failure(mock_settings, mock_llm_manager):
    """Test evaluate node uses fallback if LLM fails."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("LLM API error")

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-004",
        "response": "This is a reasonable length response with good content.",
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    # Should still have evaluation (from fallback)
    assert "quality_score" in result
    assert result["quality_score"] == 0.65  # Fallback score for short response (55 chars)
    assert result["needs_refinement"] is False


@pytest.mark.asyncio
async def test_evaluate_node_with_agent_response(mock_settings, mock_llm_manager):
    """Test evaluate node evaluates agent response."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"quality_score": 0.88, "feedback": "Agent provided relevant subreddit list", "needs_refinement": false, "suggestions": ""}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    from src.config.agent_loader import AgentConfig

    vivid_agent = AgentConfig(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        endpoint="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="grpc",
        description="Reddit agent",
        capabilities=[],
        example_queries=[]
    )

    state: UPEEState = {
        "user_message": "find me subreddits about AI",
        "request_id": "test-005",
        "response": "Here are 10 subreddits about AI:\n1. r/MachineLearning\n2. r/artificial...",
        "routing_decision": "agent",
        "selected_agent": vivid_agent,
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    assert result["quality_score"] == 0.88
    assert "relevant" in result["quality_feedback"].lower()


def test_fallback_evaluation_very_short_response():
    """Test fallback evaluation for very short response."""
    result = _fallback_evaluation("No.")

    assert result["quality_score"] == 0.4
    assert result["needs_refinement"] is True
    assert "brief" in result["feedback"].lower()


def test_fallback_evaluation_short_response():
    """Test fallback evaluation for short response."""
    result = _fallback_evaluation("Paris is the capital of France.")

    assert result["quality_score"] == 0.65
    assert result["needs_refinement"] is False
    assert "concise" in result["feedback"].lower()


def test_fallback_evaluation_long_response():
    """Test fallback evaluation for long response."""
    long_response = "Paris is the capital of France. " * 70  # Over 2000 chars (2310 chars)

    result = _fallback_evaluation(long_response)

    assert result["quality_score"] == 0.75
    assert result["needs_refinement"] is False
    assert "detailed" in result["feedback"].lower() or "comprehensive" in result["feedback"].lower()


def test_fallback_evaluation_moderate_response():
    """Test fallback evaluation for moderate length response."""
    moderate_response = "Paris is the capital of France. It has been the capital since 987 AD and is located in northern France."

    result = _fallback_evaluation(moderate_response)

    assert result["quality_score"] == 0.7
    assert result["needs_refinement"] is False


def test_fallback_evaluation_error_response():
    """Test fallback evaluation detects error in response."""
    error_response = "I apologize, but I encountered an error processing your request."

    result = _fallback_evaluation(error_response)

    # Score should be reduced
    assert result["quality_score"] < 0.5
    assert result["needs_refinement"] is True
    assert "error" in result["suggestions"].lower()


@pytest.mark.asyncio
async def test_evaluate_node_preserves_state(mock_settings, mock_llm_manager):
    """Test evaluate node preserves other state fields."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"quality_score": 0.85, "feedback": "Good", "needs_refinement": false, "suggestions": ""}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-006",
        "response": "Test response",
        "routing_decision": "core",
        "routing_reasoning": "General query",
        "understanding": {"primary_intent": "question"},
        "show_thinking": True,
        "needs_refinement": False
    }

    result = await evaluate_node(state, mock_settings, mock_llm_manager)

    # Original fields preserved
    assert result["routing_decision"] == "core"
    assert result["routing_reasoning"] == "General query"
    assert result["understanding"]["primary_intent"] == "question"
    assert result["show_thinking"] is True
    # New evaluation fields added
    assert "quality_score" in result
    assert "quality_feedback" in result
    assert "needs_refinement" in result
