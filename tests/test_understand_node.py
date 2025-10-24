"""Tests for Understand Node."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.langgraph_upee.nodes.understand import understand_node, _fallback_intent_analysis
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
async def test_understand_node_success(mock_settings, mock_llm_manager):
    """Test understand node with successful LLM analysis."""
    # Mock LLM response
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"intent_summary": "User wants to find AI-related subreddits", "primary_intent": "search", "topics": ["reddit", "ai"], "entities": ["subreddit"], "complexity": "simple", "requires_specialized_knowledge": true, "domain": "reddit", "keywords": ["find", "subreddit", "ai"]}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    # Initial state
    state: UPEEState = {
        "user_message": "find me 10 subreddits related to ai",
        "request_id": "test-001",
        "needs_refinement": False
    }

    # Run understand node
    result = await understand_node(state, mock_settings, mock_llm_manager)

    # Verify state updated correctly
    assert "understanding" in result
    assert result["understanding"]["primary_intent"] == "search"
    assert "reddit" in result["understanding"]["topics"]
    assert "ai" in result["understanding"]["topics"]
    assert result["understanding"]["domain"] == "reddit"
    assert result["intent_summary"] == "User wants to find AI-related subreddits"
    assert "understand" in result["execution_path"]
    assert "understand_start" in result["timestamps"]
    assert "understand_end" in result["timestamps"]


@pytest.mark.asyncio
async def test_understand_node_with_json_in_code_block(mock_settings, mock_llm_manager):
    """Test understand node handles JSON wrapped in markdown code blocks."""
    # Mock LLM response with markdown
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='```json\n{"intent_summary": "General question", "primary_intent": "question", "topics": ["general"], "entities": [], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": ["what", "capital", "france"]}\n```',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "What is the capital of France?",
        "request_id": "test-002",
        "needs_refinement": False
    }

    result = await understand_node(state, mock_settings, mock_llm_manager)

    assert "understanding" in result
    assert result["understanding"]["primary_intent"] == "question"
    assert result["understanding"]["domain"] == "general"


@pytest.mark.asyncio
async def test_understand_node_fallback_on_llm_failure(mock_settings, mock_llm_manager):
    """Test understand node falls back to rule-based analysis if LLM fails."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("LLM API error")

    state: UPEEState = {
        "user_message": "find subreddits about machine learning",
        "request_id": "test-003",
        "needs_refinement": False
    }

    result = await understand_node(state, mock_settings, mock_llm_manager)

    # Should still have understanding (from fallback)
    assert "understanding" in result
    assert result["understanding"]["primary_intent"] == "search"
    assert "reddit" in result["understanding"]["topics"]


@pytest.mark.asyncio
async def test_understand_node_with_conversation_history(mock_settings, mock_llm_manager):
    """Test understand node with conversation history."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"intent_summary": "Follow-up question", "primary_intent": "question", "topics": [], "entities": [], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": ["tell", "more"]}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "Tell me more",
        "request_id": "test-004",
        "conversation_history": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
        ],
        "needs_refinement": False
    }

    result = await understand_node(state, mock_settings, mock_llm_manager)

    assert "understanding" in result
    assert result["understanding"]["primary_intent"] == "question"


def test_fallback_intent_analysis_search():
    """Test fallback analysis detects search intent."""
    result = _fallback_intent_analysis("find me subreddits about ai")

    assert result["primary_intent"] == "search"
    assert "reddit" in result["topics"]
    assert "ai" in result["topics"]


def test_fallback_intent_analysis_question():
    """Test fallback analysis detects question intent."""
    result = _fallback_intent_analysis("What is the capital of France?")

    assert result["primary_intent"] == "question"


def test_fallback_intent_analysis_creative():
    """Test fallback analysis detects creative intent."""
    result = _fallback_intent_analysis("create a blog post about AI")

    assert result["primary_intent"] == "creative"


def test_fallback_intent_analysis_general():
    """Test fallback analysis defaults to general for unclear messages."""
    result = _fallback_intent_analysis("Hello there")

    assert result["primary_intent"] == "general"
    assert result["complexity"] == "moderate"
    assert result["domain"] == "general"


@pytest.mark.asyncio
async def test_understand_node_error_handling(mock_settings, mock_llm_manager):
    """Test understand node handles errors gracefully."""
    # Mock LLM to raise exception and ensure fallback also fails
    mock_llm_manager.get_completion.side_effect = Exception("Critical error")

    state: UPEEState = {
        "user_message": "test message",
        "request_id": "test-005",
        "needs_refinement": False
    }

    # Patch the fallback to also raise an error
    with patch('src.langgraph_upee.nodes.understand._fallback_intent_analysis', side_effect=Exception("Fallback failed")):
        result = await understand_node(state, mock_settings, mock_llm_manager)

    # Should have error in state
    assert "error" in result
    assert "Understanding failed" in result["error"]
    assert result["error_stage"] == "understand"
    assert "understand" in result["execution_path"]


@pytest.mark.asyncio
async def test_understand_node_preserves_existing_state(mock_settings, mock_llm_manager):
    """Test understand node preserves other state fields."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"intent_summary": "Test", "primary_intent": "general", "topics": [], "entities": [], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": []}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-006",
        "model": "gpt-4o",
        "temperature": 0.7,
        "show_thinking": True,
        "files": [{"file_name": "test.txt"}],
        "needs_refinement": False
    }

    result = await understand_node(state, mock_settings, mock_llm_manager)

    # Original fields should be preserved
    assert result["model"] == "gpt-4o"
    assert result["temperature"] == 0.7
    assert result["show_thinking"] is True
    assert result["files"][0]["file_name"] == "test.txt"
    # New fields should be added
    assert "understanding" in result
    assert "intent_summary" in result
