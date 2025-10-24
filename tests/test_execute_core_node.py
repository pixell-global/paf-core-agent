"""Tests for Execute PAF Core Node."""

import pytest
from unittest.mock import AsyncMock, Mock
from src.langgraph_upee.nodes.execute_core import execute_core_node, _build_core_prompt
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
async def test_execute_core_success(mock_settings, mock_llm_manager):
    """Test execute core node generates response successfully."""
    # Mock LLM response
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="Paris is the capital of France.",
        model="gpt-4o",
        provider="openai",
        finish_reason="stop",
        token_count=10
    )

    state: UPEEState = {
        "user_message": "What is the capital of France?",
        "request_id": "test-001",
        "routing_decision": "core",
        "understanding": {
            "intent_summary": "User wants to know the capital of France"
        },
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    # Verify response generated
    assert result["response"] == "Paris is the capital of France."
    assert "response_metadata" in result
    assert result["response_metadata"]["model"] == "gpt-4o"
    assert result["response_metadata"]["provider"] == "openai"
    assert result["response_metadata"]["execution_type"] == "core"
    assert "execute_core" in result["execution_path"]
    assert "execute_core_start" in result["timestamps"]
    assert "execute_core_end" in result["timestamps"]


@pytest.mark.asyncio
async def test_execute_core_with_custom_model(mock_settings, mock_llm_manager):
    """Test execute core uses custom model from state."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="Test response",
        model="gpt-3.5-turbo",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-002",
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    # Verify custom settings used
    assert result["response"] == "Test response"
    # Check that get_completion was called with correct params
    call_args = mock_llm_manager.get_completion.call_args[0][0]
    assert call_args.model == "gpt-3.5-turbo"
    assert call_args.temperature == 0.5


@pytest.mark.asyncio
async def test_execute_core_with_conversation_history(mock_settings, mock_llm_manager):
    """Test execute core includes conversation history in prompt."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="Paris is beautiful in spring.",
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "Tell me more",
        "request_id": "test-003",
        "conversation_history": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."}
        ],
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    assert result["response"] == "Paris is beautiful in spring."
    # Verify conversation history was included in prompt
    call_args = mock_llm_manager.get_completion.call_args[0][0]
    assert "Conversation History" in call_args.prompt


@pytest.mark.asyncio
async def test_execute_core_with_files(mock_settings, mock_llm_manager):
    """Test execute core mentions file context in prompt."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="Based on the uploaded file...",
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "Analyze this data",
        "request_id": "test-004",
        "files": [
            {"file_name": "data.csv", "file_size": 1024}
        ],
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    assert result["response"] == "Based on the uploaded file..."
    # Verify file context mentioned in prompt
    call_args = mock_llm_manager.get_completion.call_args[0][0]
    assert "attached" in call_args.prompt.lower()
    assert "file" in call_args.prompt.lower()


@pytest.mark.asyncio
async def test_execute_core_error_handling(mock_settings, mock_llm_manager):
    """Test execute core handles LLM errors gracefully."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("API timeout")

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-005",
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    # Should have error in state
    assert "error" in result
    assert "API timeout" in result["error"]
    assert result["error_stage"] == "execute_core"
    assert "apologize" in result["response"].lower()
    assert "execute_core" in result["execution_path"]


@pytest.mark.asyncio
async def test_execute_core_empty_response(mock_settings, mock_llm_manager):
    """Test execute core handles empty LLM response."""
    # Mock empty response
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="",
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-006",
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    # Should have error for empty response
    assert "error" in result
    assert "empty response" in result["error"].lower()


def test_build_core_prompt_basic():
    """Test building basic prompt without extras."""
    state: UPEEState = {
        "user_message": "Hello",
        "request_id": "test",
        "needs_refinement": False
    }

    prompt = _build_core_prompt("Hello", state)

    assert "Hello" in prompt
    assert "User Query" in prompt


def test_build_core_prompt_with_intent():
    """Test building prompt with intent summary."""
    state: UPEEState = {
        "user_message": "test",
        "request_id": "test",
        "understanding": {
            "intent_summary": "User wants to test the system"
        },
        "needs_refinement": False
    }

    prompt = _build_core_prompt("test", state)

    assert "User Intent" in prompt
    assert "test the system" in prompt


def test_build_core_prompt_with_history():
    """Test building prompt with conversation history."""
    state: UPEEState = {
        "user_message": "more info",
        "request_id": "test",
        "conversation_history": [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"}
        ],
        "needs_refinement": False
    }

    prompt = _build_core_prompt("more info", state)

    assert "Conversation History" in prompt
    assert "Question" in prompt
    assert "Answer" in prompt


def test_build_core_prompt_with_files():
    """Test building prompt with file attachments."""
    state: UPEEState = {
        "user_message": "analyze",
        "request_id": "test",
        "files": [
            {"file_name": "file1.txt"},
            {"file_name": "file2.csv"}
        ],
        "needs_refinement": False
    }

    prompt = _build_core_prompt("analyze", state)

    assert "attached" in prompt.lower()
    assert "2" in prompt or "file" in prompt.lower()


@pytest.mark.asyncio
async def test_execute_core_preserves_state(mock_settings, mock_llm_manager):
    """Test execute core preserves other state fields."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content="Response",
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-007",
        "routing_decision": "core",
        "routing_reasoning": "General query",
        "understanding": {"primary_intent": "question"},
        "show_thinking": True,
        "needs_refinement": False
    }

    result = await execute_core_node(state, mock_settings, mock_llm_manager)

    # Original fields preserved
    assert result["routing_decision"] == "core"
    assert result["routing_reasoning"] == "General query"
    assert result["understanding"]["primary_intent"] == "question"
    assert result["show_thinking"] is True
    # New fields added
    assert "response" in result
    assert "response_metadata" in result
