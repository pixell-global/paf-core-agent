"""Tests for Execute Agent Node (A2A gRPC)."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.langgraph_upee.nodes.execute_agent import (
    execute_agent_node,
    _build_a2a_message,
    _extract_response_content
)
from src.langgraph_upee.state import UPEEState
from src.config.agent_loader import AgentConfig


@pytest.fixture
def mock_settings():
    """Mock Settings object."""
    settings = Mock()
    settings.a2a_timeout = 30
    return settings


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


@pytest.mark.asyncio
async def test_execute_agent_success(mock_settings, vivid_agent):
    """Test execute agent node successfully calls agent via gRPC."""
    # Mock gRPC client
    mock_client = Mock()
    mock_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {
            "content": "Here are 10 subreddits about AI:\n1. r/MachineLearning\n2. r/artificial..."
        }
    })

    with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_client):
        state: UPEEState = {
            "user_message": "find me 10 subreddits related to ai",
            "request_id": "test-001",
            "routing_decision": "agent",
            "selected_agent": vivid_agent,
            "needs_refinement": False
        }

        result = await execute_agent_node(state, mock_settings)

    # Verify agent was called
    assert result["response"] == "Here are 10 subreddits about AI:\n1. r/MachineLearning\n2. r/artificial..."
    assert "response_metadata" in result
    assert result["response_metadata"]["agent_name"] == "Vivid Commenter"
    assert result["response_metadata"]["execution_type"] == "agent"
    assert result["response_metadata"]["success"] is True
    assert "execute_agent" in result["execution_path"]
    assert "execute_agent_start" in result["timestamps"]
    assert "execute_agent_end" in result["timestamps"]


@pytest.mark.asyncio
async def test_execute_agent_no_agent_selected(mock_settings):
    """Test execute agent handles missing selected_agent."""
    state: UPEEState = {
        "user_message": "test",
        "request_id": "test-002",
        "routing_decision": "agent",
        "selected_agent": None,  # No agent selected!
        "needs_refinement": False
    }

    result = await execute_agent_node(state, mock_settings)

    # Should have error
    assert "error" in result
    assert "no agent selected" in result["error"].lower()
    assert result["error_stage"] == "execute_agent"


@pytest.mark.asyncio
async def test_execute_agent_grpc_error(mock_settings, vivid_agent):
    """Test execute agent handles gRPC errors gracefully."""
    # Mock gRPC client to raise exception
    mock_client = Mock()
    mock_client.send_message = AsyncMock(side_effect=Exception("gRPC connection timeout"))

    with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_client):
        state: UPEEState = {
            "user_message": "test",
            "request_id": "test-003",
            "selected_agent": vivid_agent,
            "needs_refinement": False
        }

        result = await execute_agent_node(state, mock_settings)

    # Should have error in state
    assert "error" in result
    assert "timeout" in result["error"].lower()
    assert result["error_stage"] == "execute_agent"
    assert "apologize" in result["response"].lower()
    assert "execute_agent" in result["execution_path"]


@pytest.mark.asyncio
async def test_execute_agent_with_conversation_history(mock_settings, vivid_agent):
    """Test execute agent includes conversation history in A2A message."""
    mock_client = Mock()
    mock_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {"content": "Based on our previous discussion..."}
    })

    with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_client):
        state: UPEEState = {
            "user_message": "tell me more",
            "request_id": "test-004",
            "selected_agent": vivid_agent,
            "conversation_history": [
                {"role": "user", "content": "find subreddits"},
                {"role": "assistant", "content": "Here are some..."}
            ],
            "needs_refinement": False
        }

        result = await execute_agent_node(state, mock_settings)

    assert result["response"] == "Based on our previous discussion..."
    # Verify conversation history was passed
    call_args = mock_client.send_message.call_args[0][0]
    assert "conversation_history" in call_args["parameters"]


@pytest.mark.asyncio
async def test_execute_agent_with_files(mock_settings, vivid_agent):
    """Test execute agent includes files in A2A message."""
    mock_client = Mock()
    mock_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {"content": "I analyzed the file..."}
    })

    with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_client):
        state: UPEEState = {
            "user_message": "analyze this",
            "request_id": "test-005",
            "selected_agent": vivid_agent,
            "files": [{"file_name": "data.csv", "file_size": 1024}],
            "needs_refinement": False
        }

        result = await execute_agent_node(state, mock_settings)

    assert result["response"] == "I analyzed the file..."
    # Verify files were passed
    call_args = mock_client.send_message.call_args[0][0]
    assert "files" in call_args["parameters"]


def test_build_a2a_message_basic(vivid_agent):
    """Test building basic A2A message."""
    state: UPEEState = {
        "user_message": "find subreddits about AI",
        "request_id": "test-006",
        "routing_reasoning": "User wants Reddit content",
        "routing_confidence": 0.95,
        "needs_refinement": False
    }

    message = _build_a2a_message("find subreddits about AI", state)

    assert message["type"] == "chat"
    assert message["skill_id"] == "chat"
    assert message["parameters"]["message"] == "find subreddits about AI"
    assert message["parameters"]["request_id"] == "test-006"
    assert message["metadata"]["routing_source"] == "langgraph_ai_routing"
    assert message["metadata"]["routing_reasoning"] == "User wants Reddit content"
    assert message["metadata"]["routing_confidence"] == 0.95


def test_build_a2a_message_with_history():
    """Test building A2A message with conversation history."""
    state: UPEEState = {
        "user_message": "more",
        "request_id": "test-007",
        "conversation_history": [
            {"role": "user", "content": "hi"}
        ],
        "needs_refinement": False
    }

    message = _build_a2a_message("more", state)

    assert "conversation_history" in message["parameters"]
    assert len(message["parameters"]["conversation_history"]) == 1


def test_extract_response_content_data_string():
    """Test extracting response when data is a string."""
    response = {
        "success": True,
        "data": "This is the response text"
    }

    content = _extract_response_content(response)
    assert content == "This is the response text"


def test_extract_response_content_data_dict_with_content():
    """Test extracting response when data is dict with content field."""
    response = {
        "success": True,
        "data": {
            "content": "Response content here"
        }
    }

    content = _extract_response_content(response)
    assert content == "Response content here"


def test_extract_response_content_data_dict_with_message():
    """Test extracting response when data is dict with message field."""
    response = {
        "success": True,
        "data": {
            "message": "Response message here"
        }
    }

    content = _extract_response_content(response)
    assert content == "Response message here"


def test_extract_response_content_result_field():
    """Test extracting response from result field."""
    response = {
        "success": True,
        "result": "Result text here"
    }

    content = _extract_response_content(response)
    assert content == "Result text here"


def test_extract_response_content_message_field():
    """Test extracting response from message field."""
    response = {
        "success": True,
        "message": "Direct message here"
    }

    content = _extract_response_content(response)
    assert content == "Direct message here"


def test_extract_response_content_fallback_json():
    """Test extracting response falls back to JSON dump."""
    response = {
        "success": True,
        "some_field": "some value"
    }

    content = _extract_response_content(response)
    # Should be JSON formatted
    assert "success" in content
    assert "some_field" in content


@pytest.mark.asyncio
async def test_execute_agent_preserves_state(mock_settings, vivid_agent):
    """Test execute agent preserves other state fields."""
    mock_client = Mock()
    mock_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {"content": "Response"}
    })

    with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_client):
        state: UPEEState = {
            "user_message": "test",
            "request_id": "test-008",
            "selected_agent": vivid_agent,
            "routing_decision": "agent",
            "routing_reasoning": "Reddit query",
            "understanding": {"primary_intent": "search"},
            "show_thinking": True,
            "needs_refinement": False
        }

        result = await execute_agent_node(state, mock_settings)

    # Original fields preserved
    assert result["routing_decision"] == "agent"
    assert result["routing_reasoning"] == "Reddit query"
    assert result["understanding"]["primary_intent"] == "search"
    assert result["show_thinking"] is True
    # New fields added
    assert "response" in result
    assert "agent_response" in result
    assert "response_metadata" in result
