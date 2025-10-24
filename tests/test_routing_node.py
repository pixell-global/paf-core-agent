"""Tests for AI-Powered Routing Node - CRITICAL COMPONENT."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.langgraph_upee.nodes.routing import routing_node, _format_agents_for_llm, _fallback_routing_decision
from src.langgraph_upee.state import UPEEState
from src.config.agent_loader import AgentConfig
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


@pytest.fixture
def vivid_agent():
    """Vivid Commenter agent config for testing."""
    return AgentConfig(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        endpoint="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="grpc",
        description="Expert Reddit marketing agent",
        capabilities=[
            "Find and analyze relevant subreddits",
            "Research Reddit communities"
        ],
        example_queries=[
            "find subreddits about AI",
            "what are good subreddits for marketing?"
        ]
    )


@pytest.mark.asyncio
async def test_routing_to_agent_reddit_query(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing Reddit query to Vivid Commenter agent."""
    # Mock LLM to route to agent
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"decision": "agent", "agent_id": "4906eeb7-9959-414e-84c6-f2445822ebe4", "confidence": 0.95, "reasoning": "User wants to find Reddit subreddits, which matches Vivid Commenter capabilities"}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    # Mock load_agents to return Vivid agent
    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "find me 10 subreddits related to ai",
            "request_id": "test-001",
            "understanding": {
                "primary_intent": "search",
                "domain": "reddit",
                "topics": ["reddit", "ai"],
                "keywords": ["find", "subreddit", "ai"],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Verify routing to agent
    assert result["routing_decision"] == "agent"
    assert result["selected_agent"] is not None
    assert result["selected_agent"].name == "Vivid Commenter"
    assert result["routing_confidence"] == 0.95
    assert "Vivid Commenter" in result["routing_reasoning"]
    assert "routing" in result["execution_path"]
    assert "routing_start" in result["timestamps"]
    assert "routing_end" in result["timestamps"]


@pytest.mark.asyncio
async def test_routing_to_core_general_query(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing general query to core."""
    # Mock LLM to route to core
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"decision": "core", "agent_id": null, "confidence": 0.92, "reasoning": "General knowledge question best handled by core system"}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "What is the capital of France?",
            "request_id": "test-002",
            "understanding": {
                "primary_intent": "question",
                "domain": "general",
                "topics": ["geography"],
                "keywords": ["capital", "france"],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Verify routing to core
    assert result["routing_decision"] == "core"
    assert result["selected_agent"] is None
    assert result["routing_confidence"] == 0.92
    assert "core" in result["routing_reasoning"].lower()


@pytest.mark.asyncio
async def test_routing_with_markdown_json(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing handles JSON wrapped in markdown code blocks."""
    # Mock LLM response with markdown
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='```json\n{"decision": "core", "agent_id": null, "confidence": 0.85, "reasoning": "General query"}\n```',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "Tell me about AI",
            "request_id": "test-003",
            "understanding": {
                "primary_intent": "question",
                "domain": "general",
                "topics": ["ai"],
                "keywords": ["ai"],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    assert result["routing_decision"] == "core"
    assert result["routing_confidence"] == 0.85


@pytest.mark.asyncio
async def test_routing_llm_selects_nonexistent_agent(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing falls back to core if LLM selects non-existent agent."""
    # Mock LLM to select wrong agent ID
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"decision": "agent", "agent_id": "non-existent-id", "confidence": 0.9, "reasoning": "Route to agent"}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "test message",
            "request_id": "test-004",
            "understanding": {
                "primary_intent": "general",
                "domain": "general",
                "topics": [],
                "keywords": [],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Should fall back to core
    assert result["routing_decision"] == "core"
    assert result["selected_agent"] is None
    assert "not found" in result["routing_reasoning"].lower()


@pytest.mark.asyncio
async def test_routing_fallback_on_llm_failure(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing uses fallback logic if LLM fails."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("LLM API error")

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "find reddit communities about machine learning",
            "request_id": "test-005",
            "understanding": {
                "primary_intent": "search",
                "domain": "reddit",
                "topics": ["reddit"],
                "keywords": ["find", "reddit"],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Fallback should detect "reddit" keyword and route to agent
    assert result["routing_decision"] == "agent"
    assert result["selected_agent"].name == "Vivid Commenter"
    assert "fallback" in result["routing_reasoning"].lower()


@pytest.mark.asyncio
async def test_routing_fallback_to_core_no_match(mock_settings, mock_llm_manager, vivid_agent):
    """Test fallback routing defaults to core if no agent matches."""
    # Mock LLM to raise exception
    mock_llm_manager.get_completion.side_effect = Exception("LLM API error")

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "What is 2+2?",
            "request_id": "test-006",
            "understanding": {
                "primary_intent": "question",
                "domain": "math",
                "topics": [],
                "keywords": ["math"],
                "complexity": "simple"
            },
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Should fallback to core (no reddit keywords)
    assert result["routing_decision"] == "core"
    assert result["selected_agent"] is None


def test_format_agents_for_llm(vivid_agent):
    """Test formatting agents for LLM prompt."""
    formatted = _format_agents_for_llm([vivid_agent])

    assert "Vivid Commenter" in formatted
    assert vivid_agent.agent_app_id in formatted
    assert "Expert Reddit marketing agent" in formatted
    assert "Find and analyze relevant subreddits" in formatted
    assert "find subreddits about AI" in formatted


def test_format_agents_empty_list():
    """Test formatting empty agent list."""
    formatted = _format_agents_for_llm([])
    assert "No specialized agents available" in formatted


def test_fallback_routing_reddit_keywords(vivid_agent):
    """Test fallback routing detects Reddit keywords."""
    understanding = {
        "primary_intent": "search",
        "domain": "reddit",
        "topics": ["reddit"]
    }

    result = _fallback_routing_decision(
        "find me subreddits about AI",
        understanding,
        [vivid_agent]
    )

    assert result["decision"] == "agent"
    assert result["selected_agent"].name == "Vivid Commenter"
    assert "Reddit" in result["reasoning"]


def test_fallback_routing_no_match(vivid_agent):
    """Test fallback routing defaults to core."""
    understanding = {
        "primary_intent": "question",
        "domain": "general",
        "topics": []
    }

    result = _fallback_routing_decision(
        "What is the weather?",
        understanding,
        [vivid_agent]
    )

    assert result["decision"] == "core"
    assert result["selected_agent"] is None


@pytest.mark.asyncio
async def test_routing_preserves_existing_state(mock_settings, mock_llm_manager, vivid_agent):
    """Test routing node preserves other state fields."""
    mock_llm_manager.get_completion.return_value = LLMResponse(
        content='{"decision": "core", "agent_id": null, "confidence": 0.8, "reasoning": "Core handles this"}',
        model="gpt-4o",
        provider="openai",
        finish_reason="stop"
    )

    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        state: UPEEState = {
            "user_message": "test",
            "request_id": "test-007",
            "understanding": {
                "primary_intent": "general",
                "domain": "general"
            },
            "model": "gpt-4o",
            "temperature": 0.7,
            "intent_summary": "Test query",
            "needs_refinement": False
        }

        result = await routing_node(state, mock_settings, mock_llm_manager)

    # Original fields preserved
    assert result["model"] == "gpt-4o"
    assert result["temperature"] == 0.7
    assert result["intent_summary"] == "Test query"
    # New routing fields added
    assert "routing_decision" in result
    assert "routing_reasoning" in result
    assert "routing_confidence" in result
