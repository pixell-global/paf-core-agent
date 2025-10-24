"""Integration tests for full A2A routing flow from par_adapter to agent selection."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from src import par_adapter
from src.agents.agent_app_registry import AgentAppRegistry, AgentAppInfo
from src.agents.agent_app_selector import AgentSelectionResult


@pytest.fixture
def mock_vivid_agent_info():
    """Mock Vivid Commenter agent info."""
    return AgentAppInfo(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        description="AI-powered Reddit commenter",
        endpoint_url="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="grpc",
        enabled=True,
        priority=10,
        agent_card={
            "skills": [
                {
                    "id": "find-subreddits",
                    "name": "Find Subreddits",
                    "description": "Find relevant subreddits for a topic"
                }
            ]
        }
    )


@pytest.fixture
def mock_registry_with_vivid(mock_vivid_agent_info):
    """Mock registry populated with Vivid Commenter."""
    registry = MagicMock(spec=AgentAppRegistry)
    registry.agents = {
        mock_vivid_agent_info.agent_app_id: mock_vivid_agent_info
    }
    registry.get_agent = Mock(return_value=mock_vivid_agent_info)
    registry.get_enabled_agents = Mock(return_value=[mock_vivid_agent_info])
    registry.get_all_skills = Mock(return_value=[
        {
            "skill_id": "find-subreddits",
            "skill_name": "Find Subreddits",
            "skill_description": "Find relevant subreddits for a topic",
            "agent_app_id": mock_vivid_agent_info.agent_app_id,
            "agent_name": "Vivid Commenter",
            "endpoint_url": mock_vivid_agent_info.endpoint_url,
            "priority": 10,
            "skill_data": {
                "id": "find-subreddits",
                "name": "Find Subreddits",
                "description": "Find relevant subreddits for a topic"
            }
        }
    ])
    return registry


@pytest.fixture
def mock_selector_with_vivid_match():
    """Mock selector that returns Vivid Commenter for subreddit queries."""
    async def select_agent(user_request, context=None):
        if "subreddit" in user_request.lower():
            return AgentSelectionResult(
                agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
                skill_id="find-subreddits",
                skill_name="Find Subreddits",
                agent_name="Vivid Commenter",
                endpoint_url="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
                confidence="high",
                selection_method="llm"
            )
        return None

    selector = AsyncMock()
    selector.select_agent = select_agent
    return selector


@pytest.fixture(autouse=True)
def reset_par_adapter_state():
    """Reset par_adapter state before each test."""
    par_adapter._multi_agent_state = {
        "discovery_service": None,
        "selector": None,
        "client_pool": None,
        "registry": None,
        "initialized": False
    }
    yield
    par_adapter._multi_agent_state = {
        "discovery_service": None,
        "selector": None,
        "client_pool": None,
        "registry": None,
        "initialized": False
    }


@pytest.mark.asyncio
async def test_full_routing_flow_subreddit_query(
    mock_registry_with_vivid,
    mock_selector_with_vivid_match,
    mock_vivid_agent_info
):
    """Test full routing flow for 'find me 10 subreddits related to ai'."""

    # Setup: Initialize par_adapter state with components
    mock_client_pool = Mock()
    mock_client_pool.initialize = Mock()
    mock_client_pool.close_all = Mock()
    mock_client_pool.get_client = Mock(return_value=None)

    par_adapter._multi_agent_state = {
        "discovery_service": Mock(),
        "selector": mock_selector_with_vivid_match,
        "client_pool": mock_client_pool,
        "registry": mock_registry_with_vivid,
        "initialized": True
    }

    # Mock the entire UPEE flow
    with patch('src.par_adapter.UPEEEngine') as mock_upee_class, \
         patch('src.par_adapter.Settings') as mock_settings_class:

        # Setup mock settings
        mock_settings = Mock()
        mock_settings.a2a_enabled = True
        mock_settings.a2a_server_url = "http://localhost:9999"
        mock_settings.resolved_default_model = "gpt-4o"
        mock_settings.openai_api_key = "sk-test"
        mock_settings_class.return_value = mock_settings

        # Setup mock UPEE engine
        mock_upee_instance = AsyncMock()

        # Mock the process_request generator to return events
        import json
        async def mock_process():
            # Simulate CONTENT events
            yield {
                "event": "content",
                "data": json.dumps({"content": "Here are "})
            }
            yield {
                "event": "content",
                "data": json.dumps({"content": "subreddits"})
            }
            # Simulate COMPLETE event with routing metadata
            yield {
                "event": "complete",
                "data": json.dumps({
                    "model": "gpt-4o",
                    "agent_used": "Vivid Commenter",
                    "agent_app_id": "4906eeb7-9959-414e-84c6-f2445822ebe4",
                    "skill_used": "Find Subreddits",
                    "skill_id": "find-subreddits",
                    "routing_source": "a2a_agent",
                    "total_tokens": 100,
                    "duration": 1.5
                })
            }

        mock_upee_instance.process_request = Mock(return_value=mock_process())
        mock_upee_class.return_value = mock_upee_instance

        # Create the service and get handler
        service = par_adapter.create_service()
        handle_chat = service["custom_handlers"]["chat"]

        # Execute: Call handle_chat_request
        result = await handle_chat({
            "message": "find me 10 subreddits related to ai",
            "model": "gpt-4o",
            "show_thinking": "false"
        })

        # Verify: UPEE Engine was created with multi-agent components
        mock_upee_class.assert_called_once()
        call_kwargs = mock_upee_class.call_args[1]
        assert call_kwargs["registry"] == mock_registry_with_vivid
        assert call_kwargs["selector"] == mock_selector_with_vivid_match
        assert call_kwargs["client_pool"] == mock_client_pool

        # Verify: Result has success and routing metadata
        assert result["success"] is True
        assert "subreddits" in result["result"]
        assert result["metadata"]["agent_used"] == "Vivid Commenter"
        assert result["metadata"]["agent_app_id"] == "4906eeb7-9959-414e-84c6-f2445822ebe4"
        assert result["metadata"]["routing_source"] == "a2a_agent"
        assert result["metadata"]["skill_used"] == "Find Subreddits"


@pytest.mark.asyncio
async def test_routing_flow_no_match_handles_directly(
    mock_registry_with_vivid,
    mock_selector_with_vivid_match
):
    """Test that non-matching queries are handled directly by PAF Core."""

    # Setup par_adapter state
    mock_client_pool = Mock()
    par_adapter._multi_agent_state = {
        "discovery_service": Mock(),
        "selector": mock_selector_with_vivid_match,
        "client_pool": mock_client_pool,
        "registry": mock_registry_with_vivid,
        "initialized": True
    }

    with patch('src.par_adapter.UPEEEngine') as mock_upee_class, \
         patch('src.par_adapter.Settings') as mock_settings_class:

        mock_settings = Mock()
        mock_settings.a2a_enabled = True
        mock_settings.resolved_default_model = "gpt-4o"
        mock_settings_class.return_value = mock_settings

        mock_upee_instance = AsyncMock()

        # Mock process_request for direct PAF Core response
        import json
        async def mock_process():
            yield {
                "event": "content",
                "data": json.dumps({"content": "Python is a programming language"})
            }
            yield {
                "event": "complete",
                "data": json.dumps({
                    "model": "gpt-4o",
                    "agent_used": "PAF Core Agent",
                    "agent_app_id": None,
                    "skill_used": None,
                    "routing_source": "core_agent",
                    "total_tokens": 50,
                    "duration": 0.8
                })
            }

        mock_upee_instance.process_request = Mock(return_value=mock_process())
        mock_upee_class.return_value = mock_upee_instance

        service = par_adapter.create_service()
        handle_chat = service["custom_handlers"]["chat"]

        # Execute with non-matching query
        result = await handle_chat({
            "message": "what is python?",
            "model": "gpt-4o"
        })

        # Verify: Handled directly by PAF Core (no routing)
        assert result["success"] is True
        assert "Python" in result["result"]
        assert result["metadata"]["routing_source"] == "core_agent"
        assert result["metadata"]["agent_app_id"] is None


@pytest.mark.asyncio
async def test_routing_flow_without_initialization():
    """Test behavior when multi-agent components are not initialized."""

    # State is NOT initialized
    par_adapter._multi_agent_state["initialized"] = False

    with patch('src.par_adapter.UPEEEngine') as mock_upee_class, \
         patch('src.par_adapter.Settings') as mock_settings_class:

        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings

        mock_upee_instance = AsyncMock()
        async def mock_process():
            yield {"event": "content", "data": {"content": "response"}}
            yield {"event": "complete", "data": {"model": "gpt-4o"}}

        mock_upee_instance.process_request = Mock(return_value=mock_process())
        mock_upee_class.return_value = mock_upee_instance

        service = par_adapter.create_service()
        handle_chat = service["custom_handlers"]["chat"]

        result = await handle_chat({"message": "test"})

        # Verify: UPEE Engine created with None components
        call_kwargs = mock_upee_class.call_args[1]
        assert call_kwargs["registry"] is None
        assert call_kwargs["selector"] is None
        assert call_kwargs["client_pool"] is None

        # Still succeeds (degrades gracefully)
        assert result["success"] is True
