"""Tests for PAR adapter lifecycle management."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src import par_adapter
from src.agents.agent_app_registry import AgentAppRegistry


@pytest.fixture
def mock_discovery_service():
    """Mock discovery service for testing."""
    service = AsyncMock()
    service.startup = AsyncMock()
    service.shutdown = AsyncMock()

    # Mock registry
    registry = Mock(spec=AgentAppRegistry)
    registry.agents = {}
    service.get_registry = Mock(return_value=registry)

    return service


@pytest.fixture
def mock_selector():
    """Mock selector for testing."""
    return Mock()


@pytest.fixture
def mock_client_pool():
    """Mock client pool for testing."""
    pool = Mock()
    pool.initialize = Mock()
    pool.close_all = Mock()
    return pool


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state before each test."""
    par_adapter._multi_agent_state = {
        "discovery_service": None,
        "selector": None,
        "client_pool": None,
        "registry": None,
        "initialized": False
    }
    yield
    # Reset after test
    par_adapter._multi_agent_state = {
        "discovery_service": None,
        "selector": None,
        "client_pool": None,
        "registry": None,
        "initialized": False
    }


@pytest.mark.asyncio
async def test_startup_initializes_components(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that startup initializes all multi-agent components."""
    with patch('src.par_adapter.AgentAppDiscoveryService', return_value=mock_discovery_service), \
         patch('src.par_adapter.AgentAppSelector', return_value=mock_selector), \
         patch('src.par_adapter.AgentClientPool', return_value=mock_client_pool), \
         patch('src.par_adapter.Settings') as mock_settings:

        # Configure mock settings
        settings_instance = Mock()
        settings_instance.a2a_enabled = True
        settings_instance.a2a_agent_apps = [{"agent_app_id": "test-id"}]
        settings_instance.a2a_card_refresh_interval = 300
        mock_settings.return_value = settings_instance

        # Call startup
        result = await par_adapter.startup()

        # Verify discovery service was started
        mock_discovery_service.startup.assert_called_once()

        # Verify client pool was initialized
        mock_client_pool.initialize.assert_called_once()

        # Verify result
        assert result["status"] == "ready"
        assert result["multi_agent_enabled"] is True
        assert "agents_configured" in result


@pytest.mark.asyncio
async def test_startup_creates_discovery_service(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that startup creates discovery service."""
    with patch('src.par_adapter.AgentAppDiscoveryService', return_value=mock_discovery_service), \
         patch('src.par_adapter.AgentAppSelector', return_value=mock_selector), \
         patch('src.par_adapter.AgentClientPool', return_value=mock_client_pool), \
         patch('src.par_adapter.Settings'):

        await par_adapter.startup()

        # Verify state was updated
        assert par_adapter._multi_agent_state["initialized"] is True
        assert par_adapter._multi_agent_state["discovery_service"] is mock_discovery_service
        assert par_adapter._multi_agent_state["selector"] is mock_selector
        assert par_adapter._multi_agent_state["client_pool"] is mock_client_pool


@pytest.mark.asyncio
async def test_startup_creates_selector(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that startup creates agent selector."""
    with patch('src.par_adapter.AgentAppDiscoveryService', return_value=mock_discovery_service), \
         patch('src.par_adapter.AgentAppSelector', return_value=mock_selector) as mock_selector_class, \
         patch('src.par_adapter.AgentClientPool', return_value=mock_client_pool), \
         patch('src.par_adapter.Settings'):

        await par_adapter.startup()

        # Verify selector was created with correct args
        mock_selector_class.assert_called_once()
        # First arg is settings, second is registry
        call_args = mock_selector_class.call_args
        assert call_args[0][1] == mock_discovery_service.get_registry()


@pytest.mark.asyncio
async def test_shutdown_cleans_up(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that shutdown cleans up resources."""
    # Setup state
    par_adapter._multi_agent_state = {
        "discovery_service": mock_discovery_service,
        "selector": mock_selector,
        "client_pool": mock_client_pool,
        "registry": Mock(),
        "initialized": True
    }

    # Call shutdown
    result = await par_adapter.shutdown()

    # Verify discovery service was shut down
    mock_discovery_service.shutdown.assert_called_once()

    # Verify client pool was closed
    mock_client_pool.close_all.assert_called_once()

    # Verify state was reset
    assert par_adapter._multi_agent_state["initialized"] is False
    assert par_adapter._multi_agent_state["discovery_service"] is None
    assert par_adapter._multi_agent_state["selector"] is None

    # Verify result
    assert result["status"] == "shutdown_complete"


@pytest.mark.asyncio
async def test_state_persists_across_calls(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that state persists across multiple function calls."""
    with patch('src.par_adapter.AgentAppDiscoveryService', return_value=mock_discovery_service), \
         patch('src.par_adapter.AgentAppSelector', return_value=mock_selector), \
         patch('src.par_adapter.AgentClientPool', return_value=mock_client_pool), \
         patch('src.par_adapter.Settings'):

        # First call to startup
        await par_adapter.startup()

        # Verify state is initialized
        assert par_adapter._multi_agent_state["initialized"] is True
        first_discovery = par_adapter._multi_agent_state["discovery_service"]

        # State should persist (would be accessed in handle_chat_request)
        assert par_adapter._multi_agent_state["discovery_service"] is first_discovery
        assert par_adapter._multi_agent_state["initialized"] is True


@pytest.mark.asyncio
async def test_startup_handles_errors_gracefully():
    """Test that startup handles initialization errors gracefully."""
    with patch('src.par_adapter.AgentAppDiscoveryService', side_effect=Exception("Test error")), \
         patch('src.par_adapter.Settings'):

        result = await par_adapter.startup()

        # Should return ready_with_warnings, not crash
        assert result["status"] == "ready_with_warnings"
        assert result["multi_agent_enabled"] is False
        assert "error" in result

        # State should not be initialized
        assert par_adapter._multi_agent_state["initialized"] is False


@pytest.mark.asyncio
async def test_startup_stores_in_app_state(
    mock_discovery_service,
    mock_selector,
    mock_client_pool
):
    """Test that startup stores components in app.state."""
    with patch('src.par_adapter.AgentAppDiscoveryService', return_value=mock_discovery_service), \
         patch('src.par_adapter.AgentAppSelector', return_value=mock_selector), \
         patch('src.par_adapter.AgentClientPool', return_value=mock_client_pool), \
         patch('src.par_adapter.Settings'):

        # Create mock app
        mock_app = Mock()
        mock_app.state = Mock()

        # Call startup with app
        await par_adapter.startup(mock_app)

        # Verify components were stored in app.state
        assert mock_app.state.discovery_service == mock_discovery_service
        assert mock_app.state.selector == mock_selector
        assert mock_app.state.client_pool == mock_client_pool


@pytest.mark.asyncio
async def test_initialize_calls_startup():
    """Test that initialize() is an alias for startup()."""
    with patch('src.par_adapter.startup') as mock_startup:
        mock_startup.return_value = {"status": "ready"}

        result = await par_adapter.initialize()

        mock_startup.assert_called_once()
        assert result["status"] == "ready"
