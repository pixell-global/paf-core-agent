"""Tests for UPEE Engine with multi-agent components."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.core.upee_engine import UPEEEngine
from src.settings import Settings
from src.agents.agent_app_registry import AgentAppRegistry


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.a2a_enabled = True
    settings.a2a_server_url = "http://localhost:9999"
    settings.resolved_default_model = "gpt-4o"
    settings.openai_api_key = "sk-test"
    settings.anthropic_api_key = None
    settings.aws_region = None
    settings.max_context_tokens = 4000
    settings.debug = False
    return settings


@pytest.fixture
def mock_registry():
    """Mock registry for testing."""
    registry = Mock(spec=AgentAppRegistry)
    registry.agents = {}
    return registry


@pytest.fixture
def mock_selector():
    """Mock selector for testing."""
    return Mock()


@pytest.fixture
def mock_client_pool():
    """Mock client pool for testing."""
    return Mock()


@pytest.mark.asyncio
async def test_upee_engine_with_multi_agent_components(
    mock_settings,
    mock_registry,
    mock_selector,
    mock_client_pool
):
    """Test UPEE Engine initialization with multi-agent components."""
    with patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        engine = UPEEEngine(
            mock_settings,
            registry=mock_registry,
            selector=mock_selector,
            client_pool=mock_client_pool
        )

        # Verify components are stored
        assert engine.registry == mock_registry
        assert engine.selector == mock_selector
        assert engine.client_pool == mock_client_pool

        # Verify multi-agent mode is detected
        assert engine._use_multi_agent_mode is True

        # Verify agent_manager is NOT created in multi-agent mode
        assert engine.agent_manager is None


@pytest.mark.asyncio
async def test_upee_engine_legacy_mode(mock_settings):
    """Test UPEE Engine initialization without multi-agent components (legacy mode)."""
    with patch('src.core.upee_engine.AgentManager') as mock_agent_manager_class, \
         patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        mock_agent_manager = Mock()
        mock_agent_manager_class.return_value = mock_agent_manager

        engine = UPEEEngine(mock_settings)

        # Verify components are None
        assert engine.registry is None
        assert engine.selector is None
        assert engine.client_pool is None

        # Verify legacy mode
        assert engine._use_multi_agent_mode is False

        # Verify agent_manager IS created in legacy mode
        assert engine.agent_manager == mock_agent_manager


@pytest.mark.asyncio
async def test_upee_engine_passes_components_to_phases(
    mock_settings,
    mock_registry,
    mock_selector,
    mock_client_pool
):
    """Test that UPEE Engine passes components to phases."""
    with patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase') as mock_plan_phase_class, \
         patch('src.core.upee_engine.ExecutePhase') as mock_execute_phase_class, \
         patch('src.core.upee_engine.EvaluatePhase'):

        engine = UPEEEngine(
            mock_settings,
            registry=mock_registry,
            selector=mock_selector,
            client_pool=mock_client_pool
        )

        # Verify PlanPhase was created with registry and selector
        mock_plan_phase_class.assert_called_once()
        plan_call_kwargs = mock_plan_phase_class.call_args[1]
        assert plan_call_kwargs['registry'] == mock_registry
        assert plan_call_kwargs['selector'] == mock_selector

        # Verify ExecutePhase was created with client_pool
        mock_execute_phase_class.assert_called_once()
        execute_call_kwargs = mock_execute_phase_class.call_args[1]
        assert execute_call_kwargs['client_pool'] == mock_client_pool


@pytest.mark.asyncio
async def test_upee_engine_startup_with_components(
    mock_settings,
    mock_registry,
    mock_selector,
    mock_client_pool
):
    """Test that startup doesn't start agent_manager when using multi-agent components."""
    with patch('src.core.upee_engine.AgentManager') as mock_agent_manager_class, \
         patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        mock_agent_manager = AsyncMock()
        mock_agent_manager_class.return_value = mock_agent_manager

        engine = UPEEEngine(
            mock_settings,
            registry=mock_registry,
            selector=mock_selector,
            client_pool=mock_client_pool
        )

        # Call startup
        await engine.startup()

        # Verify agent_manager.startup() was NOT called
        mock_agent_manager.startup.assert_not_called()

        # Verify flag is not set
        assert engine._agent_manager_started is False


@pytest.mark.asyncio
async def test_upee_engine_startup_legacy_mode(mock_settings):
    """Test that startup starts agent_manager in legacy mode."""
    with patch('src.core.upee_engine.AgentManager') as mock_agent_manager_class, \
         patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        mock_agent_manager = AsyncMock()
        mock_agent_manager_class.return_value = mock_agent_manager

        engine = UPEEEngine(mock_settings)

        # Call startup
        await engine.startup()

        # Verify agent_manager.startup() WAS called
        mock_agent_manager.startup.assert_called_once()

        # Verify flag is set
        assert engine._agent_manager_started is True


@pytest.mark.asyncio
async def test_upee_engine_shutdown_with_components(
    mock_settings,
    mock_registry,
    mock_selector,
    mock_client_pool
):
    """Test that shutdown doesn't shutdown agent_manager when using multi-agent components."""
    with patch('src.core.upee_engine.AgentManager') as mock_agent_manager_class, \
         patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        mock_agent_manager = AsyncMock()
        mock_agent_manager_class.return_value = mock_agent_manager

        engine = UPEEEngine(
            mock_settings,
            registry=mock_registry,
            selector=mock_selector,
            client_pool=mock_client_pool
        )

        # Call shutdown
        await engine.shutdown()

        # Verify agent_manager.shutdown() was NOT called
        mock_agent_manager.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_upee_engine_partial_components_uses_legacy():
    """Test that partial components (only registry, no selector) triggers legacy mode."""
    mock_settings = Mock(spec=Settings)
    mock_settings.a2a_enabled = True
    mock_settings.a2a_server_url = "http://localhost:9999"
    mock_settings.openai_api_key = "sk-test"
    mock_registry = Mock()

    with patch('src.core.upee_engine.AgentManager') as mock_agent_manager_class, \
         patch('src.core.upee_engine.UnderstandPhase'), \
         patch('src.core.upee_engine.PlanPhase'), \
         patch('src.core.upee_engine.ExecutePhase'), \
         patch('src.core.upee_engine.EvaluatePhase'):

        mock_agent_manager = Mock()
        mock_agent_manager_class.return_value = mock_agent_manager

        # Create engine with only registry (no selector)
        engine = UPEEEngine(
            mock_settings,
            registry=mock_registry,
            selector=None  # Missing selector
        )

        # Should use legacy mode because selector is missing
        assert engine._use_multi_agent_mode is False
        assert engine.agent_manager == mock_agent_manager
