"""Test agent discovery with gRPC and HTTP fallback."""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.agents.agent_app_discovery import AgentAppDiscoveryService
from src.agents.agent_app_registry import AgentAppInfo
from src.settings import Settings


def test_agent_discovery_initialization():
    """Test that AgentAppDiscoveryService initializes correctly."""
    settings = Settings()
    discovery = AgentAppDiscoveryService(settings)

    assert discovery.settings == settings
    assert discovery.registry is not None
    assert discovery._discovery_task is None

    print("✅ AgentAppDiscoveryService initialization test passed")


@pytest.mark.asyncio
async def test_fetch_par_agent_via_grpc():
    """Test fetching agent card from PAR agent via gRPC.

    This should use gRPC for PAR-deployed agents.
    """
    settings = Settings()
    discovery = AgentAppDiscoveryService(settings)

    # Create a PAR agent
    par_agent = AgentAppInfo(
        agent_app_id="test-agent-id",
        name="Test PAR Agent",
        endpoint_url="https://par.pixell.global/agents/test-agent-id",
        protocol="https",
        enabled=True
    )

    discovery.registry.add_agent(par_agent)

    # Mock the gRPC client
    mock_agent_card = {
        "name": "Test PAR Agent",
        "version": "1.0.0",
        "description": "Test agent on PAR",
        "methods": ["Health", "DescribeCapabilities", "Invoke"],
        "skills": [
            {
                "id": "test_skill",
                "name": "Test Skill",
                "description": "A test skill",
                "tags": ["test"]
            }
        ]
    }

    with patch('src.agents.agent_app_discovery.GrpcAgentCardClient') as MockGrpcClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.fetch_agent_card.return_value = mock_agent_card
        MockGrpcClient.return_value = mock_client_instance

        # Fetch the agent card
        result = await discovery._fetch_agent_card(par_agent)

        # Verify gRPC was called
        MockGrpcClient.assert_called_once_with(timeout=10.0)
        mock_client_instance.fetch_agent_card.assert_called_once_with(
            "test-agent-id",
            "https://par.pixell.global/agents/test-agent-id"
        )

        # Verify result
        assert result == mock_agent_card

        # Verify registry was updated
        registry_agent = discovery.registry.get_agent("test-agent-id")
        assert registry_agent.agent_card == mock_agent_card
        assert registry_agent.health_status == "healthy"

    print("✅ PAR agent gRPC fetch test passed")


@pytest.mark.asyncio
async def test_fetch_non_par_agent_via_http():
    """Test fetching agent card from non-PAR agent via HTTP.

    This should skip gRPC and use HTTP directly.
    """
    settings = Settings()
    discovery = AgentAppDiscoveryService(settings)

    # Create a non-PAR agent
    http_agent = AgentAppInfo(
        agent_app_id="http-agent-id",
        name="HTTP Agent",
        endpoint_url="https://example.com/agent",
        protocol="https",
        enabled=True
    )

    discovery.registry.add_agent(http_agent)

    mock_agent_card = {
        "name": "HTTP Agent",
        "version": "1.0.0",
        "skills": [{"id": "http_skill", "name": "HTTP Skill"}]
    }

    with patch('httpx.AsyncClient') as MockHttpClient:
        mock_client = AsyncMock()

        # Create proper mock response
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_agent_card)
        mock_response.raise_for_status = Mock()

        mock_client.get = AsyncMock(return_value=mock_response)

        # Create async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_client
        mock_context.__aexit__.return_value = None
        MockHttpClient.return_value = mock_context

        # Fetch the agent card
        result = await discovery._fetch_agent_card(http_agent)

        # Verify HTTP was called
        mock_client.get.assert_called_once_with(
            "https://example.com/agent/.well-known/agent.json"
        )

        # Verify result
        assert result == mock_agent_card

        # Verify registry was updated
        registry_agent = discovery.registry.get_agent("http-agent-id")
        assert registry_agent.agent_card == mock_agent_card
        assert registry_agent.health_status == "healthy"

    print("✅ Non-PAR agent HTTP fetch test passed")


@pytest.mark.asyncio
async def test_grpc_fallback_to_http():
    """Test fallback from gRPC to HTTP when gRPC fails.

    This should try gRPC first, then fall back to HTTP.
    """
    settings = Settings()
    discovery = AgentAppDiscoveryService(settings)

    # Create a PAR agent
    par_agent = AgentAppInfo(
        agent_app_id="fallback-agent-id",
        name="Fallback Agent",
        endpoint_url="https://par.pixell.global/agents/fallback-agent-id",
        protocol="https",
        enabled=True
    )

    discovery.registry.add_agent(par_agent)

    mock_agent_card = {
        "name": "Fallback Agent",
        "version": "1.0.0",
        "skills": [{"id": "fallback_skill", "name": "Fallback Skill"}]
    }

    # Mock gRPC to fail, HTTP to succeed
    with patch('src.agents.agent_app_discovery.GrpcAgentCardClient') as MockGrpcClient:
        mock_grpc_instance = AsyncMock()
        mock_grpc_instance.fetch_agent_card.return_value = None  # gRPC fails
        MockGrpcClient.return_value = mock_grpc_instance

        with patch('httpx.AsyncClient') as MockHttpClient:
            mock_http_client = AsyncMock()

            # Create proper mock response
            mock_response = Mock()
            mock_response.json = Mock(return_value=mock_agent_card)
            mock_response.raise_for_status = Mock()

            mock_http_client.get = AsyncMock(return_value=mock_response)

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_http_client
            mock_context.__aexit__.return_value = None
            MockHttpClient.return_value = mock_context

            # Fetch the agent card
            result = await discovery._fetch_agent_card(par_agent)

            # Verify gRPC was tried first
            mock_grpc_instance.fetch_agent_card.assert_called_once()

            # Verify HTTP was called as fallback
            mock_http_client.get.assert_called_once_with(
                "https://par.pixell.global/agents/fallback-agent-id/.well-known/agent.json"
            )

            # Verify result
            assert result == mock_agent_card

            # Verify registry was updated
            registry_agent = discovery.registry.get_agent("fallback-agent-id")
            assert registry_agent.agent_card == mock_agent_card
            assert registry_agent.health_status == "healthy"

    print("✅ gRPC to HTTP fallback test passed")


@pytest.mark.asyncio
async def test_fetch_real_par_agent():
    """Integration test: Fetch from real PAR-deployed agent.

    This test requires network connectivity to par.pixell.global.
    """
    settings = Settings()
    discovery = AgentAppDiscoveryService(settings)

    # Use real deployed PAF Core Agent
    real_agent = AgentAppInfo(
        agent_app_id="ed8784f3-b602-481c-8701-3b6406c8fd98",
        name="PAF Core Agent (Real)",
        endpoint_url="https://par.pixell.global/agents/ed8784f3-b602-481c-8701-3b6406c8fd98",
        protocol="https",
        enabled=True
    )

    discovery.registry.add_agent(real_agent)

    print(f"\n🔍 Attempting to fetch from real PAR agent...")
    print(f"   Agent ID: {real_agent.agent_app_id}")
    print(f"   Endpoint: {real_agent.endpoint_url}")

    # Fetch the agent card
    result = await discovery._fetch_agent_card(real_agent)

    if result:
        print(f"\n✅ Successfully fetched real agent card!")
        print(f"   Name: {result.get('name')}")
        print(f"   Version: {result.get('version')}")
        print(f"   Methods: {result.get('methods')}")
        print(f"   Skills: {len(result.get('skills', []))}")

        # Verify basic structure
        assert "name" in result
        assert "methods" in result
        assert "skills" in result

        # Verify registry was updated
        registry_agent = discovery.registry.get_agent(real_agent.agent_app_id)
        assert registry_agent.agent_card == result
        assert registry_agent.health_status == "healthy"

        print("\n✅ Real PAR agent fetch test PASSED")
    else:
        print(f"\n⚠️  Could not fetch real agent card")
        print("   This is expected if:")
        print("   - Agent is not deployed")
        print("   - Network connectivity issues")
        print("   - gRPC endpoint not accessible")
        pytest.skip("Real agent not accessible")


async def run_async_tests():
    """Run async tests."""
    print("\n" + "=" * 70)
    print("Running async tests...")
    print("=" * 70)

    await test_fetch_par_agent_via_grpc()
    await test_fetch_non_par_agent_via_http()
    await test_grpc_fallback_to_http()
    await test_fetch_real_par_agent()


if __name__ == "__main__":
    # Run sync tests
    print("\n" + "=" * 70)
    print("🧪 Testing Agent Discovery with gRPC/HTTP Fallback")
    print("=" * 70)

    test_agent_discovery_initialization()

    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"\n⚠️  Some async tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 70)
    print("✅ All agent discovery tests passed!")
    print("=" * 70)
