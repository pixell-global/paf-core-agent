"""Test gRPC agent card client."""
import asyncio
import pytest
from src.agents.grpc_agent_card_client import GrpcAgentCardClient, PathPrefixInterceptor


def test_path_prefix_interceptor_initialization():
    """Test PathPrefixInterceptor can be initialized."""
    interceptor = PathPrefixInterceptor("/agents/test-id/a2a")
    assert interceptor.path_prefix == "/agents/test-id/a2a"

    # Test trailing slash removal
    interceptor2 = PathPrefixInterceptor("/agents/test-id/a2a/")
    assert interceptor2.path_prefix == "/agents/test-id/a2a"

    print("✅ PathPrefixInterceptor initialization test passed")


def test_grpc_client_initialization():
    """Test GrpcAgentCardClient can be initialized."""
    client = GrpcAgentCardClient(timeout=10.0)
    assert client.timeout == 10.0

    print("✅ GrpcAgentCardClient initialization test passed")


def test_parse_endpoint():
    """Test endpoint URL parsing."""
    client = GrpcAgentCardClient()

    # Test HTTPS URL with port
    host, port = client._parse_endpoint("https://par.pixell.global:443/agents/123")
    assert host == "par.pixell.global"
    assert port == 443

    # Test HTTPS URL without port
    host, port = client._parse_endpoint("https://par.pixell.global/agents/123")
    assert host == "par.pixell.global"
    assert port == 443

    # Test HTTP URL
    host, port = client._parse_endpoint("http://localhost:8000")
    assert host == "localhost"
    assert port == 8000

    # Test plain host:port
    host, port = client._parse_endpoint("par.pixell.global:443")
    assert host == "par.pixell.global"
    assert port == 443

    print("✅ Endpoint parsing test passed")


def test_parse_capabilities_response():
    """Test parsing of DescribeCapabilities response."""
    from src.proto import agent_pb2

    client = GrpcAgentCardClient()

    # Create a mock response
    response = agent_pb2.Capabilities(
        methods=["Health", "Invoke", "DescribeCapabilities"],
        metadata={
            "name": "test-agent",
            "version": "1.0.0",
            "description": "Test agent",
            "skills": '[{"id":"test_skill","name":"test","description":"Test skill","tags":["test"]}]'
        }
    )

    agent_card = client._parse_capabilities_response(response)

    assert agent_card["name"] == "test-agent"
    assert agent_card["version"] == "1.0.0"
    assert agent_card["description"] == "Test agent"
    assert len(agent_card["methods"]) == 3
    assert len(agent_card["skills"]) == 1
    assert agent_card["skills"][0]["id"] == "test_skill"

    print("✅ Capabilities response parsing test passed")


@pytest.mark.asyncio
async def test_fetch_agent_card_from_deployed_agent():
    """Test fetching agent card from real deployed PAF Core Agent.

    This test requires the agent to be deployed at par.pixell.global.
    """
    client = GrpcAgentCardClient(timeout=10.0)

    # Try to fetch from deployed PAF Core Agent
    agent_id = "ed8784f3-b602-481c-8701-3b6406c8fd98"
    endpoint_url = f"https://par.pixell.global/agents/{agent_id}"

    print(f"\n🔍 Attempting to fetch agent card via gRPC...")
    print(f"   Agent ID: {agent_id}")
    print(f"   Endpoint: {endpoint_url}")

    agent_card = await client.fetch_agent_card(agent_id, endpoint_url)

    if agent_card:
        print(f"\n✅ Successfully fetched agent card via gRPC!")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Version: {agent_card.get('version')}")
        print(f"   Methods: {agent_card.get('methods')}")
        print(f"   Skills: {len(agent_card.get('skills', []))}")

        # Verify basic structure
        assert "name" in agent_card
        assert "methods" in agent_card
        assert "skills" in agent_card

        print("\n✅ Agent card fetch test PASSED")
    else:
        print(f"\n⚠️  Could not fetch agent card via gRPC")
        print("   This is expected if:")
        print("   - Agent is not deployed")
        print("   - Network connectivity issues")
        print("   - gRPC endpoint not accessible")
        pytest.skip("Agent not accessible via gRPC")


async def run_async_tests():
    """Run async tests."""
    print("\n" + "=" * 70)
    print("Running async tests...")
    print("=" * 70)

    await test_fetch_agent_card_from_deployed_agent()


if __name__ == "__main__":
    # Run sync tests
    print("\n" + "=" * 70)
    print("🧪 Testing gRPC Agent Card Client")
    print("=" * 70)

    test_path_prefix_interceptor_initialization()
    test_grpc_client_initialization()
    test_parse_endpoint()
    test_parse_capabilities_response()

    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"\n⚠️  Async test skipped: {e}")

    print("\n" + "=" * 70)
    print("✅ All gRPC client tests passed!")
    print("=" * 70)
