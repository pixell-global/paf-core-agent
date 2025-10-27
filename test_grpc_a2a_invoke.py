"""Test gRPC A2A client for Invoke calls."""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.agents.grpc_a2a_client import GrpcA2AClient
from src.agents.hybrid_agent_client import HybridAgentClient
from src.agents.agent_app_registry import AgentAppInfo


def test_grpc_a2a_client_initialization():
    """Test GrpcA2AClient initialization."""
    client = GrpcA2AClient(
        agent_app_id="test-agent-id",
        endpoint_url="https://par.pixell.global/agents/test-agent-id",
        timeout=30.0
    )

    assert client.agent_app_id == "test-agent-id"
    assert client.timeout == 30.0

    print("✅ GrpcA2AClient initialization test passed")


def test_parse_endpoint():
    """Test endpoint URL parsing."""
    client = GrpcA2AClient(
        agent_app_id="test-id",
        endpoint_url="https://par.pixell.global:443/agents/test-id"
    )

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

    print("✅ Endpoint parsing test passed")


def test_build_action_request():
    """Test building ActionRequest from message payload with A2A format."""
    from src.proto import agent_pb2
    import json

    client = GrpcA2AClient(
        agent_app_id="test-id",
        endpoint_url="https://par.pixell.global/agents/test-id"
    )

    message = {
        "type": "skill_request",
        "skill_id": "test_skill",
        "skill_name": "Test Skill",
        "parameters": {
            "keywords": ["ai", "machine learning"],
            "limit": 10
        },
        "user_message": "Find 10 subreddits"
    }

    request = client._build_action_request(message)

    # Verify it's an ActionRequest
    assert isinstance(request, agent_pb2.ActionRequest)

    # Verify it has A2A message
    assert request.HasField("a2a_message")
    a2a_msg = request.a2a_message

    # Verify A2A structure
    assert a2a_msg.jsonrpc == "2.0"
    assert a2a_msg.id != ""
    assert a2a_msg.method == "message/send"
    assert a2a_msg.params_json != ""

    # Parse and verify params
    params = json.loads(a2a_msg.params_json)
    assert "message" in params

    msg = params["message"]
    assert msg["kind"] == "message"
    assert msg["role"] == "user"
    assert msg["messageId"] != ""

    # Verify metadata
    metadata = msg["metadata"]
    assert metadata["skill"] == "test_skill"
    assert metadata["params"]["keywords"] == ["ai", "machine learning"]
    assert metadata["params"]["limit"] == 10

    # Verify parts array exists
    assert "parts" in msg
    assert len(msg["parts"]) >= 1
    assert msg["parts"][0]["kind"] == "text"

    print("✅ ActionRequest building test passed (A2A format)")


def test_parse_action_result_success():
    """Test parsing successful ActionResult."""
    from src.proto import agent_pb2

    client = GrpcA2AClient(
        agent_app_id="test-id",
        endpoint_url="https://par.pixell.global/agents/test-id"
    )

    # Create success response
    proto_response = agent_pb2.ActionResult(
        success=True,
        result="Result data here",
        request_id="req-123",
        duration_ms=1500,
        metadata={"agent": "vivid-commenter"}
    )

    result = client._parse_action_result(proto_response, {})

    assert result["status"] == "success"
    assert result["data"] == "Result data here"
    assert result["duration_ms"] == 1500
    assert result["metadata"]["agent"] == "vivid-commenter"

    print("✅ ActionResult parsing (success) test passed")


def test_parse_action_result_error():
    """Test parsing error ActionResult."""
    from src.proto import agent_pb2

    client = GrpcA2AClient(
        agent_app_id="test-id",
        endpoint_url="https://par.pixell.global/agents/test-id"
    )

    # Create error response
    proto_response = agent_pb2.ActionResult(
        success=False,
        error="Something went wrong",
        request_id="req-123",
        duration_ms=500
    )

    result = client._parse_action_result(proto_response, {})

    assert result["status"] == "error"
    assert result["error"] == "Something went wrong"
    assert result["duration_ms"] == 500

    print("✅ ActionResult parsing (error) test passed")


def test_hybrid_client_protocol_detection_par():
    """Test HybridAgentClient detects gRPC for PAR agents."""
    agent_info = AgentAppInfo(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        endpoint_url="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="https",
        enabled=True
    )

    client = HybridAgentClient(agent_info)

    assert client.use_grpc == True
    assert client.get_protocol() == "grpc"

    print("✅ HybridAgentClient PAR detection test passed")


def test_hybrid_client_protocol_detection_local():
    """Test HybridAgentClient detects HTTP for local agents."""
    agent_info = AgentAppInfo(
        agent_app_id="local-agent",
        name="Local Agent",
        endpoint_url="http://localhost:9999",
        protocol="http",
        enabled=True
    )

    client = HybridAgentClient(agent_info)

    assert client.use_grpc == False
    assert client.get_protocol() == "http"

    print("✅ HybridAgentClient local detection test passed")


def test_hybrid_client_force_grpc():
    """Test HybridAgentClient force_grpc option."""
    agent_info = AgentAppInfo(
        agent_app_id="local-agent",
        name="Local Agent",
        endpoint_url="http://localhost:9999",
        protocol="http",
        enabled=True
    )

    client = HybridAgentClient(agent_info, force_grpc=True)

    assert client.use_grpc == True
    assert client.get_protocol() == "grpc"

    print("✅ HybridAgentClient force_grpc test passed")


def test_hybrid_client_accessors():
    """Test HybridAgentClient accessor methods."""
    agent_info = AgentAppInfo(
        agent_app_id="test-agent",
        name="Test Agent",
        endpoint_url="https://par.pixell.global/agents/test-agent",
        protocol="https",
        enabled=True
    )

    client = HybridAgentClient(agent_info)

    assert client.get_agent_id() == "test-agent"
    assert "par.pixell.global" in client.get_endpoint()

    print("✅ HybridAgentClient accessor test passed")


@pytest.mark.asyncio
async def test_grpc_a2a_client_with_mock():
    """Test GrpcA2AClient send_message with mocked gRPC."""
    from src.proto import agent_pb2

    client = GrpcA2AClient(
        agent_app_id="mock-agent-id",
        endpoint_url="https://par.pixell.global/agents/mock-agent-id"
    )

    message = {
        "type": "skill_request",
        "skill_id": "test_skill",
        "skill_name": "Test Skill",
        "parameters": {"limit": 10},
        "user_message": "Test message"
    }

    # Mock the gRPC channel and stub
    with patch('grpc.aio.secure_channel') as mock_channel_func:
        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_channel_func.return_value = mock_channel

        # Mock successful response
        mock_response = agent_pb2.ActionResult(
            success=True,
            result="Mock result",
            request_id="req-123",
            duration_ms=1000
        )

        with patch('src.agents.grpc_a2a_client.agent_pb2_grpc.AgentServiceStub') as mock_stub_class:
            mock_stub_class.return_value = mock_stub
            mock_stub.Invoke.return_value = mock_response

            # Call send_message
            result = await client.send_message(message)

            # Verify call was made with correct path prefix
            # The interceptor should prepend /agents/mock-agent-id/a2a
            mock_channel_func.assert_called_once()

            # Verify result
            assert result["status"] == "success"
            assert result["data"] == "Mock result"

    print("✅ GrpcA2AClient mock test passed")


async def run_async_tests():
    """Run async tests."""
    print("\n" + "=" * 70)
    print("Running async tests...")
    print("=" * 70)

    await test_grpc_a2a_client_with_mock()


if __name__ == "__main__":
    # Run sync tests
    print("\n" + "=" * 70)
    print("🧪 Testing gRPC A2A Client for Invoke Calls")
    print("=" * 70)

    test_grpc_a2a_client_initialization()
    test_parse_endpoint()
    test_build_action_request()
    test_parse_action_result_success()
    test_parse_action_result_error()
    test_hybrid_client_protocol_detection_par()
    test_hybrid_client_protocol_detection_local()
    test_hybrid_client_force_grpc()
    test_hybrid_client_accessors()

    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"\n⚠️  Some async tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 70)
    print("✅ All gRPC A2A client tests passed!")
    print("=" * 70)
