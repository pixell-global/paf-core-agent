"""Integration test for A2A gRPC communication.

Tests end-to-end flow of sending A2A-compliant messages via gRPC.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.grpc_a2a_client import GrpcA2AClient
from src.proto import agent_pb2


@pytest.mark.asyncio
async def test_a2a_message_flow():
    """Test complete A2A message flow from client to mock receiver."""
    client = GrpcA2AClient(
        agent_app_id="vivid-commenter",
        endpoint_url="https://par.pixell.global/agents/vivid-commenter"
    )

    # Prepare message
    message = {
        "skill_id": "generate_comment",
        "parameters": {
            "post_text": "What are the best skincare routines?",
            "context": "reddit_post"
        },
        "user_message": "Generate a helpful comment"
    }

    # Mock the gRPC channel and stub
    with patch('grpc.aio.secure_channel') as mock_channel_func:
        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_channel_func.return_value = mock_channel

        # Create a mock receiver that validates A2A format
        def validate_a2a_request(request):
            """Validate that request conforms to A2A spec."""
            # Check it has a2a_message
            assert request.HasField("a2a_message"), "Request must have a2a_message field"

            a2a_msg = request.a2a_message

            # Validate A2A structure
            assert a2a_msg.jsonrpc == "2.0", "jsonrpc must be '2.0'"
            assert a2a_msg.method == "message/send", "method must be 'message/send'"
            assert a2a_msg.id != "", "id must not be empty"

            # Parse params
            params = json.loads(a2a_msg.params_json)
            msg = params["message"]

            # Validate message structure
            assert msg["kind"] == "message"
            assert msg["role"] == "user"
            assert msg["messageId"] != ""

            # Validate metadata uses correct field names
            metadata = msg["metadata"]
            assert "skill" in metadata, "Must use 'skill' not 'action'"
            assert "params" in metadata, "Must use 'params' not 'parameters'"
            assert "action" not in metadata, "Must not use 'action'"
            assert "parameters" not in metadata, "Must not use 'parameters'"

            # Validate skill and params
            assert metadata["skill"] == "generate_comment"
            assert metadata["params"]["post_text"] == "What are the best skincare routines?"

            # Validate parts array
            assert "parts" in msg, "Must have parts array"
            assert len(msg["parts"]) > 0, "Parts array must not be empty"

            # Return mock success response
            return agent_pb2.ActionResult(
                success=True,
                result=json.dumps({"comment": "Great question! Here are some tips..."}),
                request_id=a2a_msg.id,
                duration_ms=1200
            )

        with patch('src.agents.grpc_a2a_client.agent_pb2_grpc.AgentServiceStub') as mock_stub_class:
            mock_stub_class.return_value = mock_stub
            mock_stub.Invoke.side_effect = validate_a2a_request

            # Send message
            result = await client.send_message(message)

            # Verify successful response
            assert result["status"] == "success"
            assert "comment" in json.loads(result["data"])

    print("✅ A2A gRPC integration test passed")


@pytest.mark.asyncio
async def test_receiver_compatibility():
    """Test that our A2A format would be compatible with standard receivers."""
    client = GrpcA2AClient(
        agent_app_id="test-agent",
        endpoint_url="https://par.pixell.global/agents/test-agent"
    )

    # Create test message
    message = {
        "skill_id": "reddit_search_post",
        "parameters": {"query": "skincare"}
    }

    # Build request
    request = client._build_action_request(message)

    # Parse the A2A message
    params = json.loads(request.a2a_message.params_json)

    # Simulate what a standard A2A receiver would expect
    expected_structure = {
        "message": {
            "kind": str,
            "role": str,
            "messageId": str,
            "metadata": {
                "skill": str,
                "params": dict
            },
            "parts": list
        }
    }

    def validate_structure(data, expected):
        """Recursively validate data structure."""
        for key, value_type in expected.items():
            assert key in data, f"Missing required key: {key}"

            if isinstance(value_type, dict):
                validate_structure(data[key], value_type)
            elif isinstance(value_type, type):
                assert isinstance(data[key], value_type), \
                    f"Key '{key}' should be {value_type}, got {type(data[key])}"

    # Validate structure matches what receivers expect
    validate_structure(params, expected_structure)

    # Verify no legacy field names
    metadata = params["message"]["metadata"]
    assert "action" not in metadata
    assert "parameters" not in metadata

    print("✅ Receiver compatibility test passed")


def test_a2a_format_matches_spec():
    """Test that generated A2A format exactly matches the specification."""
    client = GrpcA2AClient(
        agent_app_id="test-agent",
        endpoint_url="https://par.pixell.global/agents/test-agent"
    )

    message = {
        "skill_id": "reddit_search_post",
        "parameters": {"query": "skincare"}
    }

    request = client._build_action_request(message)

    # Expected A2A format (from issue #13 specification)
    expected_format = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "metadata": {
                    "skill": "reddit_search_post",
                    "params": {"query": "skincare"}
                },
                "parts": [
                    {"kind": "text"}
                ]
            }
        }
    }

    # Verify top-level A2A message
    assert request.a2a_message.jsonrpc == expected_format["jsonrpc"]
    assert request.a2a_message.method == expected_format["method"]
    assert request.a2a_message.id != ""  # Should have ID

    # Parse and verify nested structure
    params = json.loads(request.a2a_message.params_json)
    msg = params["message"]

    assert msg["kind"] == expected_format["params"]["message"]["kind"]
    assert msg["role"] == expected_format["params"]["message"]["role"]
    assert msg["messageId"] != ""

    # Verify metadata structure
    assert msg["metadata"]["skill"] == expected_format["params"]["message"]["metadata"]["skill"]
    assert msg["metadata"]["params"] == expected_format["params"]["message"]["metadata"]["params"]

    # Verify parts structure
    assert len(msg["parts"]) > 0
    for part in msg["parts"]:
        assert part["kind"] == "text"
        assert "text" in part

    print("✅ A2A format specification match test passed")


if __name__ == "__main__":
    import asyncio

    print("\n" + "=" * 70)
    print("🧪 Testing A2A gRPC Integration")
    print("=" * 70)

    # Run async tests
    asyncio.run(test_a2a_message_flow())
    asyncio.run(test_receiver_compatibility())

    # Run sync test
    test_a2a_format_matches_spec()

    print("\n" + "=" * 70)
    print("✅ All A2A gRPC integration tests passed!")
    print("=" * 70)
