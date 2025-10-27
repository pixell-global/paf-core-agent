"""Test A2A proto structure to verify proto changes."""
import json
from src.proto import agent_pb2


def test_a2a_message_exists():
    """Verify A2AMessage proto class exists."""
    assert hasattr(agent_pb2, 'A2AMessage')
    print("✅ A2AMessage class exists in agent_pb2")


def test_a2a_message_fields():
    """Verify A2AMessage has all required fields."""
    a2a_msg = agent_pb2.A2AMessage()

    # Check all fields exist
    assert hasattr(a2a_msg, 'jsonrpc')
    assert hasattr(a2a_msg, 'id')
    assert hasattr(a2a_msg, 'method')
    assert hasattr(a2a_msg, 'params_json')

    print("✅ A2AMessage has all required fields")


def test_a2a_message_can_be_created():
    """Test creating and populating A2AMessage."""
    params = {
        "message": {
            "kind": "message",
            "role": "user",
            "messageId": "test-msg-123",
            "metadata": {
                "skill": "test_skill",
                "params": {"query": "test"}
            },
            "parts": [
                {"kind": "text", "text": '{"query": "test"}'}
            ]
        }
    }

    a2a_msg = agent_pb2.A2AMessage(
        jsonrpc="2.0",
        id="req-123",
        method="message/send",
        params_json=json.dumps(params)
    )

    assert a2a_msg.jsonrpc == "2.0"
    assert a2a_msg.id == "req-123"
    assert a2a_msg.method == "message/send"

    # Verify params can be parsed back
    parsed_params = json.loads(a2a_msg.params_json)
    assert parsed_params["message"]["metadata"]["skill"] == "test_skill"
    assert parsed_params["message"]["metadata"]["params"]["query"] == "test"
    assert len(parsed_params["message"]["parts"]) == 1

    print("✅ A2AMessage can be created and populated correctly")


def test_action_request_has_a2a_field():
    """Verify ActionRequest has a2a_message field."""
    action_req = agent_pb2.ActionRequest()

    assert hasattr(action_req, 'a2a_message')
    print("✅ ActionRequest has a2a_message field")


def test_action_request_with_a2a_message():
    """Test creating ActionRequest with A2AMessage."""
    params = {
        "message": {
            "kind": "message",
            "role": "user",
            "messageId": "msg-456",
            "metadata": {
                "skill": "reddit_search",
                "params": {"keywords": ["ai", "ml"]}
            },
            "parts": [
                {"kind": "text", "text": '{"keywords": ["ai", "ml"]}'}
            ]
        }
    }

    a2a_msg = agent_pb2.A2AMessage(
        jsonrpc="2.0",
        id="req-456",
        method="message/send",
        params_json=json.dumps(params)
    )

    action_req = agent_pb2.ActionRequest(
        a2a_message=a2a_msg
    )

    assert action_req.a2a_message.jsonrpc == "2.0"
    assert action_req.a2a_message.id == "req-456"
    assert action_req.a2a_message.method == "message/send"

    # Verify params
    parsed = json.loads(action_req.a2a_message.params_json)
    assert parsed["message"]["metadata"]["skill"] == "reddit_search"

    print("✅ ActionRequest can wrap A2AMessage correctly")


def test_backward_compatibility_fields():
    """Verify legacy fields still exist for backward compatibility."""
    action_req = agent_pb2.ActionRequest()

    # Legacy fields should still exist
    assert hasattr(action_req, 'action')
    assert hasattr(action_req, 'parameters')
    assert hasattr(action_req, 'request_id')

    print("✅ Backward compatibility fields preserved")


def test_action_request_can_use_both_formats():
    """Test that ActionRequest can use either new or legacy format."""
    # New format
    new_req = agent_pb2.ActionRequest(
        a2a_message=agent_pb2.A2AMessage(
            jsonrpc="2.0",
            id="new-123",
            method="message/send",
            params_json='{"test": "data"}'
        )
    )
    assert new_req.a2a_message.id == "new-123"

    # Legacy format
    legacy_req = agent_pb2.ActionRequest(
        action="invoke",
        parameters={"key": "value"},
        request_id="legacy-123"
    )
    assert legacy_req.action == "invoke"
    assert legacy_req.request_id == "legacy-123"

    print("✅ ActionRequest supports both new and legacy formats")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 Testing A2A Proto Structure")
    print("=" * 70)

    test_a2a_message_exists()
    test_a2a_message_fields()
    test_a2a_message_can_be_created()
    test_action_request_has_a2a_field()
    test_action_request_with_a2a_message()
    test_backward_compatibility_fields()
    test_action_request_can_use_both_formats()

    print("\n" + "=" * 70)
    print("✅ All A2A proto structure tests passed!")
    print("=" * 70)
