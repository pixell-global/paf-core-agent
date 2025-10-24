"""Test proto imports work correctly."""
import pytest


def test_proto_imports():
    """Test that proto files can be imported."""
    try:
        from src.proto import agent_pb2, agent_pb2_grpc
        assert agent_pb2 is not None
        assert agent_pb2_grpc is not None
        print("✅ Proto imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import proto files: {e}")


def test_proto_message_creation():
    """Test that we can create proto messages."""
    from src.proto import agent_pb2

    # Create an Empty message
    empty = agent_pb2.Empty()
    assert empty is not None

    # Create an ActionRequest
    request = agent_pb2.ActionRequest(
        action="test",
        parameters={"key": "value"}
    )
    assert request.action == "test"
    assert "key" in request.parameters

    print("✅ Proto message creation successful")


if __name__ == "__main__":
    test_proto_imports()
    test_proto_message_creation()
    print("\n✅ All proto tests passed!")
