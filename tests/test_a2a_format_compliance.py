"""Test A2A format compliance for gRPC messages.

Verifies that messages built by GrpcA2AClient strictly comply with
the A2A (Agent-to-Agent) JSON-RPC 2.0 specification.
"""
import json
import pytest
from src.agents.grpc_a2a_client import GrpcA2AClient


class TestA2AFormatCompliance:
    """Test suite for A2A format compliance."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = GrpcA2AClient(
            agent_app_id="test-agent",
            endpoint_url="https://par.pixell.global/agents/test-agent"
        )

    def test_jsonrpc_version(self):
        """Test that jsonrpc field is always '2.0'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        assert request.a2a_message.jsonrpc == "2.0"

    def test_method_is_message_send(self):
        """Test that method field is 'message/send'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        assert request.a2a_message.method == "message/send"

    def test_has_unique_id(self):
        """Test that each message has a unique ID."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        req1 = self.client._build_action_request(message)
        req2 = self.client._build_action_request(message)

        assert req1.a2a_message.id != ""
        assert req2.a2a_message.id != ""
        assert req1.a2a_message.id != req2.a2a_message.id

    def test_params_structure(self):
        """Test that params has correct nested structure."""
        message = {
            "skill_id": "reddit_search",
            "parameters": {"keywords": ["ai", "ml"]}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        # Verify structure
        assert "message" in params
        assert "kind" in params["message"]
        assert "role" in params["message"]
        assert "messageId" in params["message"]
        assert "metadata" in params["message"]
        assert "parts" in params["message"]

    def test_metadata_uses_skill_not_action(self):
        """Test that metadata uses 'skill' field, not 'action'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata = params["message"]["metadata"]
        assert "skill" in metadata
        assert "action" not in metadata
        assert metadata["skill"] == "test_skill"

    def test_metadata_uses_params_not_parameters(self):
        """Test that metadata uses 'params' field, not 'parameters'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test", "limit": 10}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata = params["message"]["metadata"]
        assert "params" in metadata
        assert "parameters" not in metadata
        assert metadata["params"]["query"] == "test"
        assert metadata["params"]["limit"] == 10

    def test_parts_array_exists(self):
        """Test that parts array is present."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        assert "parts" in params["message"]
        assert isinstance(params["message"]["parts"], list)
        assert len(params["message"]["parts"]) > 0

    def test_parts_array_has_text_kind(self):
        """Test that parts have 'kind' field set to 'text'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        parts = params["message"]["parts"]
        for part in parts:
            assert "kind" in part
            assert part["kind"] == "text"
            assert "text" in part

    def test_parameters_with_dict(self):
        """Test handling of dictionary parameters."""
        message = {
            "skill_id": "test_skill",
            "parameters": {
                "config": {
                    "timeout": 30,
                    "retries": 3
                }
            }
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata_params = params["message"]["metadata"]["params"]
        assert metadata_params["config"]["timeout"] == 30
        assert metadata_params["config"]["retries"] == 3

    def test_parameters_with_list(self):
        """Test handling of list parameters."""
        message = {
            "skill_id": "test_skill",
            "parameters": {
                "keywords": ["python", "javascript", "rust"]
            }
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata_params = params["message"]["metadata"]["params"]
        assert metadata_params["keywords"] == ["python", "javascript", "rust"]

    def test_parameters_with_mixed_types(self):
        """Test handling of mixed parameter types."""
        message = {
            "skill_id": "test_skill",
            "parameters": {
                "string_val": "hello",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "list_val": [1, 2, 3],
                "dict_val": {"nested": "value"}
            }
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata_params = params["message"]["metadata"]["params"]
        assert metadata_params["string_val"] == "hello"
        assert metadata_params["int_val"] == 42
        assert metadata_params["float_val"] == 3.14
        assert metadata_params["bool_val"] is True
        assert metadata_params["list_val"] == [1, 2, 3]
        assert metadata_params["dict_val"]["nested"] == "value"

    def test_user_message_in_parts(self):
        """Test that user_message is included in parts when provided."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"},
            "user_message": "Please search for test"
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        parts = params["message"]["parts"]
        # Should have user_message part + parameters part
        assert len(parts) >= 2
        assert any("Please search for test" in part.get("text", "") for part in parts)

    def test_message_kind_is_message(self):
        """Test that message kind is 'message'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        assert params["message"]["kind"] == "message"

    def test_message_role_is_user(self):
        """Test that message role is 'user'."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        assert params["message"]["role"] == "user"

    def test_message_id_is_unique(self):
        """Test that messageId is unique for each message."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"query": "test"}
        }

        req1 = self.client._build_action_request(message)
        req2 = self.client._build_action_request(message)

        params1 = json.loads(req1.a2a_message.params_json)
        params2 = json.loads(req2.a2a_message.params_json)

        msg_id1 = params1["message"]["messageId"]
        msg_id2 = params2["message"]["messageId"]

        assert msg_id1 != ""
        assert msg_id2 != ""
        assert msg_id1 != msg_id2

    def test_empty_parameters(self):
        """Test handling of empty parameters."""
        message = {
            "skill_id": "test_skill",
            "parameters": {}
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata_params = params["message"]["metadata"]["params"]
        assert metadata_params == {}

    def test_unicode_in_parameters(self):
        """Test handling of unicode characters in parameters."""
        message = {
            "skill_id": "test_skill",
            "parameters": {
                "text": "안녕하세요 こんにちは 你好 🎉"
            }
        }

        request = self.client._build_action_request(message)
        params = json.loads(request.a2a_message.params_json)

        metadata_params = params["message"]["metadata"]["params"]
        assert metadata_params["text"] == "안녕하세요 こんにちは 你好 🎉"

    def test_params_json_is_valid_json(self):
        """Test that params_json is always valid JSON."""
        message = {
            "skill_id": "test_skill",
            "parameters": {"complex": {"nested": [1, 2, {"deep": "value"}]}}
        }

        request = self.client._build_action_request(message)

        # Should not raise exception
        params = json.loads(request.a2a_message.params_json)
        assert isinstance(params, dict)

    def test_complete_a2a_format_example(self):
        """Test complete A2A format matches specification."""
        message = {
            "skill_id": "reddit_search_post",
            "parameters": {"query": "skincare"},
            "user_message": "Find skincare posts"
        }

        request = self.client._build_action_request(message)

        # Verify top-level A2A message
        assert request.a2a_message.jsonrpc == "2.0"
        assert request.a2a_message.method == "message/send"
        assert request.a2a_message.id != ""

        # Parse and verify complete structure
        params = json.loads(request.a2a_message.params_json)
        msg = params["message"]

        # Expected structure
        assert msg["kind"] == "message"
        assert msg["role"] == "user"
        assert msg["messageId"] != ""
        assert msg["metadata"]["skill"] == "reddit_search_post"
        assert msg["metadata"]["params"]["query"] == "skincare"
        assert len(msg["parts"]) >= 1
        assert all(p["kind"] == "text" for p in msg["parts"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
