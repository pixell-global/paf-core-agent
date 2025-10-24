"""Test conversation memory preservation in UPEE engine - simplified version."""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.core.execute import ExecutePhase
from src.schemas import ChatRequest, ConversationMessage
from src.llm_providers.base import LLMRequest, LLMResponse, LLMMessage
from src.settings import Settings


class TestConversationMemorySimple:
    """Simplified test suite for conversation memory preservation."""

    @pytest.mark.asyncio
    async def test_build_messages_array(self):
        """Test that _build_messages_array correctly builds messages from conversation history."""
        settings = Settings()
        execute_phase = ExecutePhase(settings)
        
        # Create a chat request with history
        chat_request = ChatRequest(
            message="What is the result multiplied by 2?",
            history=[
                ConversationMessage(role="user", content="What is 10 + 5?"),
                ConversationMessage(role="assistant", content="10 + 5 = 15"),
                ConversationMessage(role="user", content="Add 3 to that"),
                ConversationMessage(role="assistant", content="15 + 3 = 18")
            ]
        )
        
        # Build messages array
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the result multiplied by 2?"
        messages = execute_phase._build_messages_array(chat_request, system_prompt, user_prompt)
        
        # Verify the messages array
        assert len(messages) == 6  # 1 system + 4 history + 1 current
        assert messages[0].role == "system"
        assert messages[0].content == system_prompt
        assert messages[1].role == "user"
        assert messages[1].content == "What is 10 + 5?"
        assert messages[2].role == "assistant"
        assert messages[2].content == "10 + 5 = 15"
        assert messages[3].role == "user"
        assert messages[3].content == "Add 3 to that"
        assert messages[4].role == "assistant"
        assert messages[4].content == "15 + 3 = 18"
        assert messages[5].role == "user"
        assert messages[5].content == user_prompt

    @pytest.mark.asyncio
    async def test_llm_request_contains_messages(self):
        """Test that LLM request is created with messages array when conversation history exists."""
        settings = Settings()
        execute_phase = ExecutePhase(settings)
        
        # Mock the LLM manager
        mock_llm_manager = AsyncMock()
        
        async def mock_stream():
            yield LLMResponse(
                content="Test response",
                model="test-model",
                provider="test",
                is_complete=True
            )
        
        mock_llm_manager.stream_completion.return_value = mock_stream()
        execute_phase.llm_manager = mock_llm_manager
        
        # Create request with history
        chat_request = ChatRequest(
            message="What's next?",
            history=[
                ConversationMessage(role="user", content="Hello"),
                ConversationMessage(role="assistant", content="Hi there!")
            ]
        )
        
        # Execute the streaming
        events = []
        async for event in execute_phase._execute_llm_streaming(
            prompt="Test prompt",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            request_id="test-123",
            external_results={},
            request=chat_request
        ):
            events.append(event)
        
        # Verify LLM was called with messages
        mock_llm_manager.stream_completion.assert_called_once()
        llm_request = mock_llm_manager.stream_completion.call_args[0][0]
        
        # Check that the request has messages instead of just a prompt
        assert hasattr(llm_request, 'messages')
        assert isinstance(llm_request.messages, list)
        assert len(llm_request.messages) >= 3  # At least system + history + current
        
        # Verify message types
        for msg in llm_request.messages:
            assert isinstance(msg, LLMMessage)
            assert hasattr(msg, 'role')
            assert hasattr(msg, 'content')

    @pytest.mark.asyncio 
    async def test_empty_history_still_uses_messages(self):
        """Test that even with empty history, messages array is used."""
        settings = Settings()
        execute_phase = ExecutePhase(settings)
        
        # Create request with no history
        chat_request = ChatRequest(
            message="Hello",
            history=[]
        )
        
        # Build messages array
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello"
        messages = execute_phase._build_messages_array(chat_request, system_prompt, user_prompt)
        
        # Should still have system and user messages
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == system_prompt
        assert messages[1].role == "user"
        assert messages[1].content == user_prompt