"""Test conversation memory preservation in UPEE engine."""
import pytest
import uuid
from unittest.mock import Mock, patch
from src.core.upee_engine import UPEEEngine
from src.schemas import ChatRequest, ConversationMessage
from src.llm_providers.base import LLMRequest, LLMResponse
from src.settings import Settings


class TestConversationMemory:
    """Test suite for conversation memory preservation."""

    @pytest.fixture
    def upee_engine(self):
        """Create UPEE engine instance."""
        settings = Settings()
        return UPEEEngine(settings)

    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager."""
        mock = Mock()
        
        async def mock_stream():
            yield LLMResponse(
                content="Test response",
                model="test-model",
                provider="test",
                token_count=20,
                is_complete=True
            )
        
        mock.stream_completion.return_value = mock_stream()
        return mock

    @pytest.mark.asyncio
    async def test_conversation_history_is_preserved_in_llm_request(self, upee_engine, mock_llm_manager):
        """Test that conversation history is included in LLM requests."""
        # Create a conversation with history
        chat_request = ChatRequest(
            message="What is the result multiplied by 2?",
            history=[
                ConversationMessage(role="user", content="What is 10 + 5?"),
                ConversationMessage(role="assistant", content="10 + 5 = 15"),
                ConversationMessage(role="user", content="Add 3 to that"),
                ConversationMessage(role="assistant", content="15 + 3 = 18")
            ]
        )

        with patch.object(upee_engine.execute_phase, 'llm_manager', mock_llm_manager):
            # Process the request through execute phase directly
            request_id = str(uuid.uuid4())
            understanding = await upee_engine.understand_phase.process(chat_request, request_id)
            plan = await upee_engine.plan_phase.process(chat_request, request_id, understanding)
            
            # Collect the stream to trigger the LLM call
            events = []
            async for event in upee_engine.execute_phase.process(chat_request, request_id, understanding, plan):
                events.append(event)

            # Verify LLM was called with messages including history
            mock_llm_manager.stream_completion.assert_called_once()
            llm_request = mock_llm_manager.stream_completion.call_args[0][0]

            # Check that messages array includes conversation history
            assert hasattr(llm_request, 'messages'), "LLMRequest should have messages attribute"
            assert len(llm_request.messages) >= 5, "Should include all history messages plus current"
            
            # Verify message order and content
            messages = llm_request.messages
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "What is 10 + 5?"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "10 + 5 = 15"
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "What is the result multiplied by 2?"

    @pytest.mark.asyncio
    async def test_assistant_can_reference_previous_context(self, upee_engine, mock_llm_manager):
        """Test that assistant can understand references to previous messages."""
        # Simulate a conversation where user refers back
        chat_request = ChatRequest(
            message="Double that number",
            history=[
                ConversationMessage(role="user", content="What is 25 squared?"),
                ConversationMessage(role="assistant", content="25 squared is 625")
            ]
        )

        with patch.object(upee_engine, 'llm_manager', mock_llm_manager):
            understanding = upee_engine.understand(chat_request)
            plan = upee_engine.plan(understanding)
            upee_engine.execute(plan, understanding, chat_request)

            # Verify the LLM received context to understand "that number" refers to 625
            llm_request = mock_llm_manager.generate.call_args[0][0]
            assert len(llm_request.messages) >= 3
            assert "625" in llm_request.messages[1]["content"]

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_pronouns(self, upee_engine, mock_llm_manager):
        """Test handling of pronouns and references across multiple turns."""
        chat_request = ChatRequest(
            message="What about its square root?",
            history=[
                ConversationMessage(role="user", content="Tell me about the number 144"),
                ConversationMessage(role="assistant", content="144 is a perfect square (12 × 12)"),
                ConversationMessage(role="user", content="Is it divisible by 3?"),
                ConversationMessage(role="assistant", content="Yes, 144 is divisible by 3. 144 ÷ 3 = 48")
            ]
        )

        with patch.object(upee_engine, 'llm_manager', mock_llm_manager):
            understanding = upee_engine.understand(chat_request)
            plan = upee_engine.plan(understanding)
            upee_engine.execute(plan, understanding, chat_request)

            llm_request = mock_llm_manager.generate.call_args[0][0]
            # Verify all history is preserved so "its" can be resolved to 144
            assert len(llm_request.messages) >= 5
            assert "144" in llm_request.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_conversation_with_code_context(self, upee_engine, mock_llm_manager):
        """Test memory preservation when discussing code across messages."""
        chat_request = ChatRequest(
            message="Now make it async",
            history=[
                ConversationMessage(role="user", content="Write a Python function to fetch data from an API"),
                ConversationMessage(role="assistant", content="def fetch_data(url):\n    response = requests.get(url)\n    return response.json()"),
                ConversationMessage(role="user", content="Add error handling"),
                ConversationMessage(role="assistant", content="def fetch_data(url):\n    try:\n        response = requests.get(url)\n        response.raise_for_status()\n        return response.json()\n    except requests.RequestException as e:\n        return {'error': str(e)}")
            ]
        )

        with patch.object(upee_engine, 'llm_manager', mock_llm_manager):
            understanding = upee_engine.understand(chat_request)
            plan = upee_engine.plan(understanding)
            upee_engine.execute(plan, understanding, chat_request)

            llm_request = mock_llm_manager.generate.call_args[0][0]
            # Verify the code context is preserved
            assert len(llm_request.messages) >= 5
            assert "fetch_data" in llm_request.messages[1]["content"]
            assert "fetch_data" in llm_request.messages[3]["content"]

    @pytest.mark.asyncio
    async def test_empty_conversation_history(self, upee_engine, mock_llm_manager):
        """Test handling of requests with no conversation history."""
        chat_request = ChatRequest(
            message="Hello, how are you?",
            history=[]
        )

        with patch.object(upee_engine, 'llm_manager', mock_llm_manager):
            understanding = upee_engine.understand(chat_request)
            plan = upee_engine.plan(understanding)
            upee_engine.execute(plan, understanding, chat_request)

            llm_request = mock_llm_manager.generate.call_args[0][0]
            assert hasattr(llm_request, 'messages')
            assert len(llm_request.messages) >= 1
            assert llm_request.messages[-1]["content"] == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_conversation_history_with_file_context(self, upee_engine, mock_llm_manager):
        """Test that both conversation history and file context are preserved."""
        chat_request = ChatRequest(
            message="What did I ask about earlier regarding this file?",
            history=[
                ConversationMessage(role="user", content="Can you explain the main function in app.py?"),
                ConversationMessage(role="assistant", content="The main function initializes the FastAPI app..."),
            ],
            files=[{"name": "app.py", "content": "def main():\n    pass", "metadata": {}}]
        )

        with patch.object(upee_engine, 'llm_manager', mock_llm_manager):
            understanding = upee_engine.understand(chat_request)
            plan = upee_engine.plan(understanding)
            upee_engine.execute(plan, understanding, chat_request)

            llm_request = mock_llm_manager.generate.call_args[0][0]
            # Should have both history and current message
            assert len(llm_request.messages) >= 3
            # File context should still be included (either in system message or user message)
            assert any("app.py" in str(msg) for msg in [llm_request.messages, llm_request.system_prompt, llm_request.prompt] if msg)