"""E2E Test for LangGraph AI-Native Routing to Vivid Commenter.

This test validates the complete flow:
1. User sends Reddit query: "find me 10 subreddits related to ai"
2. LangGraph Understand node extracts intent
3. LangGraph Routing node decides to route to Vivid Commenter
4. Execute Agent node calls Vivid Commenter via gRPC
5. Evaluate node assesses response quality
6. Final response is returned
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.langgraph_upee import execute_upee_graph, UPEEInput
from src.config.agent_loader import AgentConfig
from src.llm_providers import LLMResponse, LLMProviderManager
from src.settings import Settings


@pytest.fixture
def mock_settings():
    """Mock Settings with LangGraph enabled."""
    settings = Mock(spec=Settings)
    settings.default_model = "gpt-4o"
    settings.a2a_timeout = 30
    settings.use_langgraph_upee = True
    return settings


@pytest.fixture
def mock_llm_manager():
    """Mock LLM Provider Manager."""
    manager = Mock(spec=LLMProviderManager)
    manager.get_completion = AsyncMock()
    return manager


@pytest.fixture
def vivid_agent():
    """Vivid Commenter agent configuration."""
    return AgentConfig(
        agent_app_id="4906eeb7-9959-414e-84c6-f2445822ebe4",
        name="Vivid Commenter",
        endpoint="https://par.pixell.global/agents/4906eeb7-9959-414e-84c6-f2445822ebe4",
        protocol="grpc",
        description="AI-powered Reddit commenter for marketing automation. Can search Reddit, find subreddits by keywords, crawl posts, and create comments.",
        capabilities=[
            "Find and analyze relevant subreddits for any topic or niche",
            "Research Reddit communities, their rules, culture, and engagement patterns",
            "Generate authentic, contextually appropriate Reddit comments and posts"
        ],
        example_queries=[
            "find subreddits about AI and machine learning",
            "what are good subreddits for marketing my SaaS product?"
        ]
    )


@pytest.mark.asyncio
async def test_e2e_reddit_query_routes_to_vivid_commenter(
    mock_settings,
    mock_llm_manager,
    vivid_agent
):
    """
    E2E Test: Reddit subreddit query routes to Vivid Commenter via LangGraph.

    This is the CRITICAL test that validates the entire AI-native routing flow.
    """

    # Mock LLM responses for each phase
    mock_llm_manager.get_completion.side_effect = [
        # Phase 1: Understand Node - AI extracts intent
        LLMResponse(
            content='{"intent_summary": "User wants to find AI-related subreddits", "primary_intent": "search", "topics": ["reddit", "subreddits", "artificial intelligence"], "entities": ["AI"], "complexity": "simple", "requires_specialized_knowledge": true, "domain": "reddit", "keywords": ["find", "subreddit", "ai", "reddit"]}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Phase 2: Routing Node - AI decides to route to Vivid Commenter
        LLMResponse(
            content='{"decision": "agent", "agent_id": "4906eeb7-9959-414e-84c6-f2445822ebe4", "confidence": 0.98, "reasoning": "This is a Reddit subreddit discovery query which matches the Vivid Commenter agent\'s core capability of finding and analyzing subreddits."}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Phase 3: Evaluate Node - AI assesses response quality
        LLMResponse(
            content='{"quality_score": 0.92, "feedback": "Excellent subreddit recommendations with good variety", "needs_refinement": false, "suggestions": ""}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        )
    ]

    # Mock gRPC client for Vivid Commenter
    mock_grpc_client = Mock()
    mock_grpc_client.send_message = AsyncMock(return_value={
        "success": True,
        "data": {
            "content": """Here are 10 AI-related subreddits:

1. r/MachineLearning - Research and discussions on machine learning
2. r/artificial - Artificial intelligence news and discussions
3. r/learnmachinelearning - For beginners learning ML
4. r/deeplearning - Deep learning research and applications
5. r/LanguageTechnology - NLP and computational linguistics
6. r/computervision - Computer vision and image processing
7. r/reinforcementlearning - RL algorithms and research
8. r/datascience - Data science including ML applications
9. r/MLQuestions - Q&A for machine learning
10. r/ArtificialInteligence - General AI discussions

These subreddits cover various aspects of AI from research to practical applications."""
        }
    })

    # Patch agent loader and gRPC client
    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
        with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_grpc_client):
            # Execute the UPEE graph with Reddit query
            user_input: UPEEInput = {
                "user_message": "find me 10 subreddits related to ai",
                "request_id": "e2e-reddit-test-001"
            }

            output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    # Validate: Routing decision was "agent"
    assert output["routing_decision"] == "agent", \
        f"Expected routing to agent, got {output['routing_decision']}"

    # Validate: Selected agent was Vivid Commenter
    assert output["selected_agent_name"] == "Vivid Commenter", \
        f"Expected Vivid Commenter, got {output['selected_agent_name']}"

    # Validate: Response contains subreddit recommendations
    assert "MachineLearning" in output["response"], \
        "Response should contain subreddit recommendations from Vivid Commenter"
    assert "r/" in output["response"], \
        "Response should contain Reddit subreddit format (r/)"

    # Validate: Quality score is high
    assert output["quality_score"] >= 0.9, \
        f"Expected high quality score, got {output['quality_score']}"

    # Validate: No errors
    assert output["error"] is None, \
        f"Expected no errors, got {output['error']}"

    # Validate: gRPC client was called with correct message
    mock_grpc_client.send_message.assert_called_once()
    call_args = mock_grpc_client.send_message.call_args[0][0]
    assert "find me 10 subreddits related to ai" in str(call_args), \
        "gRPC call should contain original user message"


@pytest.mark.asyncio
async def test_e2e_general_query_routes_to_core(
    mock_settings,
    mock_llm_manager
):
    """
    E2E Test: General knowledge query routes to PAF Core (not agent).

    This validates that non-Reddit queries still go to core execution.
    """

    # Mock LLM responses
    mock_llm_manager.get_completion.side_effect = [
        # Understand Node
        LLMResponse(
            content='{"intent_summary": "User wants to know the capital of France", "primary_intent": "question", "topics": ["geography"], "entities": ["France"], "complexity": "simple", "requires_specialized_knowledge": false, "domain": "general", "keywords": ["capital", "france"]}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Routing Node - decides core
        LLMResponse(
            content='{"decision": "core", "agent_id": null, "confidence": 0.95, "reasoning": "General knowledge query, no specialized agent needed"}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Execute Core Node
        LLMResponse(
            content="Paris is the capital and most populous city of France.",
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        ),
        # Evaluate Node
        LLMResponse(
            content='{"quality_score": 0.95, "feedback": "Accurate and concise answer", "needs_refinement": false, "suggestions": ""}',
            model="gpt-4o",
            provider="openai",
            finish_reason="stop"
        )
    ]

    # Patch agent loader
    with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[]):
        user_input: UPEEInput = {
            "user_message": "What is the capital of France?",
            "request_id": "e2e-core-test-001"
        }

        output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

    # Validate: Routing decision was "core"
    assert output["routing_decision"] == "core", \
        f"Expected routing to core, got {output['routing_decision']}"

    # Validate: No agent was selected
    assert output["selected_agent_name"] is None, \
        f"Expected no agent selection, got {output['selected_agent_name']}"

    # Validate: Response is about Paris
    assert "Paris" in output["response"], \
        "Response should contain answer about Paris"

    # Validate: No errors
    assert output["error"] is None


@pytest.mark.asyncio
async def test_e2e_vivid_commenter_variants(
    mock_settings,
    mock_llm_manager,
    vivid_agent
):
    """
    E2E Test: Various Reddit-related queries all route to Vivid Commenter.

    Tests multiple variations of Reddit queries.
    """

    test_queries = [
        "find me subreddits about machine learning",
        "what are good reddit communities for AI",
        "where can I post about artificial intelligence on reddit",
        "show me AI subreddits"
    ]

    for query in test_queries:
        # Reset mock
        mock_llm_manager.get_completion.reset_mock()

        # Mock LLM responses
        mock_llm_manager.get_completion.side_effect = [
            # Understand
            LLMResponse(
                content=f'{{"intent_summary": "User wants Reddit subreddits", "primary_intent": "search", "topics": ["reddit"], "entities": [], "complexity": "simple", "requires_specialized_knowledge": true, "domain": "reddit", "keywords": ["reddit", "subreddit"]}}',
                model="gpt-4o",
                provider="openai",
                finish_reason="stop"
            ),
            # Routing - should always route to agent
            LLMResponse(
                content='{"decision": "agent", "agent_id": "4906eeb7-9959-414e-84c6-f2445822ebe4", "confidence": 0.95, "reasoning": "Reddit query"}',
                model="gpt-4o",
                provider="openai",
                finish_reason="stop"
            ),
            # Evaluate
            LLMResponse(
                content='{"quality_score": 0.90, "feedback": "Good", "needs_refinement": false, "suggestions": ""}',
                model="gpt-4o",
                provider="openai",
                finish_reason="stop"
            )
        ]

        # Mock gRPC client
        mock_grpc_client = Mock()
        mock_grpc_client.send_message = AsyncMock(return_value={
            "success": True,
            "data": {"content": "Here are some subreddits..."}
        })

        # Execute
        with patch('src.langgraph_upee.nodes.routing.load_agents', return_value=[vivid_agent]):
            with patch('src.langgraph_upee.nodes.execute_agent.GrpcA2AClient', return_value=mock_grpc_client):
                user_input: UPEEInput = {
                    "user_message": query,
                    "request_id": f"e2e-variant-{hash(query)}"
                }

                output = await execute_upee_graph(user_input, mock_settings, mock_llm_manager)

        # Validate each query routes to agent
        assert output["routing_decision"] == "agent", \
            f"Query '{query}' should route to agent, got {output['routing_decision']}"
        assert output["selected_agent_name"] == "Vivid Commenter", \
            f"Query '{query}' should select Vivid Commenter"
