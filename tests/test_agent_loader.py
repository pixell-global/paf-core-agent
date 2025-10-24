"""Tests for agent configuration loader."""

import json
import os
import pytest
from pathlib import Path
from src.config.agent_loader import (
    load_agents,
    get_agent_by_id,
    get_agent_by_name,
    get_all_agents,
    clear_cache,
    AgentConfig
)


@pytest.fixture(autouse=True)
def clear_agent_cache():
    """Clear agent cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def mock_env_agents(monkeypatch):
    """Mock A2A_AGENT_APPS environment variable."""
    env_agents = [
        {
            "agent_app_id": "env-agent-001",
            "name": "Env Test Agent",
            "endpoint_url": "https://example.com/env-agent",
            "protocol": "grpc",
            "description": "Agent from environment variable",
            "capabilities": ["test capability from env"],
            "example_queries": ["env test query"]
        }
    ]
    monkeypatch.setenv("A2A_AGENT_APPS", json.dumps(env_agents))
    return env_agents


def test_load_agents_from_file():
    """Test loading agents from agents_config.json file."""
    agents = load_agents()

    assert len(agents) > 0, "Should load at least one agent from file"

    # Check Vivid Commenter is loaded
    vivid = next((a for a in agents if a.name == "Vivid Commenter"), None)
    assert vivid is not None, "Vivid Commenter should be loaded"
    assert vivid.agent_app_id == "4906eeb7-9959-414e-84c6-f2445822ebe4"
    assert vivid.protocol == "grpc"
    assert len(vivid.capabilities) > 0, "Should have capabilities"
    assert len(vivid.example_queries) > 0, "Should have example queries"


def test_load_agents_from_env(mock_env_agents, monkeypatch):
    """Test loading agents from A2A_AGENT_APPS environment variable."""
    # Clear file path so only env is loaded
    agents = load_agents()

    # Should include env agent
    env_agent = next((a for a in agents if a.agent_app_id == "env-agent-001"), None)
    assert env_agent is not None, "Env agent should be loaded"
    assert env_agent.name == "Env Test Agent"
    assert env_agent.endpoint == "https://example.com/env-agent"


def test_load_agents_merges_sources(mock_env_agents):
    """Test that agents from both file and env are merged."""
    agents = load_agents()

    # Should have agents from both sources
    agent_names = [a.name for a in agents]

    assert "Vivid Commenter" in agent_names, "Should have agent from file"
    assert "Env Test Agent" in agent_names, "Should have agent from env"


def test_load_agents_avoids_duplicates(monkeypatch):
    """Test that duplicate agents (same agent_app_id) are not loaded twice."""
    # Create env config with same agent_app_id as file
    duplicate_agent = [
        {
            "agent_app_id": "4906eeb7-9959-414e-84c6-f2445822ebe4",  # Same as Vivid
            "name": "Duplicate Vivid",
            "endpoint_url": "https://different.com",
            "protocol": "http",
            "description": "Duplicate agent",
            "capabilities": [],
            "example_queries": []
        }
    ]
    monkeypatch.setenv("A2A_AGENT_APPS", json.dumps(duplicate_agent))

    agents = load_agents()

    # Should only have one agent with this ID
    vivid_agents = [a for a in agents if a.agent_app_id == "4906eeb7-9959-414e-84c6-f2445822ebe4"]
    assert len(vivid_agents) == 1, "Should not load duplicate agent_app_id"
    assert vivid_agents[0].name == "Vivid Commenter", "Should keep file version (loaded first)"


def test_agent_config_caching():
    """Test that agents are cached after first load."""
    # First load
    agents1 = load_agents()

    # Second load (should use cache)
    agents2 = load_agents()

    # Should be same object (cached)
    assert agents1 is agents2, "Should return cached agents"


def test_agent_config_force_reload():
    """Test that force_reload bypasses cache."""
    # First load
    agents1 = load_agents()

    # Force reload
    agents2 = load_agents(force_reload=True)

    # Should be different objects (reloaded)
    assert agents1 is not agents2, "Should reload agents when force_reload=True"


def test_get_agent_by_id():
    """Test getting agent by agent_app_id."""
    agent = get_agent_by_id("4906eeb7-9959-414e-84c6-f2445822ebe4")

    assert agent is not None, "Should find Vivid Commenter by ID"
    assert agent.name == "Vivid Commenter"


def test_get_agent_by_id_not_found():
    """Test getting agent with non-existent ID."""
    agent = get_agent_by_id("non-existent-id")

    assert agent is None, "Should return None for non-existent ID"


def test_get_agent_by_name():
    """Test getting agent by name."""
    agent = get_agent_by_name("Vivid Commenter")

    assert agent is not None, "Should find agent by name"
    assert agent.agent_app_id == "4906eeb7-9959-414e-84c6-f2445822ebe4"


def test_get_agent_by_name_case_insensitive():
    """Test getting agent by name (case-insensitive)."""
    agent = get_agent_by_name("vivid commenter", case_sensitive=False)

    assert agent is not None, "Should find agent with case-insensitive match"
    assert agent.name == "Vivid Commenter"


def test_get_agent_by_name_case_sensitive():
    """Test getting agent by name (case-sensitive)."""
    agent = get_agent_by_name("vivid commenter", case_sensitive=True)

    assert agent is None, "Should not find agent with wrong case when case_sensitive=True"


def test_get_all_agents():
    """Test getting all agents."""
    agents = get_all_agents()

    assert len(agents) > 0, "Should return all loaded agents"
    assert all(isinstance(a, AgentConfig) for a in agents), "All should be AgentConfig instances"


def test_clear_cache():
    """Test clearing the agent cache."""
    # Load agents
    agents1 = load_agents()

    # Clear cache
    clear_cache()

    # Load again
    agents2 = load_agents()

    # Should be different objects (cache was cleared)
    assert agents1 is not agents2, "Should reload after cache clear"


def test_agent_config_to_dict():
    """Test AgentConfig to_dict conversion."""
    agent = AgentConfig(
        agent_app_id="test-001",
        name="Test Agent",
        endpoint="https://test.com",
        protocol="grpc",
        description="Test description",
        capabilities=["cap1", "cap2"],
        example_queries=["query1", "query2"]
    )

    agent_dict = agent.to_dict()

    assert agent_dict["agent_app_id"] == "test-001"
    assert agent_dict["name"] == "Test Agent"
    assert agent_dict["capabilities"] == ["cap1", "cap2"]
    assert agent_dict["example_queries"] == ["query1", "query2"]


def test_agent_config_from_dict():
    """Test AgentConfig from_dict creation."""
    data = {
        "agent_app_id": "test-002",
        "name": "Test Agent 2",
        "endpoint": "https://test2.com",
        "protocol": "http",
        "description": "Test description 2",
        "capabilities": ["cap3"],
        "example_queries": ["query3"]
    }

    agent = AgentConfig.from_dict(data)

    assert agent.agent_app_id == "test-002"
    assert agent.name == "Test Agent 2"
    assert agent.endpoint == "https://test2.com"
    assert agent.protocol == "http"
    assert agent.capabilities == ["cap3"]


def test_agent_config_from_dict_with_endpoint_url():
    """Test AgentConfig from_dict with endpoint_url instead of endpoint."""
    data = {
        "agent_app_id": "test-003",
        "name": "Test Agent 3",
        "endpoint_url": "https://test3.com",  # Use endpoint_url instead
        "protocol": "grpc",
        "description": "Test description 3",
        "capabilities": [],
        "example_queries": []
    }

    agent = AgentConfig.from_dict(data)

    assert agent.endpoint == "https://test3.com", "Should map endpoint_url to endpoint"


def test_invalid_json_in_env(monkeypatch, capsys):
    """Test handling of invalid JSON in A2A_AGENT_APPS."""
    monkeypatch.setenv("A2A_AGENT_APPS", "invalid json {{{")

    agents = load_agents()

    # Should still load agents from file (not crash)
    assert len(agents) > 0, "Should load file agents despite env error"

    # Check that error was logged (structlog writes to stdout)
    captured = capsys.readouterr()
    assert "Failed to parse A2A_AGENT_APPS JSON" in captured.out or "Expecting value" in captured.out
