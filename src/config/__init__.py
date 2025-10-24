"""Configuration management for PAF Core Agent."""

from src.config.agent_loader import load_agents, get_agent_by_id, get_all_agents

__all__ = ["load_agents", "get_agent_by_id", "get_all_agents"]
