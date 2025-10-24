"""Agent configuration loader - loads agents from static config and environment."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Agent configuration with all metadata for AI-powered routing."""

    agent_app_id: str
    name: str
    endpoint: str
    protocol: str
    description: str
    capabilities: List[str]
    example_queries: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_app_id": self.agent_app_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "description": self.description,
            "capabilities": self.capabilities,
            "example_queries": self.example_queries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from dictionary."""
        return cls(
            agent_app_id=data.get("agent_app_id", ""),
            name=data.get("name", ""),
            endpoint=data.get("endpoint", data.get("endpoint_url", "")),
            protocol=data.get("protocol", "grpc"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            example_queries=data.get("example_queries", [])
        )


# Module-level cache for loaded agents
_agents_cache: Optional[List[AgentConfig]] = None


def load_agents(force_reload: bool = False) -> List[AgentConfig]:
    """
    Load agents from both JSON file and environment variable.

    Sources (in priority order):
    1. agents_config.json file (if exists)
    2. A2A_AGENT_APPS environment variable (if set)

    Results are cached. Use force_reload=True to bypass cache.

    Args:
        force_reload: If True, reload from sources even if cached

    Returns:
        List of AgentConfig objects
    """
    global _agents_cache

    # Return cached agents if available and not forcing reload
    if _agents_cache is not None and not force_reload:
        logger.debug("Returning cached agents", count=len(_agents_cache))
        return _agents_cache

    agents: List[AgentConfig] = []

    # Load from JSON file
    config_path = Path(__file__).parent.parent.parent / "agents_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)

            file_agents = data.get("agents", []) if isinstance(data, dict) else data

            for agent_data in file_agents:
                try:
                    agent = AgentConfig.from_dict(agent_data)
                    agents.append(agent)
                    logger.debug(
                        "Loaded agent from file",
                        agent_name=agent.name,
                        agent_id=agent.agent_app_id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to parse agent from file",
                        agent_data=agent_data,
                        error=str(e)
                    )

            logger.info(
                "Loaded agents from file",
                file=str(config_path),
                count=len(agents)
            )
        except Exception as e:
            logger.error(
                "Failed to load agents from file",
                file=str(config_path),
                error=str(e)
            )
    else:
        logger.debug("No agents_config.json file found", path=str(config_path))

    # Load from environment variable
    env_config = os.getenv("A2A_AGENT_APPS")
    if env_config:
        try:
            env_agents_data = json.loads(env_config)

            for agent_data in env_agents_data:
                try:
                    # Check if already loaded from file (avoid duplicates)
                    agent_app_id = agent_data.get("agent_app_id")
                    if agent_app_id and any(a.agent_app_id == agent_app_id for a in agents):
                        logger.debug(
                            "Skipping duplicate agent from env",
                            agent_id=agent_app_id
                        )
                        continue

                    agent = AgentConfig.from_dict(agent_data)
                    agents.append(agent)
                    logger.debug(
                        "Loaded agent from env",
                        agent_name=agent.name,
                        agent_id=agent.agent_app_id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to parse agent from env",
                        agent_data=agent_data,
                        error=str(e)
                    )

            logger.info("Loaded agents from A2A_AGENT_APPS env", count=len(env_agents_data))
        except json.JSONDecodeError as e:
            logger.error("Failed to parse A2A_AGENT_APPS JSON", error=str(e))
        except Exception as e:
            logger.error("Failed to load agents from env", error=str(e))
    else:
        logger.debug("No A2A_AGENT_APPS environment variable set")

    # Cache the results
    _agents_cache = agents

    logger.info(
        "Agent configuration loaded",
        total_agents=len(agents),
        agent_names=[a.name for a in agents]
    )

    return agents


def get_agent_by_id(agent_app_id: str) -> Optional[AgentConfig]:
    """
    Get agent configuration by agent_app_id.

    Args:
        agent_app_id: The agent's unique identifier

    Returns:
        AgentConfig if found, None otherwise
    """
    agents = load_agents()

    for agent in agents:
        if agent.agent_app_id == agent_app_id:
            return agent

    return None


def get_agent_by_name(name: str, case_sensitive: bool = False) -> Optional[AgentConfig]:
    """
    Get agent configuration by name.

    Args:
        name: The agent's name
        case_sensitive: If True, match name exactly. If False, case-insensitive.

    Returns:
        AgentConfig if found, None otherwise
    """
    agents = load_agents()

    for agent in agents:
        if case_sensitive:
            if agent.name == name:
                return agent
        else:
            if agent.name.lower() == name.lower():
                return agent

    return None


def get_all_agents() -> List[AgentConfig]:
    """
    Get all loaded agent configurations.

    Returns:
        List of all AgentConfig objects
    """
    return load_agents()


def clear_cache() -> None:
    """Clear the agents cache, forcing reload on next call."""
    global _agents_cache
    _agents_cache = None
    logger.debug("Agent cache cleared")
