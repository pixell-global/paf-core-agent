"""Agent Client Pool - Manages multiple A2A agent clients."""

from typing import Dict, Optional
from src.utils.agent_client import AgentClient
from src.agents.hybrid_agent_client import HybridAgentClient
from src.agents.agent_app_registry import AgentAppRegistry, AgentAppInfo
from src.utils.logging_config import get_logger

logger = get_logger("agent_client_pool")


class AgentClientPool:
    """Pool of A2A agent clients for multiple agent apps."""

    def __init__(self, registry: AgentAppRegistry):
        """Initialize the client pool.

        Args:
            registry: Agent app registry with discovered agents
        """
        self.registry = registry
        self._clients: Dict[str, AgentClient] = {}
        self._initialized = False

    def initialize(self):
        """Initialize clients for all enabled agents."""
        if self._initialized:
            logger.warning("Client pool already initialized")
            return

        enabled_agents = self.registry.get_enabled_agents()

        for agent in enabled_agents:
            try:
                # Use HybridAgentClient which auto-detects HTTP vs gRPC
                client = HybridAgentClient(agent)
                self._clients[agent.agent_app_id] = client

                logger.info(
                    f"Initialized A2A client for agent: {agent.name}",
                    agent_app_id=agent.agent_app_id,
                    endpoint=agent.endpoint_url,
                    protocol=client.get_protocol()
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize client for agent: {agent.name}",
                    agent_app_id=agent.agent_app_id,
                    error=str(e)
                )

        self._initialized = True
        logger.info(f"Client pool initialized with {len(self._clients)} clients")

    def get_client(self, agent_app_id: str) -> Optional[AgentClient]:
        """Get client for specific agent app.

        Args:
            agent_app_id: Agent app ID

        Returns:
            AgentClient instance or None if not found
        """
        if not self._initialized:
            logger.warning("Client pool not initialized, initializing now")
            self.initialize()

        client = self._clients.get(agent_app_id)

        if not client:
            logger.warning(
                f"Client not found for agent_app_id: {agent_app_id}",
                available_clients=list(self._clients.keys())
            )

        return client

    def has_client(self, agent_app_id: str) -> bool:
        """Check if client exists for agent app.

        Args:
            agent_app_id: Agent app ID

        Returns:
            True if client exists, False otherwise
        """
        return agent_app_id in self._clients

    def get_agent_info(self, agent_app_id: str) -> Optional[AgentAppInfo]:
        """Get agent info from registry.

        Args:
            agent_app_id: Agent app ID

        Returns:
            AgentAppInfo or None if not found
        """
        return self.registry.get_agent(agent_app_id)

    def refresh_clients(self):
        """Refresh clients based on current registry state.

        This will:
        - Add clients for newly enabled agents
        - Remove clients for disabled agents
        """
        enabled_agents = self.registry.get_enabled_agents()
        enabled_ids = {agent.agent_app_id for agent in enabled_agents}

        # Remove clients for disabled agents
        for agent_id in list(self._clients.keys()):
            if agent_id not in enabled_ids:
                del self._clients[agent_id]
                logger.info(f"Removed client for disabled agent: {agent_id}")

        # Add clients for newly enabled agents
        for agent in enabled_agents:
            if agent.agent_app_id not in self._clients:
                try:
                    # Use HybridAgentClient which auto-detects HTTP vs gRPC
                    client = HybridAgentClient(agent)
                    self._clients[agent.agent_app_id] = client

                    logger.info(
                        f"Added client for new agent: {agent.name}",
                        agent_app_id=agent.agent_app_id,
                        protocol=client.get_protocol()
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to add client for agent: {agent.name}",
                        agent_app_id=agent.agent_app_id,
                        error=str(e)
                    )

        logger.info(f"Client pool refreshed, now has {len(self._clients)} clients")

    def close_all(self):
        """Close all clients and cleanup resources."""
        for agent_id, client in self._clients.items():
            try:
                # AgentClient doesn't have explicit close method currently
                # This is a placeholder for future cleanup logic
                logger.debug(f"Closing client for agent: {agent_id}")

            except Exception as e:
                logger.error(
                    f"Error closing client for agent: {agent_id}",
                    error=str(e)
                )

        self._clients.clear()
        self._initialized = False
        logger.info("All clients closed")

    def get_client_count(self) -> int:
        """Get the number of active clients.

        Returns:
            Number of active clients
        """
        return len(self._clients)

    def get_all_agent_ids(self) -> list[str]:
        """Get list of all agent app IDs with clients.

        Returns:
            List of agent app IDs
        """
        return list(self._clients.keys())
