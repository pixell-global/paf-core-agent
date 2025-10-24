"""Hybrid A2A client that supports both HTTP and gRPC protocols.

Automatically selects the appropriate protocol based on agent configuration:
- gRPC for PAR-deployed agents (par.pixell.global)
- HTTP for local/direct agents
"""

from typing import Dict, Any, Optional
from src.utils.agent_client import AgentClient
from src.agents.grpc_a2a_client import GrpcA2AClient
from src.agents.agent_app_registry import AgentAppInfo
from src.utils.logging_config import get_logger

logger = get_logger("hybrid_agent_client")


class HybridAgentClient:
    """Smart A2A client that uses HTTP or gRPC based on agent configuration.

    This client automatically detects whether to use:
    - gRPC: For PAR-deployed agents (requires correct agent ID in path)
    - HTTP: For local agents or agents with direct HTTP endpoints
    """

    def __init__(self, agent_info: AgentAppInfo, force_grpc: bool = False):
        """Initialize hybrid client.

        Args:
            agent_info: Agent app information from registry
            force_grpc: Force gRPC protocol (for testing)
        """
        self.agent_info = agent_info
        self.force_grpc = force_grpc

        # Determine protocol
        self.use_grpc = self._should_use_grpc()

        # Initialize appropriate client
        if self.use_grpc:
            self._client = GrpcA2AClient(
                agent_app_id=agent_info.agent_app_id,
                endpoint_url=agent_info.endpoint_url,
                timeout=30.0
            )
            logger.info(
                f"Using gRPC client for agent: {agent_info.name}",
                agent_id=agent_info.agent_app_id,
                endpoint=agent_info.endpoint_url
            )
        else:
            self._client = AgentClient(agent_info.endpoint_url)
            logger.info(
                f"Using HTTP client for agent: {agent_info.name}",
                agent_id=agent_info.agent_app_id,
                endpoint=agent_info.endpoint_url
            )

    def _should_use_grpc(self) -> bool:
        """Determine if gRPC should be used.

        Uses gRPC for:
        - PAR-deployed agents (par.pixell.global)
        - Agents with protocol="grpc"
        - When force_grpc is True

        Returns:
            True if gRPC should be used, False for HTTP
        """
        # Force gRPC if requested
        if self.force_grpc:
            return True

        # Check if agent has explicit gRPC protocol
        if hasattr(self.agent_info, 'protocol'):
            if self.agent_info.protocol == "grpc":
                return True

        # Detect PAR-deployed agents by endpoint
        endpoint = self.agent_info.endpoint_url.lower()

        # Use gRPC for PAR agents
        if "par.pixell.global" in endpoint:
            return True

        # Use gRPC for any https PAR endpoint
        if self.agent_info.protocol == "https" and "/agents/" in endpoint:
            return True

        # Default to HTTP
        return False

    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to agent using appropriate protocol.

        Args:
            message: Message payload

        Returns:
            Response dictionary
        """
        logger.debug(
            f"Sending message via {('gRPC' if self.use_grpc else 'HTTP')}",
            agent_id=self.agent_info.agent_app_id,
            skill_id=message.get("skill_id")
        )

        return await self._client.send_message(message)

    async def health_check(self) -> bool:
        """Perform health check on agent.

        Returns:
            True if healthy, False otherwise
        """
        return await self._client.health_check()

    async def discover_agents(self) -> list[dict[str, Any]]:
        """Discover agents (HTTP only).

        Returns:
            List of agent cards
        """
        if hasattr(self._client, 'discover_agents'):
            return await self._client.discover_agents()
        return []

    async def get_agent_card(self) -> Optional[dict[str, Any]]:
        """Get agent card (HTTP only).

        Returns:
            Agent card or None
        """
        if hasattr(self._client, 'get_agent_card'):
            return await self._client.get_agent_card()
        return None

    def get_protocol(self) -> str:
        """Get the protocol being used.

        Returns:
            "grpc" or "http"
        """
        return "grpc" if self.use_grpc else "http"

    def get_agent_id(self) -> str:
        """Get the target agent's ID.

        Returns:
            Agent app ID
        """
        return self.agent_info.agent_app_id

    def get_endpoint(self) -> str:
        """Get the endpoint URL.

        Returns:
            Endpoint URL
        """
        return self.agent_info.endpoint_url
