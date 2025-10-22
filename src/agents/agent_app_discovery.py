"""Agent App Discovery Service - Discovers and manages multiple agent apps."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.agents.agent_app_registry import AgentAppRegistry, AgentAppInfo
from src.agents.grpc_agent_card_client import GrpcAgentCardClient
from src.utils.logging_config import get_logger
from src.settings import Settings

logger = get_logger("agent_app_discovery")


class AgentAppDiscoveryService:
    """Service for discovering and managing multiple agent apps."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = AgentAppRegistry()
        self._discovery_task: Optional[asyncio.Task] = None

    async def startup(self):
        """Initialize the discovery service."""
        logger.info("Starting agent app discovery service")

        # Load agents from configuration
        await self._load_agents_from_config()

        # Perform initial card fetch
        await self.fetch_all_agent_cards()

        # Start background refresh task if configured
        if self.settings.a2a_card_refresh_interval > 0:
            self._discovery_task = asyncio.create_task(self._refresh_loop())

        logger.info(
            f"Discovery service started with {len(self.registry.agents)} agents"
        )

    async def shutdown(self):
        """Shutdown the discovery service."""
        logger.info("Shutting down agent app discovery service")

        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

    async def _load_agents_from_config(self):
        """Load agent apps from settings configuration."""
        for agent_config in self.settings.a2a_agent_apps:
            try:
                agent = AgentAppInfo(**agent_config)
                self.registry.add_agent(agent)
            except Exception as e:
                logger.error(
                    f"Failed to load agent config: {e}",
                    agent_config=agent_config
                )

        logger.info(f"Loaded {len(self.registry.agents)} agents from configuration")

    async def fetch_all_agent_cards(self) -> int:
        """Fetch agent cards from all configured agents.

        Returns:
            Number of successfully fetched cards
        """
        enabled_agents = self.registry.get_enabled_agents()

        if not enabled_agents:
            logger.warning("No enabled agents to fetch cards from")
            return 0

        logger.info(f"Fetching agent cards from {len(enabled_agents)} agents")

        # Fetch cards concurrently
        tasks = [
            self._fetch_agent_card(agent)
            for agent in enabled_agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        success_count = sum(1 for r in results if r and not isinstance(r, Exception))

        logger.info(
            f"Fetched {success_count}/{len(enabled_agents)} agent cards successfully"
        )

        return success_count

    async def _fetch_agent_card(
        self,
        agent: AgentAppInfo
    ) -> Optional[Dict[str, Any]]:
        """Fetch agent card from a single agent.

        Tries gRPC first for PAR-deployed agents, falls back to HTTP.

        Args:
            agent: AgentAppInfo instance

        Returns:
            Agent card dictionary or None on failure
        """
        # Try gRPC first for PAR-deployed agents
        if agent.protocol == "https" and "par.pixell.global" in agent.endpoint_url:
            logger.debug(f"Attempting gRPC fetch for PAR agent: {agent.name}")

            grpc_client = GrpcAgentCardClient(timeout=10.0)
            agent_card = await grpc_client.fetch_agent_card(
                agent.agent_app_id,
                agent.endpoint_url
            )

            if agent_card:
                # Update registry
                self.registry.update_agent_card(agent.agent_app_id, agent_card)
                self.registry.update_health_status(agent.agent_app_id, "healthy")

                skills_count = len(agent_card.get("skills", []))
                logger.info(
                    f"Fetched card via gRPC for {agent.name}: {skills_count} skills",
                    agent_id=agent.agent_app_id
                )
                return agent_card
            else:
                logger.warning(
                    f"gRPC fetch failed for {agent.name}, falling back to HTTP"
                )

        # Fallback to HTTP (standard A2A discovery endpoint)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{agent.endpoint_url}/.well-known/agent.json"

                logger.debug(f"Fetching agent card from {url}")

                response = await client.get(url)
                response.raise_for_status()

                agent_card = response.json()

                # Update registry
                self.registry.update_agent_card(agent.agent_app_id, agent_card)
                self.registry.update_health_status(agent.agent_app_id, "healthy")

                skills_count = len(agent_card.get("skills", []))
                logger.info(
                    f"Fetched card via HTTP for {agent.name}: {skills_count} skills",
                    agent_id=agent.agent_app_id
                )

                return agent_card

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching card from {agent.name}")
            self.registry.update_health_status(agent.agent_app_id, "timeout")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching card from {agent.name}: {e.response.status_code}"
            )
            self.registry.update_health_status(agent.agent_app_id, "error")
            return None

        except Exception as e:
            logger.error(
                f"Error fetching card from {agent.name}: {e}",
                exc_info=True
            )
            self.registry.update_health_status(agent.agent_app_id, "error")
            return None

    async def _refresh_loop(self):
        """Background task to periodically refresh agent cards."""
        interval = self.settings.a2a_card_refresh_interval

        while True:
            try:
                await asyncio.sleep(interval)
                logger.debug("Refreshing agent cards")
                await self.fetch_all_agent_cards()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def get_registry(self) -> AgentAppRegistry:
        """Get the current registry."""
        return self.registry

    def get_agent(self, agent_app_id: str) -> Optional[AgentAppInfo]:
        """Get a specific agent by ID."""
        return self.registry.get_agent(agent_app_id)

    async def health_check_agent(self, agent_app_id: str) -> bool:
        """Perform health check on a specific agent.

        Args:
            agent_app_id: Agent ID to check

        Returns:
            True if healthy, False otherwise
        """
        agent = self.registry.get_agent(agent_app_id)
        if not agent:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"{agent.endpoint_url}/.well-known/agent.json"
                response = await client.get(url)

                is_healthy = response.status_code == 200
                self.registry.update_health_status(
                    agent_app_id,
                    "healthy" if is_healthy else "unhealthy"
                )
                return is_healthy

        except Exception:
            self.registry.update_health_status(agent_app_id, "unhealthy")
            return False
