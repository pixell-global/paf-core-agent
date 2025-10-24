"""gRPC client for fetching agent cards from PAR-deployed agents."""

import asyncio
import grpc
import grpc.aio
from typing import Optional, Dict, Any
from src.proto import agent_pb2, agent_pb2_grpc
from src.utils.logging_config import get_logger

logger = get_logger("grpc_agent_card_client")


class PathPrefixInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """Interceptor to prepend path prefix to all gRPC calls for ALB routing.

    This is required for PAR's path-based routing where each agent is accessed via:
    /agents/{agent_id}/a2a/{grpc_method}
    """

    def __init__(self, path_prefix: str):
        """Initialize with path prefix.

        Args:
            path_prefix: Path prefix to prepend (e.g., "/agents/{agent_id}/a2a")
        """
        self.path_prefix = path_prefix.rstrip('/')

    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept and modify the gRPC call path.

        Transforms paths like:
          /pixell.agent.AgentService/DescribeCapabilities
        Into:
          /agents/{agent_id}/a2a/pixell.agent.AgentService/DescribeCapabilities
        """
        # Modify the method (path) in the call details
        # gRPC method is bytes, so decode -> concatenate -> encode
        original_method = client_call_details.method.decode('utf-8')
        new_method = f"{self.path_prefix}{original_method}".encode('utf-8')

        # Create new call details with modified method
        new_details = grpc.aio.ClientCallDetails(
            method=new_method,
            timeout=client_call_details.timeout,
            metadata=client_call_details.metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
        )

        logger.debug(
            "Rewriting gRPC path",
            original=original_method,
            rewritten=new_method.decode('utf-8')
        )

        return await continuation(new_details, request)


class GrpcAgentCardClient:
    """Client for fetching agent cards via gRPC DescribeCapabilities."""

    def __init__(self, timeout: float = 10.0):
        """Initialize the gRPC agent card client.

        Args:
            timeout: Connection timeout in seconds
        """
        self.timeout = timeout

    async def fetch_agent_card(
        self,
        agent_app_id: str,
        endpoint_url: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch agent card via gRPC DescribeCapabilities.

        Args:
            agent_app_id: The agent app ID
            endpoint_url: Base endpoint URL (e.g., https://par.pixell.global/agents/{id})

        Returns:
            Agent card dictionary with skills, or None on failure
        """
        try:
            # Parse endpoint to get host and port
            host, port = self._parse_endpoint(endpoint_url)

            # Create path prefix for ALB routing
            path_prefix = f"/agents/{agent_app_id}/a2a"
            interceptor = PathPrefixInterceptor(path_prefix)

            # Create gRPC channel with TLS
            credentials = grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(
                f"{host}:{port}",
                credentials,
                options=[
                    ('grpc.dns_resolver', 'native'),  # Use native DNS resolver
                ],
                interceptors=[interceptor]
            )

            # Create stub
            stub = agent_pb2_grpc.AgentServiceStub(channel)

            logger.debug(
                f"Fetching agent card via gRPC",
                agent_id=agent_app_id,
                host=host,
                port=port
            )

            # Call DescribeCapabilities
            try:
                response = await asyncio.wait_for(
                    stub.DescribeCapabilities(agent_pb2.Empty()),
                    timeout=self.timeout
                )

                # Parse response into agent card format
                agent_card = self._parse_capabilities_response(response)

                logger.info(
                    f"Fetched agent card via gRPC",
                    agent_id=agent_app_id,
                    skills_count=len(agent_card.get("skills", []))
                )

                return agent_card

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout fetching agent card via gRPC",
                    agent_id=agent_app_id,
                    timeout=self.timeout
                )
                return None

            except grpc.RpcError as e:
                logger.error(
                    f"gRPC error fetching agent card",
                    agent_id=agent_app_id,
                    code=e.code(),
                    details=e.details()
                )
                return None

            finally:
                await channel.close()

        except Exception as e:
            logger.error(
                f"Error fetching agent card via gRPC",
                agent_id=agent_app_id,
                error=str(e),
                exc_info=True
            )
            return None

    def _parse_endpoint(self, endpoint_url: str) -> tuple[str, int]:
        """Parse endpoint URL to extract host and port.

        Args:
            endpoint_url: Full endpoint URL

        Returns:
            Tuple of (host, port)
        """
        # Remove protocol
        if "://" in endpoint_url:
            endpoint_url = endpoint_url.split("://")[1]

        # Remove path
        if "/" in endpoint_url:
            endpoint_url = endpoint_url.split("/")[0]

        # Extract host and port
        if ":" in endpoint_url:
            host, port_str = endpoint_url.split(":")
            port = int(port_str)
        else:
            # Default ports
            host = endpoint_url
            port = 443  # HTTPS default

        return host, port

    def _parse_capabilities_response(
        self,
        response: agent_pb2.Capabilities
    ) -> Dict[str, Any]:
        """Parse gRPC DescribeCapabilities response into agent card format.

        Args:
            response: gRPC response object

        Returns:
            Agent card dictionary
        """
        metadata = dict(response.metadata) if response.metadata else {}

        # Extract basic info
        agent_card = {
            "name": metadata.get("name", "Unknown Agent"),
            "version": metadata.get("version", "unknown"),
            "description": metadata.get("description", ""),
            "methods": list(response.methods) if response.methods else [],
        }

        # Try to extract skills from metadata
        # Skills might be in different formats depending on agent implementation
        skills = []

        # Check if skills are in metadata as JSON string
        if "skills" in metadata:
            import json
            try:
                skills_data = json.loads(metadata["skills"])
                if isinstance(skills_data, list):
                    skills = skills_data
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse skills from metadata")

        # If no skills found, create basic skill from available methods
        if not skills and "Invoke" in agent_card["methods"]:
            skills = [{
                "id": "invoke",
                "name": "invoke",
                "description": f"Invoke {agent_card['name']} capabilities",
                "tags": ["general"]
            }]

        agent_card["skills"] = skills

        return agent_card
