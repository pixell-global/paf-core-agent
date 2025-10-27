"""gRPC-based A2A client for invoking other agents via gRPC protocol.

This client is designed for PAR-deployed agents where A2A communication uses gRPC
instead of HTTP. It correctly uses the target agent's ID in the path prefix.
"""

import asyncio
import grpc
import grpc.aio
import uuid
import json
from typing import Dict, Any, Optional
from src.proto import agent_pb2, agent_pb2_grpc
from src.agents.grpc_agent_card_client import PathPrefixInterceptor
from src.utils.logging_config import get_logger

logger = get_logger("grpc_a2a_client")


class GrpcA2AClient:
    """gRPC-based A2A client for invoking other agents.

    This client correctly uses the target agent's ID in the gRPC path prefix,
    which is required for PAR's routing to work correctly.
    """

    def __init__(self, agent_app_id: str, endpoint_url: str, timeout: float = 30.0):
        """Initialize the gRPC A2A client.

        Args:
            agent_app_id: The TARGET agent's app ID (not the caller's ID)
            endpoint_url: Base endpoint URL (e.g., https://par.pixell.global/agents/{id})
            timeout: Request timeout in seconds
        """
        self.agent_app_id = agent_app_id
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send an Invoke request to the target agent via gRPC.

        Args:
            message: Message payload with type, skill_id, parameters, etc.

        Returns:
            Response dictionary with status, data, metadata
        """
        try:
            # Parse endpoint to get host and port
            host, port = self._parse_endpoint(self.endpoint_url)

            # Create path prefix using TARGET agent's ID (critical for PAR routing)
            path_prefix = f"/agents/{self.agent_app_id}/a2a"
            interceptor = PathPrefixInterceptor(path_prefix)

            # Create gRPC channel with TLS
            credentials = grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(
                f"{host}:{port}",
                credentials,
                options=[
                    ('grpc.dns_resolver', 'native'),
                ],
                interceptors=[interceptor]
            )

            # Create stub
            stub = agent_pb2_grpc.AgentServiceStub(channel)

            logger.debug(
                f"Invoking agent via gRPC",
                agent_id=self.agent_app_id,
                host=host,
                port=port,
                path_prefix=path_prefix
            )

            # Build ActionRequest from message payload
            action_request = self._build_action_request(message)

            # Call Invoke with timeout
            try:
                response = await asyncio.wait_for(
                    stub.Invoke(action_request),
                    timeout=self.timeout
                )

                # Parse response
                result = self._parse_action_result(response, message)

                logger.info(
                    f"gRPC Invoke succeeded",
                    agent_id=self.agent_app_id,
                    success=response.success
                )

                return result

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout invoking agent via gRPC",
                    agent_id=self.agent_app_id,
                    timeout=self.timeout
                )
                return {
                    "status": "error",
                    "error": f"Timeout after {self.timeout}s"
                }

            except grpc.RpcError as e:
                logger.error(
                    f"gRPC error invoking agent",
                    agent_id=self.agent_app_id,
                    code=e.code(),
                    details=e.details()
                )
                return {
                    "status": "error",
                    "error": f"gRPC error: {e.code()} - {e.details()}"
                }

            finally:
                await channel.close()

        except Exception as e:
            logger.error(
                f"Error invoking agent via gRPC",
                agent_id=self.agent_app_id,
                error=str(e),
                exc_info=True
            )
            return {
                "status": "error",
                "error": str(e)
            }

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

    def _build_action_request(self, message: Dict[str, Any]) -> agent_pb2.ActionRequest:
        """Build proto ActionRequest from message payload using standard A2A format.

        Builds a JSON-RPC 2.0 compliant A2A message with the following structure:
        {
            "jsonrpc": "2.0",
            "id": "<uuid>",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": "<uuid>",
                    "metadata": {
                        "skill": "<skill_id>",
                        "params": {<parameters>}
                    },
                    "parts": [
                        {"kind": "text", "text": "<json_params>"}
                    ]
                }
            }
        }

        Args:
            message: Message payload dict with skill_id, parameters, etc.

        Returns:
            ActionRequest proto message with A2A format
        """
        # Extract key fields
        skill_id = message.get("skill_id", "")
        parameters = message.get("parameters", {})
        user_message = message.get("user_message", "")

        # Generate unique IDs
        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        # Build standard A2A message structure
        a2a_params = {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": message_id,
                "metadata": {
                    "skill": skill_id,
                    "params": parameters
                },
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(parameters, ensure_ascii=False)
                    }
                ]
            }
        }

        # Add user_message to parts if provided
        if user_message:
            a2a_params["message"]["parts"].insert(0, {
                "kind": "text",
                "text": user_message
            })

        # Create A2A message
        a2a_message = agent_pb2.A2AMessage(
            jsonrpc="2.0",
            id=request_id,
            method="message/send",
            params_json=json.dumps(a2a_params)
        )

        # Wrap in ActionRequest
        request = agent_pb2.ActionRequest(
            a2a_message=a2a_message
        )

        logger.debug(
            "Built A2A message",
            request_id=request_id,
            message_id=message_id,
            skill=skill_id
        )

        return request

    def _parse_action_result(
        self,
        response: agent_pb2.ActionResult,
        original_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse proto ActionResult into response dict.

        Args:
            response: ActionResult proto message
            original_message: Original message for context

        Returns:
            Response dictionary
        """
        if response.success:
            return {
                "status": "success",
                "data": response.result,
                "metadata": dict(response.metadata) if response.metadata else {},
                "duration_ms": response.duration_ms,
                "request_id": response.request_id
            }
        else:
            return {
                "status": "error",
                "error": response.error or "Unknown error",
                "metadata": dict(response.metadata) if response.metadata else {},
                "duration_ms": response.duration_ms,
                "request_id": response.request_id
            }

    async def health_check(self) -> bool:
        """Perform health check on the target agent.

        Returns:
            True if healthy, False otherwise
        """
        try:
            host, port = self._parse_endpoint(self.endpoint_url)
            path_prefix = f"/agents/{self.agent_app_id}/a2a"
            interceptor = PathPrefixInterceptor(path_prefix)

            credentials = grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(
                f"{host}:{port}",
                credentials,
                options=[('grpc.dns_resolver', 'native')],
                interceptors=[interceptor]
            )

            stub = agent_pb2_grpc.AgentServiceStub(channel)

            try:
                response = await asyncio.wait_for(
                    stub.Health(agent_pb2.Empty()),
                    timeout=5.0
                )

                await channel.close()
                return response.status == "ok"

            except asyncio.TimeoutError:
                await channel.close()
                return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
