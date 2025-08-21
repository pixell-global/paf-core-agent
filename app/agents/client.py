"""Agent-to-Agent communication client."""

import asyncio
import aiohttp
import grpc
from typing import Dict, Any, Optional
from datetime import datetime

from app.agents.models import A2ARequest, A2AResponse, AgentInfo, AgentStatus
from app.utils.logging_config import get_logger
from app.settings import Settings


class A2AClient:
    """Client for communicating with external agents."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("a2a_client")
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.grpc_channels: Dict[str, grpc.aio.Channel] = {}
        
    async def startup(self):
        """Initialize the A2A client."""
        self.logger.info("Starting A2A client")
        self.http_session = aiohttp.ClientSession()
        
    async def shutdown(self):
        """Cleanup A2A client resources."""
        self.logger.info("Shutting down A2A client")
        
        if self.http_session:
            await self.http_session.close()
            
        for channel in self.grpc_channels.values():
            await channel.close()
        self.grpc_channels.clear()
    
    async def send_request(
        self,
        agent: AgentInfo,
        request: A2ARequest
    ) -> A2AResponse:
        """Send a request to an external agent."""
        self.logger.info(
            f"Sending A2A request",
            agent_id=agent.agent_id,
            capability=request.capability,
            request_id=request.request_id
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Check agent availability
            if agent.status != AgentStatus.AVAILABLE:
                return A2AResponse(
                    request_id=request.request_id,
                    agent_id=agent.agent_id,
                    status="error",
                    error=f"Agent is not available: {agent.status}"
                )
            
            # Route based on protocol
            if agent.protocol == "http":
                response = await self._send_http_request(agent, request)
            elif agent.protocol == "grpc":
                response = await self._send_grpc_request(agent, request)
            elif agent.protocol == "websocket":
                response = await self._send_websocket_request(agent, request)
            else:
                response = A2AResponse(
                    request_id=request.request_id,
                    agent_id=agent.agent_id,
                    status="error",
                    error=f"Unsupported protocol: {agent.protocol}"
                )
            
            # Calculate execution time
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            response.execution_time_ms = execution_time
            
            self.logger.info(
                f"A2A request completed",
                request_id=request.request_id,
                status=response.status,
                execution_time_ms=execution_time
            )
            
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(
                f"A2A request timed out",
                request_id=request.request_id,
                timeout_ms=request.timeout_ms
            )
            return A2AResponse(
                request_id=request.request_id,
                agent_id=agent.agent_id,
                status="timeout",
                error=f"Request timed out after {request.timeout_ms}ms"
            )
        except Exception as e:
            self.logger.error(
                f"A2A request failed",
                request_id=request.request_id,
                error=str(e),
                exc_info=True
            )
            return A2AResponse(
                request_id=request.request_id,
                agent_id=agent.agent_id,
                status="error",
                error=str(e)
            )
    
    async def _send_http_request(
        self,
        agent: AgentInfo,
        request: A2ARequest
    ) -> A2AResponse:
        """Send HTTP request to agent."""
        if not self.http_session:
            raise RuntimeError("HTTP session not initialized")
            
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request.request_id,
            "X-Capability": request.capability
        }
        
        # Add authentication headers if needed
        if agent.authentication:
            if "bearer_token" in agent.authentication:
                headers["Authorization"] = f"Bearer {agent.authentication['bearer_token']}"
            elif "api_key" in agent.authentication:
                headers["X-API-Key"] = agent.authentication["api_key"]
        
        # Prepare request data
        data = {
            "request_id": request.request_id,
            "capability": request.capability,
            "payload": request.payload,
            "context": request.context,
            "priority": request.priority
        }
        # Attach UI capabilities if present in context to top-level metadata for agents expecting it in metadata
        if isinstance(request.context, dict) and "ui.capabilities" in request.context:
            caps = request.context.get("ui.capabilities")
            if isinstance(data.get("context"), dict):
                data["context"]["ui.capabilities"] = caps
        
        # Send request with timeout
        timeout = aiohttp.ClientTimeout(total=request.timeout_ms / 1000.0)
        
        async with self.http_session.post(
            f"{agent.endpoint}/execute",
            json=data,
            headers=headers,
            timeout=timeout
        ) as response:
            response_data = await response.json()
            
            if response.status == 200:
                return A2AResponse(
                    request_id=request.request_id,
                    agent_id=agent.agent_id,
                    status="success",
                    result=response_data.get("result"),
                    metadata=response_data.get("metadata", {})
                )
            else:
                return A2AResponse(
                    request_id=request.request_id,
                    agent_id=agent.agent_id,
                    status="error",
                    error=response_data.get("error", f"HTTP {response.status}")
                )
    
    async def _send_grpc_request(
        self,
        agent: AgentInfo,
        request: A2ARequest
    ) -> A2AResponse:
        """Send gRPC request to agent."""
        # Get or create channel
        if agent.agent_id not in self.grpc_channels:
            self.grpc_channels[agent.agent_id] = grpc.aio.insecure_channel(
                agent.endpoint,
                options=[
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                ]
            )
        
        channel = self.grpc_channels[agent.agent_id]
        
        # For now, return a placeholder response
        # In a real implementation, you would:
        # 1. Import the generated gRPC stubs
        # 2. Create a stub client
        # 3. Call the appropriate RPC method
        
        self.logger.warning(
            "gRPC implementation pending - returning mock response",
            agent_id=agent.agent_id
        )
        
        return A2AResponse(
            request_id=request.request_id,
            agent_id=agent.agent_id,
            status="success",
            result={"message": "gRPC mock response"},
            metadata={"protocol": "grpc", "mock": True}
        )
    
    async def _send_websocket_request(
        self,
        agent: AgentInfo,
        request: A2ARequest
    ) -> A2AResponse:
        """Send WebSocket request to agent."""
        # WebSocket implementation would go here
        # For now, return error
        
        self.logger.warning(
            "WebSocket implementation pending",
            agent_id=agent.agent_id
        )
        
        return A2AResponse(
            request_id=request.request_id,
            agent_id=agent.agent_id,
            status="error",
            error="WebSocket protocol not yet implemented"
        )
    
    async def health_check(self, agent: AgentInfo) -> bool:
        """Perform health check on an agent."""
        try:
            # Simple health check request
            health_request = A2ARequest(
                request_id=f"health-{agent.agent_id}-{datetime.utcnow().timestamp()}",
                agent_id=agent.agent_id,
                capability="health",
                payload={"check": "ping"},
                timeout_ms=5000
            )
            
            response = await self.send_request(agent, health_request)
            return response.status == "success"
            
        except Exception as e:
            self.logger.error(
                f"Health check failed for agent {agent.agent_id}: {e}"
            )
            return False