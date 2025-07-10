"""Base gRPC client infrastructure."""

import asyncio
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import json

try:
    import grpc
    from grpc import aio as grpc_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from app.utils.logging_config import get_logger
from app.settings import Settings

logger = get_logger("grpc_base")


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISCONNECTED = "disconnected"
    BUSY = "busy"


@dataclass
class ServiceEndpoint:
    """gRPC service endpoint information."""
    host: str
    port: int
    name: str
    service_type: str
    health_check_path: str = "/health"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def address(self) -> str:
        """Get the full address."""
        return f"{self.host}:{self.port}"


@dataclass
class ServiceHealth:
    """Service health information."""
    status: ServiceStatus
    last_check: float
    response_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GRPCRequest:
    """Generic gRPC request."""
    service_type: str
    method: str
    payload: Dict[str, Any]
    timeout: Optional[float] = None
    metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GRPCResponse:
    """Generic gRPC response."""
    success: bool
    data: Any
    error: Optional[str] = None
    response_time: float = 0.0
    service_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseGRPCClient(ABC):
    """Base class for gRPC clients."""
    
    def __init__(self, settings: Settings, service_type: str):
        self.settings = settings
        self.service_type = service_type
        self.logger = get_logger(f"grpc_{service_type}")
        
        # Service discovery
        self.endpoints: List[ServiceEndpoint] = []
        self.health_status: Dict[str, ServiceHealth] = {}
        
        # Connection management
        self.channels: Dict[str, grpc_aio.Channel] = {}
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Load balancing
        self.current_endpoint_index = 0
        self.connection_failures: Dict[str, int] = {}
        
        # Initialize endpoints
        self._initialize_endpoints()
    
    def _initialize_endpoints(self):
        """Initialize service endpoints from configuration."""
        # This would typically come from service discovery
        # For now, we'll use configuration-based endpoints
        
        if self.service_type == "worker_agent":
            # Worker agent endpoints
            worker_endpoints = getattr(self.settings, 'worker_agent_endpoints', [])
            if not worker_endpoints:
                # Default development endpoint
                worker_endpoints = [{"host": "localhost", "port": 50051}]
            
            for endpoint_config in worker_endpoints:
                endpoint = ServiceEndpoint(
                    host=endpoint_config.get("host", "localhost"),
                    port=endpoint_config.get("port", 50051),
                    name=f"worker-{endpoint_config.get('host', 'localhost')}",
                    service_type=self.service_type,
                    metadata=endpoint_config.get("metadata", {})
                )
                self.endpoints.append(endpoint)
                
                # Initialize health status
                self.health_status[endpoint.address] = ServiceHealth(
                    status=ServiceStatus.UNKNOWN,
                    last_check=0
                )
        
        self.logger.info(
            "Initialized gRPC endpoints",
            service_type=self.service_type,
            endpoint_count=len(self.endpoints)
        )
    
    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Perform health checks on all endpoints."""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return self.health_status
        
        self.logger.debug(f"Performing health checks for {self.service_type}")
        
        # Check all endpoints concurrently
        tasks = []
        for endpoint in self.endpoints:
            task = asyncio.create_task(self._health_check_endpoint(endpoint))
            tasks.append(task)
        
        # Wait for all health checks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update health status
        for endpoint, result in zip(self.endpoints, results):
            if isinstance(result, Exception):
                self.health_status[endpoint.address] = ServiceHealth(
                    status=ServiceStatus.UNHEALTHY,
                    last_check=current_time,
                    error=str(result)
                )
            else:
                self.health_status[endpoint.address] = result
        
        self.last_health_check = current_time
        return self.health_status
    
    async def _health_check_endpoint(self, endpoint: ServiceEndpoint) -> ServiceHealth:
        """Perform health check on a specific endpoint."""
        start_time = time.time()
        
        try:
            # Create temporary channel for health check
            channel = grpc_aio.insecure_channel(endpoint.address)
            
            # Simple health check - try to connect
            # In a real implementation, this would use gRPC health checking protocol
            try:
                # Create a simple call with timeout
                await asyncio.wait_for(
                    self._test_connection(channel),
                    timeout=5.0
                )
                
                response_time = time.time() - start_time
                await channel.close()
                
                return ServiceHealth(
                    status=ServiceStatus.HEALTHY,
                    last_check=time.time(),
                    response_time=response_time
                )
                
            except asyncio.TimeoutError:
                await channel.close()
                return ServiceHealth(
                    status=ServiceStatus.UNHEALTHY,
                    last_check=time.time(),
                    error="Health check timeout"
                )
        
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                last_check=time.time(),
                error=str(e)
            )
    
    async def _test_connection(self, channel: grpc_aio.Channel):
        """Test connection to endpoint."""
        # This is a placeholder - in real implementation,
        # this would use the actual service's health check method
        await asyncio.sleep(0.1)  # Simulate connection test
    
    def get_healthy_endpoints(self) -> List[ServiceEndpoint]:
        """Get list of healthy endpoints."""
        healthy = []
        for endpoint in self.endpoints:
            health = self.health_status.get(endpoint.address)
            if health and health.status == ServiceStatus.HEALTHY:
                healthy.append(endpoint)
        return healthy
    
    def select_endpoint(self) -> Optional[ServiceEndpoint]:
        """Select an endpoint using round-robin load balancing."""
        healthy_endpoints = self.get_healthy_endpoints()
        
        if not healthy_endpoints:
            # Fallback to any endpoint if none are healthy
            if self.endpoints:
                self.logger.warning(f"No healthy endpoints for {self.service_type}, using fallback")
                return self.endpoints[0]
            return None
        
        # Round-robin selection
        endpoint = healthy_endpoints[self.current_endpoint_index % len(healthy_endpoints)]
        self.current_endpoint_index += 1
        
        return endpoint
    
    async def get_channel(self, endpoint: ServiceEndpoint) -> grpc_aio.Channel:
        """Get or create a gRPC channel for the endpoint."""
        address = endpoint.address
        
        if address not in self.channels:
            # Create new channel
            channel = grpc_aio.insecure_channel(address)
            self.channels[address] = channel
            
            self.logger.debug(f"Created new gRPC channel to {address}")
        
        return self.channels[address]
    
    async def close_all_channels(self):
        """Close all gRPC channels."""
        for address, channel in self.channels.items():
            try:
                await channel.close()
                self.logger.debug(f"Closed gRPC channel to {address}")
            except Exception as e:
                self.logger.warning(f"Error closing channel to {address}: {e}")
        
        self.channels.clear()
    
    @abstractmethod
    async def call_service(self, request: GRPCRequest) -> GRPCResponse:
        """Make a gRPC call to the service."""
        pass
    
    async def call_with_retry(
        self, 
        request: GRPCRequest, 
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ) -> GRPCResponse:
        """Make a gRPC call with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Select endpoint
                endpoint = self.select_endpoint()
                if not endpoint:
                    return GRPCResponse(
                        success=False,
                        data=None,
                        error="No available endpoints"
                    )
                
                # Make the call
                response = await self.call_service(request)
                
                if response.success:
                    # Reset failure count on success
                    self.connection_failures[endpoint.address] = 0
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                last_error = str(e)
                
                # Track failures
                endpoint = self.select_endpoint()
                if endpoint:
                    self.connection_failures[endpoint.address] = \
                        self.connection_failures.get(endpoint.address, 0) + 1
            
            # Wait before retry (with jitter)
            if attempt < max_retries:
                delay = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                
                self.logger.warning(
                    f"gRPC call failed, retrying in {delay:.2f}s",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=last_error
                )
        
        return GRPCResponse(
            success=False,
            data=None,
            error=f"All retry attempts failed. Last error: {last_error}"
        )


class GRPCClientError(Exception):
    """Base exception for gRPC client errors."""
    pass


class ServiceUnavailableError(GRPCClientError):
    """Raised when no healthy services are available."""
    pass


class ServiceTimeoutError(GRPCClientError):
    """Raised when service call times out."""
    pass 