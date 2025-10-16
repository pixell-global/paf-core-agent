"""Agent-to-Agent (A2A) protocol models and interfaces."""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    """Agent availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(BaseModel):
    """Describes a specific capability an agent provides."""
    name: str = Field(..., description="Capability name (e.g., 'code_analysis', 'data_processing')")
    description: str = Field(..., description="Human-readable description of the capability")
    version: str = Field(default="1.0.0", description="Capability version")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for expected input")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for output format")
    estimated_duration_ms: Optional[int] = Field(None, description="Estimated execution time in milliseconds")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost per invocation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class AgentInfo(BaseModel):
    """Information about an external agent."""
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(..., description="Agent description and purpose")
    version: str = Field(default="1.0.0", description="Agent version")
    status: AgentStatus = Field(default=AgentStatus.AVAILABLE)
    capabilities: List[AgentCapability] = Field(default_factory=list)
    endpoint: str = Field(..., description="Agent endpoint URL or connection string")
    protocol: Literal["grpc", "http", "websocket"] = Field(default="grpc")
    authentication: Optional[Dict[str, Any]] = Field(None, description="Authentication requirements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    last_seen: Optional[datetime] = Field(None, description="Last health check timestamp")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")


class A2ARequest(BaseModel):
    """Request to an external agent."""
    request_id: str = Field(..., description="Unique request identifier")
    agent_id: str = Field(..., description="Target agent ID")
    capability: str = Field(..., description="Requested capability name")
    payload: Dict[str, Any] = Field(..., description="Request payload")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    timeout_ms: Optional[int] = Field(default=30000, description="Request timeout in milliseconds")
    priority: Literal["low", "medium", "high"] = Field(default="medium")
    callback_url: Optional[str] = Field(None, description="Callback URL for async responses")


class A2AResponse(BaseModel):
    """Response from an external agent."""
    request_id: str = Field(..., description="Original request ID")
    agent_id: str = Field(..., description="Responding agent ID")
    status: Literal["success", "error", "timeout", "partial"] = Field(..., description="Response status")
    result: Optional[Dict[str, Any]] = Field(None, description="Response payload")
    error: Optional[str] = Field(None, description="Error message if status is error")
    execution_time_ms: Optional[int] = Field(None, description="Actual execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentDecision(BaseModel):
    """Decision to use an external agent."""
    phase: str = Field(..., description="UPEE phase making the decision")
    agent_id: str = Field(..., description="Selected agent ID")
    capability: str = Field(..., description="Selected capability")
    reasoning: str = Field(..., description="Why this agent was selected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative agents considered")
    estimated_value: float = Field(..., ge=0.0, le=1.0, description="Expected value contribution")


class AgentRegistry(BaseModel):
    """Registry of available agents from pixell list."""
    agents: List[AgentInfo] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="pixell", description="Registry source")
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that have a specific capability."""
        matching_agents = []
        for agent in self.agents:
            if any(cap.name == capability for cap in agent.capabilities):
                matching_agents.append(agent)
        return matching_agents
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get all available agents."""
        return [agent for agent in self.agents if agent.status == AgentStatus.AVAILABLE]