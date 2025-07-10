"""
Plugin interfaces for different types of functionality.
"""

import abc
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime

from app.core.bridge.protocol import A2AMessage


class UPEEPluginInterface(abc.ABC):
    """Interface for UPEE loop enhancement plugins."""
    
    @abc.abstractmethod
    async def enhance_understand(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Understand phase."""
        pass
    
    @abc.abstractmethod
    async def enhance_plan(self, understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Plan phase."""
        pass
    
    @abc.abstractmethod
    async def enhance_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Execute phase."""
        pass
    
    @abc.abstractmethod
    async def enhance_evaluate(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Evaluate phase."""
        pass


class WorkerPluginInterface(abc.ABC):
    """Interface for worker agent plugins."""
    
    @abc.abstractmethod
    async def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this worker can handle the given task."""
        pass
    
    @abc.abstractmethod
    async def execute_task(
        self, 
        task_type: str, 
        task_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task."""
        pass
    
    @abc.abstractmethod
    async def get_capabilities(self) -> List[str]:
        """Get worker capabilities."""
        pass
    
    @abc.abstractmethod
    async def get_current_load(self) -> int:
        """Get current workload (0-100)."""
        pass
    
    @abc.abstractmethod
    async def get_max_capacity(self) -> int:
        """Get maximum task capacity."""
        pass


class BridgePluginInterface(abc.ABC):
    """Interface for A2A bridge extension plugins."""
    
    @abc.abstractmethod
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle/transform an A2A message."""
        pass
    
    @abc.abstractmethod
    async def route_message(self, message: A2AMessage) -> List[str]:
        """Determine routing for a message. Returns list of agent IDs."""
        pass
    
    @abc.abstractmethod
    async def authenticate_agent(self, agent_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate an agent."""
        pass
    
    @abc.abstractmethod
    async def authorize_message(self, message: A2AMessage) -> bool:
        """Authorize a message for delivery."""
        pass


class DataPluginInterface(abc.ABC):
    """Interface for data processing and storage plugins."""
    
    @abc.abstractmethod
    async def store_data(
        self, 
        data_type: str, 
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store data and return storage ID."""
        pass
    
    @abc.abstractmethod
    async def retrieve_data(self, storage_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored data."""
        pass
    
    @abc.abstractmethod
    async def search_data(
        self, 
        query: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search stored data."""
        pass
    
    @abc.abstractmethod
    async def delete_data(self, storage_id: str) -> bool:
        """Delete stored data."""
        pass
    
    @abc.abstractmethod
    async def stream_data(
        self,
        query: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream data results."""
        pass


class LLMProviderPluginInterface(abc.ABC):
    """Interface for LLM provider plugins."""
    
    @abc.abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        model: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text completion."""
        pass
    
    @abc.abstractmethod
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        pass
    
    @abc.abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @abc.abstractmethod
    async def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """Estimate usage cost."""
        pass


class IntegrationPluginInterface(abc.ABC):
    """Interface for external system integration plugins."""
    
    @abc.abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Establish connection to external system."""
        pass
    
    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from external system."""
        pass
    
    @abc.abstractmethod
    async def send_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to external system."""
        pass
    
    @abc.abstractmethod
    async def receive_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Receive data from external system."""
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check connection health."""
        pass


class MiddlewarePluginInterface(abc.ABC):
    """Interface for middleware plugins."""
    
    @abc.abstractmethod
    async def process_request(
        self,
        request_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process incoming request."""
        pass
    
    @abc.abstractmethod
    async def process_response(
        self,
        response_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process outgoing response."""
        pass
    
    @abc.abstractmethod
    async def process_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process errors."""
        pass


class MonitoringPluginInterface(abc.ABC):
    """Interface for monitoring and observability plugins."""
    
    @abc.abstractmethod
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric."""
        pass
    
    @abc.abstractmethod
    async def record_event(
        self,
        event_name: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record an event."""
        pass
    
    @abc.abstractmethod
    async def start_trace(self, operation_name: str) -> str:
        """Start a distributed trace."""
        pass
    
    @abc.abstractmethod
    async def end_trace(
        self,
        trace_id: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a distributed trace."""
        pass
    
    @abc.abstractmethod
    async def get_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List[float]]:
        """Retrieve metrics data."""
        pass