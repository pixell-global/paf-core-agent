"""
Base plugin class and metadata definitions.
"""

import abc
from enum import Enum
from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid


class PluginType(str, Enum):
    """Types of plugins supported by the system."""
    UPEE = "upee"  # UPEE loop enhancers
    WORKER = "worker"  # Worker agent implementations
    BRIDGE = "bridge"  # A2A bridge extensions
    DATA = "data"  # Data processing and storage
    LLM_PROVIDER = "llm_provider"  # LLM provider implementations
    INTEGRATIONS = "integrations"  # External system integrations
    MIDDLEWARE = "middleware"  # Request/response middleware
    MONITORING = "monitoring"  # Monitoring and observability


class PluginStatus(str, Enum):
    """Plugin status states."""
    REGISTERED = "registered"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"
    DISABLED = "disabled"


class PluginMetadata(BaseModel):
    """Plugin metadata and configuration."""
    id: str
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    dependencies: List[str] = []
    required_capabilities: List[str] = []
    configuration_schema: Optional[Dict[str, Any]] = None
    api_version: str = "1.0"
    min_core_version: str = "1.0.0"
    max_core_version: Optional[str] = None
    tags: List[str] = []
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PluginConfig(BaseModel):
    """Runtime plugin configuration."""
    enabled: bool = True
    priority: int = 100  # Lower numbers = higher priority
    config: Dict[str, Any] = {}
    environment: str = "production"
    debug: bool = False
    rate_limits: Optional[Dict[str, int]] = None
    resource_limits: Optional[Dict[str, Any]] = None


class BasePlugin(abc.ABC):
    """Base class for all plugins."""
    
    def __init__(self, metadata: PluginMetadata, config: PluginConfig):
        self.metadata = metadata
        self.config = config
        self.status = PluginStatus.REGISTERED
        self.instance_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
        self.error_count = 0
        self.execution_count = 0
        self._hooks: Dict[str, List[callable]] = {}
        self._context: Dict[str, Any] = {}
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin. Called once during plugin loading."""
        pass
    
    @abc.abstractmethod
    async def activate(self) -> None:
        """Activate the plugin. Called when plugin becomes active."""
        pass
    
    @abc.abstractmethod 
    async def deactivate(self) -> None:
        """Deactivate the plugin. Called when plugin is suspended/stopped."""
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources. Called during plugin unloading."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform plugin health check."""
        return {
            "plugin_id": self.metadata.id,
            "status": self.status.value,
            "healthy": self.status in [PluginStatus.ACTIVE, PluginStatus.INITIALIZED],
            "error_count": self.error_count,
            "execution_count": self.execution_count,
            "uptime_seconds": (datetime.now(timezone.utc) - self.created_at).total_seconds()
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "metadata": self.metadata.model_dump(),
            "config": self.config.model_dump(),
            "status": self.status.value,
            "instance_id": self.instance_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "error_count": self.error_count,
            "execution_count": self.execution_count
        }
    
    def register_hook(self, event_name: str, callback: callable) -> None:
        """Register a hook for an event."""
        if event_name not in self._hooks:
            self._hooks[event_name] = []
        self._hooks[event_name].append(callback)
    
    async def trigger_hook(self, event_name: str, *args, **kwargs) -> List[Any]:
        """Trigger hooks for an event."""
        results = []
        if event_name in self._hooks:
            for callback in self._hooks[event_name]:
                try:
                    if hasattr(callback, '__call__'):
                        result = callback(*args, **kwargs)
                        if hasattr(result, '__await__'):
                            result = await result
                        results.append(result)
                except Exception as e:
                    self.error_count += 1
                    results.append({"error": str(e)})
        return results
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        self.execution_count += 1
    
    def increment_error(self) -> None:
        """Increment error count."""
        self.error_count += 1
    
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies."""
        return self.metadata.dependencies.copy()
    
    def get_capabilities(self) -> List[str]:
        """Get required capabilities."""
        return self.metadata.required_capabilities.copy()
    
    def is_compatible(self, core_version: str) -> bool:
        """Check if plugin is compatible with core version."""
        # Simple version comparison - in real implementation would use proper semver
        min_version = self.metadata.min_core_version
        max_version = self.metadata.max_core_version
        
        if core_version < min_version:
            return False
        
        if max_version and core_version > max_version:
            return False
        
        return True
    
    def validate_config(self) -> bool:
        """Validate plugin configuration against schema."""
        if not self.metadata.configuration_schema:
            return True
        
        # Here would implement JSON schema validation
        # For now, just return True
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics."""
        uptime = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return {
            "uptime_seconds": uptime,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1),
            "status": self.status.value,
            "last_activity_seconds_ago": (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        }
    
    def __str__(self) -> str:
        return f"{self.metadata.name} v{self.metadata.version} ({self.metadata.id})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.metadata.id}>"