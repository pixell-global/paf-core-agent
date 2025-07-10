"""
Plugin manager for loading, managing, and executing plugins.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type, Set
from datetime import datetime, timezone

from .base.plugin_base import BasePlugin, PluginMetadata, PluginConfig, PluginStatus, PluginType
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """Manager for plugin lifecycle and execution."""
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.running = False
        
        # Plugin execution tracking
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.plugin_hooks: Dict[str, List[str]] = {}  # event -> [plugin_ids]
        
        # Core version for compatibility checking
        self.core_version = "1.0.0"
    
    async def start(self) -> None:
        """Start the plugin manager."""
        if self.running:
            logger.warning("Plugin manager is already running")
            return
        
        self.running = True
        
        # Discover plugins
        self.registry.discover_plugins()
        
        # Load enabled plugins
        await self._load_enabled_plugins()
        
        logger.info(f"Plugin manager started with {len(self.loaded_plugins)} plugins")
    
    async def stop(self) -> None:
        """Stop the plugin manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Deactivate and cleanup all plugins
        for plugin in self.loaded_plugins.values():
            try:
                await plugin.deactivate()
                await plugin.cleanup()
            except Exception as e:
                logger.error(f"Error stopping plugin {plugin.metadata.id}: {e}")
        
        self.loaded_plugins.clear()
        logger.info("Plugin manager stopped")
    
    async def load_plugin(self, plugin_id: str, config: Optional[PluginConfig] = None) -> bool:
        """Load and initialize a specific plugin."""
        try:
            # Check if already loaded
            if plugin_id in self.loaded_plugins:
                logger.warning(f"Plugin {plugin_id} is already loaded")
                return True
            
            # Get metadata
            metadata = self.registry.get_plugin_metadata(plugin_id)
            if not metadata:
                logger.error(f"Plugin metadata not found: {plugin_id}")
                return False
            
            # Check compatibility
            if not self.registry.is_plugin_compatible(plugin_id, self.core_version):
                logger.error(f"Plugin {plugin_id} is not compatible with core version {self.core_version}")
                return False
            
            # Validate dependencies
            if not self.registry.validate_dependencies(plugin_id):
                logger.error(f"Plugin {plugin_id} has missing dependencies")
                return False
            
            # Load dependencies first
            dependencies = self.registry.resolve_dependencies(plugin_id)
            for dep_id in dependencies[:-1]:  # Exclude self
                if dep_id not in self.loaded_plugins:
                    if not await self.load_plugin(dep_id):
                        logger.error(f"Failed to load dependency {dep_id} for plugin {plugin_id}")
                        return False
            
            # Get plugin class
            plugin_class = self.registry.get_plugin_class(plugin_id)
            if not plugin_class:
                logger.error(f"Plugin class not found: {plugin_id}")
                return False
            
            # Create config
            if config is None:
                config = self.plugin_configs.get(plugin_id, PluginConfig())
            
            # Validate config
            if not self._validate_plugin_config(metadata, config):
                logger.error(f"Invalid configuration for plugin {plugin_id}")
                return False
            
            # Instantiate plugin
            plugin = plugin_class(metadata, config)
            
            # Initialize plugin
            await plugin.initialize()
            plugin.status = PluginStatus.INITIALIZED
            
            # Store plugin
            self.loaded_plugins[plugin_id] = plugin
            self.plugin_configs[plugin_id] = config
            
            # Initialize execution stats
            self.execution_stats[plugin_id] = {
                "loaded_at": datetime.now(timezone.utc),
                "executions": 0,
                "errors": 0,
                "last_execution": None
            }
            
            logger.info(f"Loaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a specific plugin."""
        try:
            if plugin_id not in self.loaded_plugins:
                logger.warning(f"Plugin {plugin_id} is not loaded")
                return True
            
            plugin = self.loaded_plugins[plugin_id]
            
            # Check for dependents
            dependents = self._get_plugin_dependents(plugin_id)
            if dependents:
                logger.error(f"Cannot unload plugin {plugin_id}: has dependents {dependents}")
                return False
            
            # Deactivate if active
            if plugin.status == PluginStatus.ACTIVE:
                await plugin.deactivate()
            
            # Cleanup
            await plugin.cleanup()
            
            # Remove from tracking
            del self.loaded_plugins[plugin_id]
            del self.execution_stats[plugin_id]
            
            # Remove from hooks
            for event_name, plugin_list in self.plugin_hooks.items():
                if plugin_id in plugin_list:
                    plugin_list.remove(plugin_id)
            
            logger.info(f"Unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    async def activate_plugin(self, plugin_id: str) -> bool:
        """Activate a loaded plugin."""
        try:
            plugin = self.loaded_plugins.get(plugin_id)
            if not plugin:
                logger.error(f"Plugin {plugin_id} is not loaded")
                return False
            
            if plugin.status == PluginStatus.ACTIVE:
                logger.warning(f"Plugin {plugin_id} is already active")
                return True
            
            # Activate plugin
            await plugin.activate()
            plugin.status = PluginStatus.ACTIVE
            
            logger.info(f"Activated plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating plugin {plugin_id}: {e}")
            if plugin_id in self.loaded_plugins:
                self.loaded_plugins[plugin_id].status = PluginStatus.ERROR
            return False
    
    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate an active plugin."""
        try:
            plugin = self.loaded_plugins.get(plugin_id)
            if not plugin:
                logger.error(f"Plugin {plugin_id} is not loaded")
                return False
            
            if plugin.status != PluginStatus.ACTIVE:
                logger.warning(f"Plugin {plugin_id} is not active")
                return True
            
            # Deactivate plugin
            await plugin.deactivate()
            plugin.status = PluginStatus.SUSPENDED
            
            logger.info(f"Deactivated plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating plugin {plugin_id}: {e}")
            if plugin_id in self.loaded_plugins:
                self.loaded_plugins[plugin_id].status = PluginStatus.ERROR
            return False
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all loaded plugins of a specific type."""
        return [
            plugin for plugin in self.loaded_plugins.values()
            if plugin.metadata.plugin_type == plugin_type and plugin.status == PluginStatus.ACTIVE
        ]
    
    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Get a specific loaded plugin."""
        return self.loaded_plugins.get(plugin_id)
    
    async def execute_plugin_hook(self, event_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute all plugins registered for a specific hook."""
        results = {}
        
        plugin_ids = self.plugin_hooks.get(event_name, [])
        
        for plugin_id in plugin_ids:
            plugin = self.loaded_plugins.get(plugin_id)
            if not plugin or plugin.status != PluginStatus.ACTIVE:
                continue
            
            try:
                self.execution_stats[plugin_id]["executions"] += 1
                self.execution_stats[plugin_id]["last_execution"] = datetime.now(timezone.utc)
                
                plugin.update_activity()
                result = await plugin.trigger_hook(event_name, *args, **kwargs)
                results[plugin_id] = result
                
            except Exception as e:
                logger.error(f"Error executing hook {event_name} for plugin {plugin_id}: {e}")
                plugin.increment_error()
                self.execution_stats[plugin_id]["errors"] += 1
                results[plugin_id] = {"error": str(e)}
        
        return results
    
    def register_plugin_hook(self, plugin_id: str, event_name: str) -> bool:
        """Register a plugin for a specific hook event."""
        if plugin_id not in self.loaded_plugins:
            logger.error(f"Plugin {plugin_id} is not loaded")
            return False
        
        if event_name not in self.plugin_hooks:
            self.plugin_hooks[event_name] = []
        
        if plugin_id not in self.plugin_hooks[event_name]:
            self.plugin_hooks[event_name].append(plugin_id)
            logger.debug(f"Registered plugin {plugin_id} for hook {event_name}")
        
        return True
    
    def unregister_plugin_hook(self, plugin_id: str, event_name: str) -> bool:
        """Unregister a plugin from a specific hook event."""
        if event_name in self.plugin_hooks and plugin_id in self.plugin_hooks[event_name]:
            self.plugin_hooks[event_name].remove(plugin_id)
            logger.debug(f"Unregistered plugin {plugin_id} from hook {event_name}")
            return True
        
        return False
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all loaded plugins."""
        health_results = {}
        
        for plugin_id, plugin in self.loaded_plugins.items():
            try:
                health_results[plugin_id] = await plugin.health_check()
            except Exception as e:
                health_results[plugin_id] = {
                    "plugin_id": plugin_id,
                    "healthy": False,
                    "error": str(e)
                }
        
        return health_results
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get plugin manager status."""
        plugin_counts_by_status = {}
        for plugin in self.loaded_plugins.values():
            status = plugin.status.value
            plugin_counts_by_status[status] = plugin_counts_by_status.get(status, 0) + 1
        
        return {
            "running": self.running,
            "total_plugins": len(self.loaded_plugins),
            "plugin_counts_by_status": plugin_counts_by_status,
            "hooks_registered": len(self.plugin_hooks),
            "core_version": self.core_version
        }
    
    def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for all plugins."""
        return {
            plugin_id: {
                **stats,
                "current_status": self.loaded_plugins[plugin_id].status.value,
                "plugin_metrics": self.loaded_plugins[plugin_id].get_metrics()
            }
            for plugin_id, stats in self.execution_stats.items()
            if plugin_id in self.loaded_plugins
        }
    
    async def _load_enabled_plugins(self) -> None:
        """Load all enabled plugins."""
        for plugin_id, metadata in self.registry.get_all_plugins().items():
            config = self.plugin_configs.get(plugin_id, PluginConfig())
            
            if config.enabled:
                await self.load_plugin(plugin_id, config)
                if plugin_id in self.loaded_plugins:
                    await self.activate_plugin(plugin_id)
    
    def _get_plugin_dependents(self, plugin_id: str) -> List[str]:
        """Get plugins that depend on the given plugin."""
        dependents = []
        
        for pid, dependencies in self.registry.plugin_dependencies.items():
            if plugin_id in dependencies and pid in self.loaded_plugins:
                dependents.append(pid)
        
        return dependents
    
    def _validate_plugin_config(self, metadata: PluginMetadata, config: PluginConfig) -> bool:
        """Validate plugin configuration."""
        # Basic validation - in real implementation would use JSON schema
        return True
    
    def set_plugin_config(self, plugin_id: str, config: PluginConfig) -> None:
        """Set configuration for a plugin."""
        self.plugin_configs[plugin_id] = config