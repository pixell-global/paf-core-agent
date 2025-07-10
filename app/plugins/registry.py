"""
Plugin registry for managing plugin metadata and discovery.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

from .base.plugin_base import BasePlugin, PluginMetadata, PluginType, PluginStatus

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for plugin discovery and metadata management."""
    
    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or ["app/plugins/installed"]
        self.registered_plugins: Dict[str, PluginMetadata] = {}
        self.plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        
        # Ensure plugin directories exist
        for directory in self.plugin_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def register_plugin_class(self, plugin_class: Type[BasePlugin]) -> None:
        """Register a plugin class for instantiation."""
        # Plugin class should have metadata as class attribute
        if hasattr(plugin_class, 'METADATA'):
            metadata = plugin_class.METADATA
            if isinstance(metadata, dict):
                metadata = PluginMetadata(**metadata)
            
            self.plugin_classes[metadata.id] = plugin_class
            logger.info(f"Registered plugin class: {metadata.id}")
        else:
            logger.warning(f"Plugin class {plugin_class.__name__} missing METADATA attribute")
    
    def discover_plugins(self) -> None:
        """Discover plugins in configured directories."""
        logger.info("Starting plugin discovery...")
        
        for directory in self.plugin_directories:
            self._discover_in_directory(directory)
        
        logger.info(f"Discovered {len(self.registered_plugins)} plugins")
    
    def _discover_in_directory(self, directory: str) -> None:
        """Discover plugins in a specific directory."""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return
        
        # Look for plugin.json files
        for plugin_file in directory_path.rglob("plugin.json"):
            try:
                self._load_plugin_metadata(plugin_file)
            except Exception as e:
                logger.error(f"Error loading plugin metadata from {plugin_file}: {e}")
    
    def _load_plugin_metadata(self, metadata_file: Path) -> None:
        """Load plugin metadata from JSON file."""
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = PluginMetadata(**metadata_dict)
            
            # Validate required fields
            if not metadata.id or not metadata.name or not metadata.version:
                logger.warning(f"Invalid plugin metadata in {metadata_file}: missing required fields")
                return
            
            # Store metadata
            self.registered_plugins[metadata.id] = metadata
            
            # Build dependency graph
            self.plugin_dependencies[metadata.id] = metadata.dependencies.copy()
            
            logger.debug(f"Loaded plugin metadata: {metadata.id} v{metadata.version}")
            
        except Exception as e:
            logger.error(f"Error parsing plugin metadata {metadata_file}: {e}")
    
    def register_plugin_metadata(self, metadata: PluginMetadata) -> None:
        """Manually register plugin metadata."""
        self.registered_plugins[metadata.id] = metadata
        self.plugin_dependencies[metadata.id] = metadata.dependencies.copy()
        logger.info(f"Registered plugin metadata: {metadata.id}")
    
    def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin."""
        return self.registered_plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """Get all plugins of a specific type."""
        return [
            metadata for metadata in self.registered_plugins.values()
            if metadata.plugin_type == plugin_type
        ]
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Get all registered plugins."""
        return self.registered_plugins.copy()
    
    def resolve_dependencies(self, plugin_id: str) -> List[str]:
        """Resolve plugin dependencies in order."""
        resolved = []
        visited = set()
        
        def _resolve(current_id: str):
            if current_id in visited:
                return  # Avoid cycles
            
            visited.add(current_id)
            
            dependencies = self.plugin_dependencies.get(current_id, [])
            for dep_id in dependencies:
                if dep_id in self.registered_plugins:
                    _resolve(dep_id)
                else:
                    logger.warning(f"Dependency {dep_id} not found for plugin {current_id}")
            
            if current_id not in resolved:
                resolved.append(current_id)
        
        _resolve(plugin_id)
        return resolved
    
    def validate_dependencies(self, plugin_id: str) -> bool:
        """Validate that all dependencies are available."""
        if plugin_id not in self.plugin_dependencies:
            return True
        
        dependencies = self.plugin_dependencies[plugin_id]
        for dep_id in dependencies:
            if dep_id not in self.registered_plugins:
                logger.error(f"Missing dependency {dep_id} for plugin {plugin_id}")
                return False
        
        return True
    
    def find_plugins_by_capability(self, capability: str) -> List[PluginMetadata]:
        """Find plugins that provide a specific capability."""
        matching_plugins = []
        
        for metadata in self.registered_plugins.values():
            if capability in metadata.required_capabilities:
                matching_plugins.append(metadata)
        
        return matching_plugins
    
    def find_plugins_by_tag(self, tag: str) -> List[PluginMetadata]:
        """Find plugins with a specific tag."""
        matching_plugins = []
        
        for metadata in self.registered_plugins.values():
            if tag in metadata.tags:
                matching_plugins.append(metadata)
        
        return matching_plugins
    
    def search_plugins(self, query: str) -> List[PluginMetadata]:
        """Search plugins by name, description, or tags."""
        query_lower = query.lower()
        matching_plugins = []
        
        for metadata in self.registered_plugins.values():
            # Check name and description
            if (query_lower in metadata.name.lower() or 
                query_lower in metadata.description.lower()):
                matching_plugins.append(metadata)
                continue
            
            # Check tags
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    matching_plugins.append(metadata)
                    break
        
        return matching_plugins
    
    def get_plugin_class(self, plugin_id: str) -> Optional[Type[BasePlugin]]:
        """Get the plugin class for instantiation."""
        return self.plugin_classes.get(plugin_id)
    
    def is_plugin_compatible(self, plugin_id: str, core_version: str) -> bool:
        """Check if plugin is compatible with core version."""
        metadata = self.get_plugin_metadata(plugin_id)
        if not metadata:
            return False
        
        # Simple version comparison
        min_version = metadata.min_core_version
        max_version = metadata.max_core_version
        
        if core_version < min_version:
            return False
        
        if max_version and core_version > max_version:
            return False
        
        return True
    
    def export_registry(self, output_file: str) -> None:
        """Export registry to JSON file."""
        registry_data = {
            "plugins": {
                plugin_id: metadata.model_dump()
                for plugin_id, metadata in self.registered_plugins.items()
            },
            "dependencies": self.plugin_dependencies
        }
        
        with open(output_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Exported plugin registry to {output_file}")
    
    def import_registry(self, input_file: str) -> None:
        """Import registry from JSON file."""
        try:
            with open(input_file, 'r') as f:
                registry_data = json.load(f)
            
            # Load plugins
            for plugin_id, metadata_dict in registry_data.get("plugins", {}).items():
                metadata = PluginMetadata(**metadata_dict)
                self.registered_plugins[plugin_id] = metadata
            
            # Load dependencies
            self.plugin_dependencies.update(registry_data.get("dependencies", {}))
            
            logger.info(f"Imported plugin registry from {input_file}")
            
        except Exception as e:
            logger.error(f"Error importing plugin registry from {input_file}: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        plugin_counts_by_type = {}
        for metadata in self.registered_plugins.values():
            plugin_type = metadata.plugin_type.value
            plugin_counts_by_type[plugin_type] = plugin_counts_by_type.get(plugin_type, 0) + 1
        
        return {
            "total_plugins": len(self.registered_plugins),
            "plugin_types": plugin_counts_by_type,
            "plugin_directories": self.plugin_directories,
            "plugins_with_dependencies": len([
                pid for pid, deps in self.plugin_dependencies.items() if deps
            ])
        }