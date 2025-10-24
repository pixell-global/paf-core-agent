"""
Plugin system for PAF Core Agent.
"""

from .base.plugin_base import BasePlugin, PluginMetadata, PluginType
from .manager import PluginManager
from .registry import PluginRegistry

__all__ = [
    "BasePlugin",
    "PluginMetadata", 
    "PluginType",
    "PluginManager",
    "PluginRegistry"
]