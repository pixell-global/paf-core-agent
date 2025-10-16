"""
Base plugin classes and interfaces.
"""

from .plugin_base import BasePlugin, PluginMetadata, PluginType
from .interfaces import (
    UPEEPluginInterface,
    WorkerPluginInterface, 
    BridgePluginInterface,
    DataPluginInterface
)

__all__ = [
    "BasePlugin",
    "PluginMetadata",
    "PluginType", 
    "UPEEPluginInterface",
    "WorkerPluginInterface",
    "BridgePluginInterface", 
    "DataPluginInterface"
]