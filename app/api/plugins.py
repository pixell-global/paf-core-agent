"""
Plugin management API endpoints.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.plugins.registry import PluginRegistry
from app.plugins.manager import PluginManager
from app.plugins.base.plugin_base import PluginConfig, PluginType

router = APIRouter()

# Global plugin components - in production would be managed differently
plugin_registry: Optional[PluginRegistry] = None
plugin_manager: Optional[PluginManager] = None


class PluginConfigRequest(BaseModel):
    enabled: bool = True
    priority: int = 100
    config: Dict[str, Any] = {}
    environment: str = "production"
    debug: bool = False


def get_registry() -> PluginRegistry:
    """Get the plugin registry."""
    global plugin_registry
    if not plugin_registry:
        plugin_registry = PluginRegistry()
    return plugin_registry


def get_manager() -> PluginManager:
    """Get the plugin manager."""
    global plugin_manager
    if not plugin_manager:
        registry = get_registry()
        plugin_manager = PluginManager(registry)
    return plugin_manager


@router.post("/start")
async def start_plugin_system(db: AsyncSession = Depends(get_db)):
    """Start the plugin system."""
    try:
        manager = get_manager()
        await manager.start()
        
        return {
            "message": "Plugin system started successfully",
            "status": manager.get_manager_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start plugin system: {str(e)}")


@router.post("/stop")
async def stop_plugin_system(db: AsyncSession = Depends(get_db)):
    """Stop the plugin system."""
    try:
        manager = get_manager()
        await manager.stop()
        
        return {"message": "Plugin system stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop plugin system: {str(e)}")


@router.get("/status")
async def get_plugin_system_status(db: AsyncSession = Depends(get_db)):
    """Get plugin system status."""
    try:
        manager = get_manager()
        return manager.get_manager_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/registry/discover")
async def discover_plugins(db: AsyncSession = Depends(get_db)):
    """Discover plugins in configured directories."""
    try:
        registry = get_registry()
        registry.discover_plugins()
        
        stats = registry.get_registry_stats()
        return {
            "message": "Plugin discovery completed",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover plugins: {str(e)}")


@router.get("/registry/all")
async def get_all_plugins(db: AsyncSession = Depends(get_db)):
    """Get all registered plugins."""
    try:
        registry = get_registry()
        plugins = registry.get_all_plugins()
        
        return {
            "plugins": {
                plugin_id: metadata.model_dump()
                for plugin_id, metadata in plugins.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugins: {str(e)}")


@router.get("/registry/by-type/{plugin_type}")
async def get_plugins_by_type(
    plugin_type: PluginType,
    db: AsyncSession = Depends(get_db)
):
    """Get plugins by type."""
    try:
        registry = get_registry()
        plugins = registry.get_plugins_by_type(plugin_type)
        
        return {
            "plugin_type": plugin_type.value,
            "plugins": [metadata.model_dump() for metadata in plugins]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugins by type: {str(e)}")


@router.get("/registry/search")
async def search_plugins(
    query: str = Query(..., description="Search query for plugins"),
    db: AsyncSession = Depends(get_db)
):
    """Search plugins by name, description, or tags."""
    try:
        registry = get_registry()
        plugins = registry.search_plugins(query)
        
        return {
            "query": query,
            "results": [metadata.model_dump() for metadata in plugins]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search plugins: {str(e)}")


@router.get("/loaded")
async def get_loaded_plugins(db: AsyncSession = Depends(get_db)):
    """Get all loaded plugins."""
    try:
        manager = get_manager()
        
        loaded_plugins = {}
        for plugin_id, plugin in manager.loaded_plugins.items():
            loaded_plugins[plugin_id] = plugin.get_info()
        
        return {"loaded_plugins": loaded_plugins}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get loaded plugins: {str(e)}")


@router.post("/load/{plugin_id}")
async def load_plugin(
    plugin_id: str,
    config_request: Optional[PluginConfigRequest] = None,
    db: AsyncSession = Depends(get_db)
):
    """Load a specific plugin."""
    try:
        manager = get_manager()
        
        config = None
        if config_request:
            config = PluginConfig(**config_request.model_dump())
        
        success = await manager.load_plugin(plugin_id, config)
        
        if success:
            return {
                "message": f"Plugin {plugin_id} loaded successfully",
                "plugin_info": manager.get_plugin(plugin_id).get_info() if manager.get_plugin(plugin_id) else None
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load plugin {plugin_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading plugin: {str(e)}")


@router.post("/unload/{plugin_id}")
async def unload_plugin(plugin_id: str, db: AsyncSession = Depends(get_db)):
    """Unload a specific plugin."""
    try:
        manager = get_manager()
        success = await manager.unload_plugin(plugin_id)
        
        if success:
            return {"message": f"Plugin {plugin_id} unloaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to unload plugin {plugin_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unloading plugin: {str(e)}")


@router.post("/activate/{plugin_id}")
async def activate_plugin(plugin_id: str, db: AsyncSession = Depends(get_db)):
    """Activate a loaded plugin."""
    try:
        manager = get_manager()
        success = await manager.activate_plugin(plugin_id)
        
        if success:
            return {"message": f"Plugin {plugin_id} activated successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to activate plugin {plugin_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating plugin: {str(e)}")


@router.post("/deactivate/{plugin_id}")
async def deactivate_plugin(plugin_id: str, db: AsyncSession = Depends(get_db)):
    """Deactivate an active plugin."""
    try:
        manager = get_manager()
        success = await manager.deactivate_plugin(plugin_id)
        
        if success:
            return {"message": f"Plugin {plugin_id} deactivated successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to deactivate plugin {plugin_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating plugin: {str(e)}")


@router.get("/health")
async def health_check_plugins(db: AsyncSession = Depends(get_db)):
    """Perform health check on all loaded plugins."""
    try:
        manager = get_manager()
        health_results = await manager.health_check_all()
        
        return {"health_results": health_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform health check: {str(e)}")


@router.get("/metrics")
async def get_plugin_metrics(db: AsyncSession = Depends(get_db)):
    """Get execution metrics for all plugins."""
    try:
        manager = get_manager()
        metrics = manager.get_plugin_metrics()
        
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plugin metrics: {str(e)}")


@router.get("/{plugin_id}")
async def get_plugin_info(plugin_id: str, db: AsyncSession = Depends(get_db)):
    """Get information about a specific plugin."""
    try:
        manager = get_manager()
        plugin = manager.get_plugin(plugin_id)
        
        if not plugin:
            # Try to get from registry
            registry = get_registry()
            metadata = registry.get_plugin_metadata(plugin_id)
            if metadata:
                return {
                    "metadata": metadata.model_dump(),
                    "loaded": False
                }
            else:
                raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")
        
        return {
            "plugin_info": plugin.get_info(),
            "loaded": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting plugin info: {str(e)}")


@router.post("/hook/execute/{event_name}")
async def execute_hook(
    event_name: str,
    payload: Dict[str, Any] = {},
    db: AsyncSession = Depends(get_db)
):
    """Execute a plugin hook for testing purposes."""
    try:
        manager = get_manager()
        results = await manager.execute_plugin_hook(event_name, payload=payload)
        
        return {
            "event_name": event_name,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing hook: {str(e)}")


@router.post("/hook/register/{plugin_id}/{event_name}")
async def register_plugin_hook(
    plugin_id: str,
    event_name: str,
    db: AsyncSession = Depends(get_db)
):
    """Register a plugin for a specific hook event."""
    try:
        manager = get_manager()
        success = manager.register_plugin_hook(plugin_id, event_name)
        
        if success:
            return {"message": f"Plugin {plugin_id} registered for hook {event_name}"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to register hook")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering hook: {str(e)}")