"""API endpoints for agent management and information."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from app.agents.models import AgentInfo, AgentRegistry, AgentDecision
from app.agents.manager import AgentManager
from app.utils.logging_config import get_logger
from app.api.dependencies import get_upee_engine

logger = get_logger("agents_api")

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/", response_model=List[AgentInfo])
async def list_agents(
    capability: Optional[str] = Query(None, description="Filter by capability"),
    status: Optional[str] = Query(None, description="Filter by status"),
    engine=Depends(get_upee_engine)
) -> List[AgentInfo]:
    """
    List all discovered agents.
    
    Args:
        capability: Optional capability filter
        status: Optional status filter
        
    Returns:
        List of agent information
    """
    try:
        registry = engine.agent_manager.discovery_service.get_registry()
        agents = registry.agents
        
        # Apply filters
        if capability:
            agents = [
                agent for agent in agents
                if any(cap.name == capability for cap in agent.capabilities)
            ]
        
        if status:
            agents = [
                agent for agent in agents
                if agent.status.value == status
            ]
        
        return agents
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    engine=Depends(get_upee_engine)
) -> AgentInfo:
    """
    Get information about a specific agent.
    
    Args:
        agent_id: The agent ID
        
    Returns:
        Agent information
    """
    try:
        agent = engine.agent_manager.discovery_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_agents(
    engine=Depends(get_upee_engine)
) -> Dict[str, Any]:
    """
    Manually refresh the agent registry.
    
    Returns:
        Refresh status and agent count
    """
    try:
        count = await engine.agent_manager.refresh_agents()
        
        return {
            "status": "success",
            "message": f"Refreshed {count} agents",
            "agent_count": count,
            "timestamp": engine.agent_manager.discovery_service.registry.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/usage")
async def get_agent_usage_stats(
    engine=Depends(get_upee_engine)
) -> Dict[str, Any]:
    """
    Get agent usage statistics.
    
    Returns:
        Usage statistics including request counts and performance metrics
    """
    try:
        stats = engine.agent_manager.get_agent_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/health")
async def get_agent_health(
    engine=Depends(get_upee_engine)
) -> Dict[str, Any]:
    """
    Get health status of all agents.
    
    Returns:
        Health information for each agent
    """
    try:
        registry = engine.agent_manager.discovery_service.get_registry()
        
        health_summary = {
            "total_agents": len(registry.agents),
            "healthy_agents": len([a for a in registry.agents if a.status.value == "available"]),
            "offline_agents": len([a for a in registry.agents if a.status.value == "offline"]),
            "error_agents": len([a for a in registry.agents if a.status.value == "error"]),
            "agents": []
        }
        
        for agent in registry.agents:
            health_summary["agents"].append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "last_seen": agent.last_seen.isoformat() if agent.last_seen else None,
                "endpoint": agent.endpoint
            })
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Failed to get agent health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities/list")
async def list_capabilities(
    engine=Depends(get_upee_engine)
) -> List[Dict[str, Any]]:
    """
    List all unique capabilities across all agents.
    
    Returns:
        List of capabilities with agent counts
    """
    try:
        registry = engine.agent_manager.discovery_service.get_registry()
        
        # Collect all capabilities
        capability_map = {}
        for agent in registry.agents:
            for cap in agent.capabilities:
                if cap.name not in capability_map:
                    capability_map[cap.name] = {
                        "name": cap.name,
                        "description": cap.description,
                        "agents": [],
                        "tags": list(set(cap.tags))
                    }
                
                capability_map[cap.name]["agents"].append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "status": agent.status.value
                })
        
        # Convert to list and add counts
        capabilities = []
        for cap_name, cap_info in capability_map.items():
            cap_info["agent_count"] = len(cap_info["agents"])
            cap_info["available_count"] = len([
                a for a in cap_info["agents"] 
                if a["status"] == "available"
            ])
            capabilities.append(cap_info)
        
        # Sort by agent count
        capabilities.sort(key=lambda x: x["agent_count"], reverse=True)
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to list capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-requests")
async def get_active_requests(
    engine=Depends(get_upee_engine)
) -> List[Dict[str, Any]]:
    """
    Get currently active agent requests.
    
    Returns:
        List of active A2A requests
    """
    try:
        active_requests = engine.agent_manager.get_active_requests()
        
        return [
            {
                "request_id": req.request_id,
                "agent_id": req.agent_id,
                "capability": req.capability,
                "priority": req.priority,
                "timeout_ms": req.timeout_ms
            }
            for req in active_requests.values()
        ]
        
    except Exception as e:
        logger.error(f"Failed to get active requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))