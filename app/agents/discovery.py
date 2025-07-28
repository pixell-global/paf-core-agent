"""Agent discovery service using pixell list command."""

import asyncio
import json
import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.agents.models import AgentInfo, AgentStatus, AgentCapability, AgentRegistry
from app.utils.logging_config import get_logger
from app.settings import Settings


class AgentDiscoveryService:
    """Service for discovering and managing external agents."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("agent_discovery")
        self.registry = AgentRegistry()
        self.discovery_interval = 60  # seconds
        self.last_discovery = None
        self._discovery_task = None
        self._discovery_lock = asyncio.Lock()
        
    async def startup(self):
        """Start the discovery service."""
        self.logger.info("Starting agent discovery service")
        
        # Perform initial discovery
        # await self.discover_agents() # TODO: 윈도우에서 NOT IMPLEMENTED error 예외 발생
        
        # Start background discovery task
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        
    async def shutdown(self):
        """Shutdown the discovery service."""
        self.logger.info("Shutting down agent discovery service")
        
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
    
    async def discover_agents(self) -> AgentRegistry:
        """Discover available agents using pixell list command."""
        async with self._discovery_lock:
            try:
                self.logger.debug("Running pixell list to discover agents")
                
                # Run pixell list command
                result = await self._run_pixell_list()
                
                if result:
                    # Parse the output and update registry
                    agents = self._parse_pixell_output(result)
                    self.registry.agents = agents
                    self.registry.last_updated = datetime.utcnow()
                    self.last_discovery = datetime.utcnow()
                    
                    self.logger.info(
                        f"Discovered {len(agents)} agents",
                        agent_ids=[a.agent_id for a in agents]
                    )
                else:
                    self.logger.warning("No agents discovered from pixell list")
                    
            except Exception as e:
                self.logger.error(f"Agent discovery failed: {e}", exc_info=True)
                
        return self.registry
    
    async def _run_pixell_list(self) -> Optional[str]:
        """Execute pixell list command and return output."""
        try:
            # Run pixell list command TODO: 윈도우에서 NOT IMPLEMENTED error 예외 발생
            process = await asyncio.create_subprocess_exec(
                "pixell", "list", "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0  # 10 second timeout
            )
            
            if process.returncode != 0:
                self.logger.error(
                    f"pixell list failed with code {process.returncode}",
                    stderr=stderr.decode() if stderr else None
                )
                return None
                
            return stdout.decode() if stdout else None
            
        except asyncio.TimeoutError:
            self.logger.error("pixell list command timed out")
            return None
        except FileNotFoundError:
            self.logger.error("pixell command not found. Please ensure pixell is installed.")
            return None
        except Exception as e:
            self.logger.error(f"Error running pixell list: {e}")
            return None
    
    def _parse_pixell_output(self, output: str) -> List[AgentInfo]:
        """Parse pixell list output and convert to AgentInfo objects."""
        agents = []
        
        try:
            # Assume pixell returns JSON format
            data = json.loads(output)
            
            # Handle different possible output formats
            if isinstance(data, list):
                agent_list = data
            elif isinstance(data, dict) and "agents" in data:
                agent_list = data["agents"]
            else:
                self.logger.warning(f"Unexpected pixell output format: {type(data)}")
                return agents
            
            for agent_data in agent_list:
                try:
                    # Convert pixell format to our AgentInfo format
                    agent_info = self._convert_to_agent_info(agent_data)
                    if agent_info:
                        agents.append(agent_info)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse agent data: {e}",
                        agent_data=agent_data
                    )
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse pixell JSON output: {e}")
            # Try to parse as line-based format
            agents = self._parse_line_format(output)
            
        return agents
    
    def _convert_to_agent_info(self, data: Dict[str, Any]) -> Optional[AgentInfo]:
        """Convert pixell agent data to AgentInfo model."""
        try:
            # Map pixell fields to our model
            capabilities = []
            if "capabilities" in data:
                for cap in data.get("capabilities", []):
                    if isinstance(cap, dict):
                        capabilities.append(AgentCapability(
                            name=cap.get("name", "unknown"),
                            description=cap.get("description", ""),
                            version=cap.get("version", "1.0.0"),
                            input_schema=cap.get("input_schema"),
                            output_schema=cap.get("output_schema"),
                            estimated_duration_ms=cap.get("estimated_duration_ms"),
                            tags=cap.get("tags", [])
                        ))
                    elif isinstance(cap, str):
                        # Simple capability name
                        capabilities.append(AgentCapability(
                            name=cap,
                            description=f"Capability: {cap}"
                        ))
            
            # Determine status
            status_map = {
                "online": AgentStatus.AVAILABLE,
                "available": AgentStatus.AVAILABLE,
                "busy": AgentStatus.BUSY,
                "offline": AgentStatus.OFFLINE,
                "error": AgentStatus.ERROR,
                "maintenance": AgentStatus.MAINTENANCE
            }
            status = status_map.get(
                data.get("status", "").lower(),
                AgentStatus.AVAILABLE
            )
            
            agent_info = AgentInfo(
                agent_id=data.get("id", data.get("agent_id", "")),
                name=data.get("name", "Unknown Agent"),
                description=data.get("description", "No description available"),
                version=data.get("version", "1.0.0"),
                status=status,
                capabilities=capabilities,
                endpoint=data.get("endpoint", data.get("url", "")),
                protocol=data.get("protocol", "grpc"),
                authentication=data.get("auth", data.get("authentication")),
                metadata=data.get("metadata", {}),
                health_check_interval=data.get("health_check_interval", 30)
            )
            
            return agent_info
            
        except Exception as e:
            self.logger.error(f"Failed to convert agent data: {e}")
            return None
    
    def _parse_line_format(self, output: str) -> List[AgentInfo]:
        """Parse line-based format as fallback."""
        agents = []
        
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
                
            # Simple format: agent_id|name|endpoint|status
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    agent = AgentInfo(
                        agent_id=parts[0].strip(),
                        name=parts[1].strip(),
                        endpoint=parts[2].strip(),
                        status=AgentStatus.AVAILABLE if len(parts) < 4 else AgentStatus(parts[3].strip()),
                        description=f"Agent: {parts[1].strip()}"
                    )
                    agents.append(agent)
                except Exception as e:
                    self.logger.warning(f"Failed to parse line: {line}, error: {e}")
                    
        return agents
    
    async def _discovery_loop(self):
        """Background task to periodically discover agents."""
        while True:
            try:
                await asyncio.sleep(self.discovery_interval)
                await self.discover_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def get_registry(self) -> AgentRegistry:
        """Get the current agent registry."""
        return self.registry
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get a specific agent by ID."""
        return self.registry.get_agent(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that support a specific capability."""
        return self.registry.get_agents_by_capability(capability)
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get all available agents."""
        return self.registry.get_available_agents()
    
    def is_stale(self) -> bool:
        """Check if the registry data is stale."""
        if not self.last_discovery:
            return True
            
        age = datetime.utcnow() - self.last_discovery
        return age > timedelta(seconds=self.discovery_interval * 2)