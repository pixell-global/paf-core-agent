"""Agent Manager - Orchestrates agent discovery, communication, and decision-making."""

import asyncio
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from app.agents.models import (
    AgentInfo, AgentDecision, A2ARequest, A2AResponse,
    AgentStatus, AgentCapability
)
from app.agents.discovery import AgentDiscoveryService
from app.agents.client import A2AClient
from app.schemas import UPEEPhase
from app.utils.logging_config import get_logger
from app.settings import Settings


class AgentManager:
    """Manages external agent discovery, selection, and communication."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("agent_manager")
        
        # Initialize components
        self.discovery_service = AgentDiscoveryService(settings)
        self.a2a_client = A2AClient(settings)
        
        # Decision cache to avoid repeated agent selection
        self.decision_cache: Dict[str, AgentDecision] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Execution tracking
        self.active_requests: Dict[str, A2ARequest] = {}
        self.request_history: List[Tuple[A2ARequest, A2AResponse]] = []
        self.max_history = 100
        
    async def startup(self):
        """Initialize the agent manager."""
        self.logger.info("Starting agent manager")
        
        await self.discovery_service.startup()
        await self.a2a_client.startup()
        
        # Start health monitoring task
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        self.logger.info("Agent manager started successfully")
        
    async def shutdown(self):
        """Shutdown the agent manager."""
        self.logger.info("Shutting down agent manager")
        
        # Cancel health monitor task
        if hasattr(self, '_health_monitor_task'):
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.discovery_service.shutdown()
        await self.a2a_client.shutdown()
        
        self.logger.info("Agent manager shutdown complete")
    
    async def should_use_agent(
        self,
        phase: UPEEPhase,
        context: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """
        Determine if an external agent should be used for the current context.
        
        Args:
            phase: Current UPEE phase
            context: Phase context including user request, understanding results, etc.
            
        Returns:
            AgentDecision if an agent should be used, None otherwise
        """
        cache_key = f"{phase.value}:{hash(str(sorted(context.items())))}"
        
        # Check cache
        if cache_key in self.decision_cache:
            decision = self.decision_cache[cache_key]
            if (datetime.utcnow() - decision.timestamp).seconds < self.cache_ttl:
                self.logger.debug(
                    "Using cached agent decision",
                    phase=phase.value,
                    agent_id=decision.agent_id
                )
                return decision
        
        # Analyze context to determine if agent is needed
        decision = await self._analyze_agent_need(phase, context)
        
        if decision:
            self.decision_cache[cache_key] = decision
            self.logger.info(
                "Agent selected for use",
                phase=phase.value,
                agent_id=decision.agent_id,
                capability=decision.capability,
                confidence=decision.confidence
            )
        
        return decision
    
    async def _analyze_agent_need(
        self,
        phase: UPEEPhase,
        context: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """
        Analyze context to determine if an external agent is needed.
        
        This is where the AI decision logic lives - analyzing the context
        to determine if external agents would be beneficial.
        """
        # Get available agents
        agents = self.discovery_service.get_available_agents()
        if not agents:
            return None
        
        # Phase-specific analysis
        if phase == UPEEPhase.UNDERSTAND:
            return await self._analyze_understand_phase(context, agents)
        elif phase == UPEEPhase.PLAN:
            return await self._analyze_plan_phase(context, agents)
        elif phase == UPEEPhase.EXECUTE:
            return await self._analyze_execute_phase(context, agents)
        elif phase == UPEEPhase.EVALUATE:
            return await self._analyze_evaluate_phase(context, agents)
        
        return None
    
    async def _analyze_understand_phase(
        self,
        context: Dict[str, Any],
        agents: List[AgentInfo]
    ) -> Optional[AgentDecision]:
        """Analyze if understanding phase needs external agents."""
        user_message = context.get("message", "")
        files = context.get("files", [])
        
        # Check for specialized understanding needs
        needs = []
        
        # Complex code analysis
        if any(f.get("type") == "code" for f in files) and len(files) > 5:
            needs.append(("code_analysis", 0.8))
            
        # Domain-specific language understanding
        domain_keywords = {
            "medical": ["diagnosis", "symptoms", "treatment", "patient"],
            "legal": ["contract", "liability", "clause", "jurisdiction"],
            "financial": ["investment", "portfolio", "risk", "trading"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_message.lower() for keyword in keywords):
                needs.append((f"{domain}_understanding", 0.7))
        
        # Multi-language content
        if context.get("detected_languages", []) and len(context.get("detected_languages", [])) > 1:
            needs.append(("translation", 0.9))
        
        # Select best agent for highest priority need
        for capability_needed, confidence in sorted(needs, key=lambda x: x[1], reverse=True):
            capable_agents = self.discovery_service.get_agents_by_capability(capability_needed)
            if capable_agents:
                selected_agent = capable_agents[0]  # Simple selection - could be more sophisticated
                return AgentDecision(
                    phase=UPEEPhase.UNDERSTAND.value,
                    agent_id=selected_agent.agent_id,
                    capability=capability_needed,
                    reasoning=f"Specialized {capability_needed} needed for complex understanding",
                    confidence=confidence,
                    alternatives=[{"agent_id": a.agent_id, "name": a.name} for a in capable_agents[1:3]],
                    estimated_value=confidence * 0.8
                )
        
        return None
    
    async def _analyze_plan_phase(
        self,
        context: Dict[str, Any],
        agents: List[AgentInfo]
    ) -> Optional[AgentDecision]:
        """Analyze if planning phase needs external agents."""
        understanding_result = context.get("understanding_result", {})
        intent = understanding_result.get("metadata", {}).get("intent", "")
        complexity = understanding_result.get("metadata", {}).get("complexity", "simple")
        
        # Check for planning needs
        if complexity == "complex" or complexity == "very_complex":
            # Look for strategy planning agents
            planning_agents = self.discovery_service.get_agents_by_capability("strategic_planning")
            if planning_agents:
                return AgentDecision(
                    phase=UPEEPhase.PLAN.value,
                    agent_id=planning_agents[0].agent_id,
                    capability="strategic_planning",
                    reasoning=f"Complex task requires specialized planning for {intent}",
                    confidence=0.85,
                    alternatives=[{"agent_id": a.agent_id, "name": a.name} for a in planning_agents[1:3]],
                    estimated_value=0.9
                )
        
        # Check for specific domain planning
        if intent in ["architecture_design", "system_design", "workflow_optimization"]:
            design_agents = self.discovery_service.get_agents_by_capability(f"{intent}_planning")
            if design_agents:
                return AgentDecision(
                    phase=UPEEPhase.PLAN.value,
                    agent_id=design_agents[0].agent_id,
                    capability=f"{intent}_planning",
                    reasoning=f"Specialized planning needed for {intent}",
                    confidence=0.75,
                    alternatives=[{"agent_id": a.agent_id, "name": a.name} for a in design_agents[1:2]],
                    estimated_value=0.8
                )
        
        return None
    
    async def _analyze_execute_phase(
        self,
        context: Dict[str, Any],
        agents: List[AgentInfo]
    ) -> Optional[AgentDecision]:
        """Analyze if execution phase needs external agents."""
        plan_result = context.get("plan_result", {})
        strategy = plan_result.get("metadata", {}).get("strategy", "")
        needs_external = plan_result.get("metadata", {}).get("needs_external_calls", False)
        
        if not needs_external:
            return None
        
        # Check what kind of execution is needed
        execution_types = plan_result.get("metadata", {}).get("execution_types", [])
        
        for exec_type in execution_types:
            # Map execution types to capabilities
            capability_map = {
                "code_generation": "advanced_code_generation",
                "data_processing": "large_scale_data_processing",
                "api_integration": "api_orchestration",
                "image_generation": "image_synthesis",
                "document_analysis": "document_extraction"
            }
            
            capability = capability_map.get(exec_type, exec_type)
            capable_agents = self.discovery_service.get_agents_by_capability(capability)
            
            if capable_agents:
                return AgentDecision(
                    phase=UPEEPhase.EXECUTE.value,
                    agent_id=capable_agents[0].agent_id,
                    capability=capability,
                    reasoning=f"Specialized execution needed for {exec_type}",
                    confidence=0.9,
                    alternatives=[{"agent_id": a.agent_id, "name": a.name} for a in capable_agents[1:3]],
                    estimated_value=0.95
                )
        
        return None
    
    async def _analyze_evaluate_phase(
        self,
        context: Dict[str, Any],
        agents: List[AgentInfo]
    ) -> Optional[AgentDecision]:
        """Analyze if evaluation phase needs external agents."""
        execute_result = context.get("execute_result", {})
        response_type = execute_result.get("metadata", {}).get("response_type", "")
        
        # Check for specialized evaluation needs
        if response_type in ["code", "technical_design", "mathematical_proof"]:
            verification_agents = self.discovery_service.get_agents_by_capability(f"{response_type}_verification")
            if verification_agents:
                return AgentDecision(
                    phase=UPEEPhase.EVALUATE.value,
                    agent_id=verification_agents[0].agent_id,
                    capability=f"{response_type}_verification",
                    reasoning=f"Specialized verification needed for {response_type}",
                    confidence=0.8,
                    alternatives=[{"agent_id": a.agent_id, "name": a.name} for a in verification_agents[1:2]],
                    estimated_value=0.85
                )
        
        return None
    
    async def execute_agent_request(
        self,
        decision: AgentDecision,
        payload: Dict[str, Any],
        context: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> A2AResponse:
        """
        Execute a request to an external agent based on the decision.
        
        Args:
            decision: The agent decision
            payload: Request payload
            context: Request context
            timeout_ms: Optional timeout override
            
        Returns:
            A2A response from the agent
        """
        # Get agent info
        agent = self.discovery_service.get_agent(decision.agent_id)
        if not agent:
            self.logger.error(f"Agent not found: {decision.agent_id}")
            return A2AResponse(
                request_id=str(uuid.uuid4()),
                agent_id=decision.agent_id,
                status="error",
                error="Agent not found in registry"
            )
        
        # Create A2A request
        request = A2ARequest(
            request_id=str(uuid.uuid4()),
            agent_id=decision.agent_id,
            capability=decision.capability,
            payload=payload,
            context={
                **context,
                "upee_phase": decision.phase,
                "decision_confidence": decision.confidence,
                "decision_reasoning": decision.reasoning
            },
            timeout_ms=timeout_ms or 30000,
            priority="high" if decision.confidence > 0.8 else "medium"
        )
        
        # Track active request
        self.active_requests[request.request_id] = request
        
        try:
            # Send request
            response = await self.a2a_client.send_request(agent, request)
            
            # Store in history
            self.request_history.append((request, response))
            if len(self.request_history) > self.max_history:
                self.request_history.pop(0)
            
            return response
            
        finally:
            # Remove from active requests
            self.active_requests.pop(request.request_id, None)
    
    def get_active_requests(self) -> Dict[str, A2ARequest]:
        """Get currently active A2A requests."""
        return self.active_requests.copy()
    
    def get_request_history(self) -> List[Tuple[A2ARequest, A2AResponse]]:
        """Get recent request history."""
        return self.request_history.copy()
    
    async def refresh_agents(self) -> int:
        """Manually refresh the agent registry."""
        registry = await self.discovery_service.discover_agents()
        return len(registry.agents)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agent usage."""
        total_requests = len(self.request_history)
        successful_requests = sum(
            1 for _, response in self.request_history
            if response.status == "success"
        )
        
        agent_usage = {}
        for request, response in self.request_history:
            agent_id = request.agent_id
            if agent_id not in agent_usage:
                agent_usage[agent_id] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_execution_time_ms": 0
                }
            
            agent_usage[agent_id]["total"] += 1
            if response.status == "success":
                agent_usage[agent_id]["successful"] += 1
            else:
                agent_usage[agent_id]["failed"] += 1
            
            if response.execution_time_ms:
                # Simple moving average
                current_avg = agent_usage[agent_id]["avg_execution_time_ms"]
                count = agent_usage[agent_id]["total"]
                new_avg = ((current_avg * (count - 1)) + response.execution_time_ms) / count
                agent_usage[agent_id]["avg_execution_time_ms"] = new_avg
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "active_requests": len(self.active_requests),
            "available_agents": len(self.discovery_service.get_available_agents()),
            "agent_usage": agent_usage
        }
    
    async def _health_monitor_loop(self):
        """Periodically monitor health of registered agents."""
        health_check_interval = 60  # seconds
        
        while True:
            try:
                await asyncio.sleep(health_check_interval)
                
                # Get all agents
                agents = self.discovery_service.get_registry().agents
                
                if not agents:
                    continue
                
                self.logger.debug(f"Running health checks on {len(agents)} agents")
                
                # Check health of each agent
                health_tasks = []
                for agent in agents:
                    health_tasks.append(self._check_agent_health(agent))
                
                # Run health checks concurrently
                health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
                
                # Update agent statuses
                healthy_count = 0
                for agent, result in zip(agents, health_results):
                    if isinstance(result, Exception):
                        self.logger.warning(
                            f"Health check failed for agent {agent.agent_id}: {result}"
                        )
                        agent.status = AgentStatus.ERROR
                    elif result:
                        agent.status = AgentStatus.AVAILABLE
                        healthy_count += 1
                    else:
                        agent.status = AgentStatus.OFFLINE
                    
                    agent.last_seen = datetime.utcnow()
                
                self.logger.info(
                    f"Health check complete: {healthy_count}/{len(agents)} agents healthy"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _check_agent_health(self, agent: AgentInfo) -> bool:
        """Check health of a single agent."""
        try:
            return await self.a2a_client.health_check(agent)
        except Exception as e:
            self.logger.error(
                f"Health check failed for agent {agent.agent_id}: {e}"
            )
            return False