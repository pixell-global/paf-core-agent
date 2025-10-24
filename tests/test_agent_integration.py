"""Test script for agent integration with UPEE engine."""

import asyncio
import json
from datetime import datetime

from src.settings import Settings
from src.core.upee_engine import UPEEEngine
from src.schemas import ChatRequest, FileContext
from src.agents.models import AgentInfo, AgentCapability, AgentStatus


async def mock_pixell_list_output():
    """Mock the output of 'pixell list' command for testing."""
    mock_agents = [
        {
            "id": "code-analyzer-001",
            "name": "Advanced Code Analyzer",
            "description": "Specialized agent for deep code analysis and understanding",
            "version": "2.1.0",
            "status": "available",
            "endpoint": "http://localhost:8081",
            "protocol": "http",
            "capabilities": [
                {
                    "name": "code_analysis",
                    "description": "Analyzes code structure, patterns, and quality",
                    "estimated_duration_ms": 5000,
                    "tags": ["code", "analysis", "quality"]
                },
                {
                    "name": "dependency_analysis",
                    "description": "Maps code dependencies and relationships",
                    "estimated_duration_ms": 3000,
                    "tags": ["dependencies", "architecture"]
                }
            ],
            "health_check_interval": 30
        },
        {
            "id": "planner-ai-002",
            "name": "Strategic Planning AI",
            "description": "AI agent specialized in complex task planning and strategy",
            "version": "1.5.0",
            "status": "available",
            "endpoint": "grpc://localhost:50051",
            "protocol": "grpc",
            "capabilities": [
                {
                    "name": "strategic_planning",
                    "description": "Creates detailed strategic plans for complex tasks",
                    "estimated_duration_ms": 8000,
                    "tags": ["planning", "strategy", "complex"]
                },
                {
                    "name": "architecture_design_planning",
                    "description": "Plans software architecture and system design",
                    "estimated_duration_ms": 10000,
                    "tags": ["architecture", "design", "planning"]
                }
            ],
            "health_check_interval": 60
        },
        {
            "id": "data-processor-003",
            "name": "Large Scale Data Processor",
            "description": "Handles large-scale data processing and transformations",
            "version": "3.0.0",
            "status": "available",
            "endpoint": "http://localhost:8082",
            "protocol": "http",
            "capabilities": [
                {
                    "name": "large_scale_data_processing",
                    "description": "Processes and transforms large datasets efficiently",
                    "estimated_duration_ms": 15000,
                    "tags": ["data", "processing", "scale"]
                },
                {
                    "name": "data_analysis",
                    "description": "Performs statistical and analytical operations on data",
                    "estimated_duration_ms": 7000,
                    "tags": ["data", "analysis", "statistics"]
                }
            ],
            "authentication": {
                "api_key": "test-api-key-123"
            },
            "health_check_interval": 45
        }
    ]
    
    return json.dumps({"agents": mock_agents})


async def test_agent_discovery():
    """Test agent discovery functionality."""
    print("\n=== Testing Agent Discovery ===")
    
    settings = Settings()
    engine = UPEEEngine(settings)
    
    try:
        await engine.startup()
        
        # Manually inject mock agents for testing
        # In real scenario, this would come from 'pixell list'
        mock_output = await mock_pixell_list_output()
        agents = engine.agent_manager.discovery_service._parse_pixell_output(mock_output)
        engine.agent_manager.discovery_service.registry.agents = agents
        engine.agent_manager.discovery_service.registry.last_updated = datetime.utcnow()
        
        # Get discovered agents
        registry = engine.agent_manager.discovery_service.get_registry()
        print(f"\nDiscovered {len(registry.agents)} agents:")
        
        for agent in registry.agents:
            print(f"\n- {agent.name} ({agent.agent_id})")
            print(f"  Status: {agent.status}")
            print(f"  Protocol: {agent.protocol}")
            print(f"  Endpoint: {agent.endpoint}")
            print(f"  Capabilities: {', '.join(cap.name for cap in agent.capabilities)}")
        
        # Test capability search
        print("\n\nAgents with 'code_analysis' capability:")
        code_agents = engine.agent_manager.discovery_service.get_agents_by_capability("code_analysis")
        for agent in code_agents:
            print(f"- {agent.name}")
        
        return True
        
    finally:
        await engine.shutdown()


async def test_agent_decision_making():
    """Test agent decision making in UPEE phases."""
    print("\n\n=== Testing Agent Decision Making ===")
    
    settings = Settings()
    engine = UPEEEngine(settings)
    
    try:
        await engine.startup()
        
        # Inject mock agents
        mock_output = await mock_pixell_list_output()
        agents = engine.agent_manager.discovery_service._parse_pixell_output(mock_output)
        engine.agent_manager.discovery_service.registry.agents = agents
        
        # Test scenarios for different phases
        test_contexts = [
            {
                "phase": "UNDERSTAND",
                "context": {
                    "message": "Analyze this complex codebase with 50 files",
                    "files": [{"type": "code", "name": f"file{i}.py"} for i in range(50)]
                }
            },
            {
                "phase": "PLAN",
                "context": {
                    "understanding_result": {
                        "metadata": {
                            "intent": "architecture_design",
                            "complexity": "very_complex"
                        }
                    }
                }
            },
            {
                "phase": "EXECUTE",
                "context": {
                    "plan_result": {
                        "metadata": {
                            "needs_external_calls": True,
                            "execution_types": ["data_processing", "code_generation"]
                        }
                    }
                }
            }
        ]
        
        for test in test_contexts:
            phase = test["phase"]
            context = test["context"]
            
            print(f"\n\nTesting {phase} phase decision:")
            print(f"Context: {json.dumps(context, indent=2)}")
            
            # Get decision
            from src.schemas import UPEEPhase
            phase_enum = UPEEPhase[phase]
            decision = await engine.agent_manager.should_use_agent(phase_enum, context)
            
            if decision:
                print(f"\nDecision: Use agent '{decision.agent_id}'")
                print(f"Capability: {decision.capability}")
                print(f"Reasoning: {decision.reasoning}")
                print(f"Confidence: {decision.confidence}")
            else:
                print("\nDecision: No external agent needed")
        
        return True
        
    finally:
        await engine.shutdown()


async def test_upee_with_agents():
    """Test full UPEE flow with agent integration."""
    print("\n\n=== Testing UPEE Flow with Agents ===")
    
    settings = Settings()
    engine = UPEEEngine(settings)
    
    try:
        await engine.startup()
        
        # Inject mock agents
        mock_output = await mock_pixell_list_output()
        agents = engine.agent_manager.discovery_service._parse_pixell_output(mock_output)
        engine.agent_manager.discovery_service.registry.agents = agents
        
        # Create a complex request that should trigger agent usage
        request = ChatRequest(
            message="I need to analyze and refactor this large codebase with 100+ files to improve performance",
            files=[
                FileContext(
                    name=f"module_{i}.py",
                    content=f"# Module {i} code here\nclass Module{i}:\n    pass",
                    type="python"
                ) for i in range(10)  # Simplified for testing
            ],
            model="gpt-4",
            show_thinking=True
        )
        
        print("\nProcessing request through UPEE with agent support...")
        print(f"Request: {request.message}")
        print(f"Files: {len(request.files)} Python files")
        
        # Process through UPEE
        events = []
        async for event in engine.process_request(request):
            events.append(event)
            
            if event.get("event") == "thinking":
                data = json.loads(event["data"])
                print(f"\n[{data['phase']}] {data['content']}")
            elif event.get("event") == "content":
                print("\n[CONTENT] Response chunk received")
            elif event.get("event") == "complete":
                print("\n[COMPLETE] Processing finished")
        
        # Get agent usage stats
        stats = engine.agent_manager.get_agent_stats()
        print(f"\n\nAgent Usage Statistics:")
        print(f"- Total agent requests: {stats['total_requests']}")
        print(f"- Successful requests: {stats['successful_requests']}")
        print(f"- Active requests: {stats['active_requests']}")
        print(f"- Available agents: {stats['available_agents']}")
        
        return True
        
    finally:
        await engine.shutdown()


async def main():
    """Run all tests."""
    print("=== Agent Integration Test Suite ===")
    
    tests = [
        ("Agent Discovery", test_agent_discovery),
        ("Agent Decision Making", test_agent_decision_making),
        ("UPEE Flow with Agents", test_upee_with_agents)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\n\nError in {test_name}: {e}")
            results.append((test_name, "ERROR"))
    
    print("\n\n=== Test Results ===")
    for test_name, result in results:
        print(f"{test_name}: {result}")


if __name__ == "__main__":
    asyncio.run(main())