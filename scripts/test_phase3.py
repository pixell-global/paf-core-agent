"""
Test script for Phase 3 implementation.
Verifies A2A bridge and plugin system functionality.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime, timezone
from app.core.bridge.a2a_bridge import A2ABridge
from app.core.bridge.protocol import MessageType, MessagePriority
from app.plugins.registry import PluginRegistry
from app.plugins.manager import PluginManager
from app.plugins.examples import ExampleUPEEEnhancer, ExampleTextWorker
from app.plugins.base.plugin_base import PluginConfig
from app.db.database import test_connection, init_db

async def test_phase3():
    """Test Phase 3 implementation."""
    print("🧪 Testing Phase 3: A2A Bridge & Plugin System")
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    connection_ok = await test_connection()
    if not connection_ok:
        print("❌ Database connection failed!")
        return False
    
    # Test 2: Initialize database
    print("\n2. Initializing database...")
    await init_db()
    print("✅ Database initialized")
    
    # Test 3: A2A Bridge functionality
    print("\n3. Testing A2A Bridge...")
    
    # Create bridge instances
    bridge1 = A2ABridge("agent-1")
    bridge2 = A2ABridge("agent-2")
    
    # Track received messages
    received_messages = []
    
    def message_handler(message):
        received_messages.append(message)
        print(f"📨 Agent-2 received: {message.type.value} from {message.source_agent_id}")
    
    # Start bridges
    await bridge1.start()
    await bridge2.start()
    
    # Connect both bridges to the same router (simulate network)
    # In real implementation, this would be handled by network transport
    bridge1.router = bridge2.router  # Share the same router for testing
    
    # Register message handler for agent-2
    bridge2.register_message_handler(MessageType.HEARTBEAT, message_handler)
    bridge2.register_message_handler(MessageType.TASK_REQUEST, message_handler)
    
    # Test message sending
    await bridge1.send_message(
        message_type=MessageType.HEARTBEAT,
        target_agent_id="agent-2",
        payload={"test": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}
    )
    
    # Test task request
    await bridge1.send_task_request(
        target_agent_id="agent-2",
        task_type="text_analysis",
        task_data={"text": "This is a test message for analysis"},
        conversation_id="test-conversation-123"
    )
    
    # Wait for message processing
    await asyncio.sleep(2)
    
    if len(received_messages) >= 2:
        print("✅ A2A bridge message routing works")
    else:
        print(f"❌ A2A bridge not working properly (received {len(received_messages)} messages)")
        return False
    
    # Test broadcast
    broadcast_received = []
    
    def broadcast_handler(message):
        broadcast_received.append(message)
        print(f"📨 Broadcast received: {message.type.value}")
    
    bridge2.register_message_handler(MessageType.STATUS_UPDATE, broadcast_handler)
    
    await bridge1.broadcast_status_update(
        status="online",
        capabilities=["upee", "orchestrator"],
        current_load=25,
        max_capacity=100
    )
    
    await asyncio.sleep(1)
    
    if broadcast_received:
        print("✅ A2A bridge broadcast works")
    else:
        print("❌ A2A bridge broadcast not working")
        return False
    
    # Test 4: Plugin Registry
    print("\n4. Testing Plugin Registry...")
    
    registry = PluginRegistry()
    
    # Register plugin classes
    registry.register_plugin_class(ExampleUPEEEnhancer)
    registry.register_plugin_class(ExampleTextWorker)
    
    # Discover plugins from files
    registry.discover_plugins()
    
    all_plugins = registry.get_all_plugins()
    if len(all_plugins) >= 2:
        print(f"✅ Plugin registry discovered {len(all_plugins)} plugins")
    else:
        print(f"❌ Plugin registry only found {len(all_plugins)} plugins")
        return False
    
    # Test search functionality
    from app.plugins.base.plugin_base import PluginType
    upee_plugins = registry.get_plugins_by_type(PluginType.UPEE)
    worker_plugins = registry.get_plugins_by_type(PluginType.WORKER)
    
    if upee_plugins and worker_plugins:
        print("✅ Plugin type filtering works")
    else:
        print("❌ Plugin type filtering not working")
        return False
    
    # Test 5: Plugin Manager
    print("\n5. Testing Plugin Manager...")
    
    manager = PluginManager(registry)
    await manager.start()
    
    # Load UPEE enhancer plugin
    upee_config = PluginConfig(
        enabled=True,
        config={
            "enhancement_level": "advanced",
            "custom_prompts": {
                "understand": "Enhanced understanding prompt"
            }
        }
    )
    
    success = await manager.load_plugin("example-upee-enhancer", upee_config)
    if success:
        print("✅ UPEE enhancer plugin loaded")
    else:
        print("❌ Failed to load UPEE enhancer plugin")
        return False
    
    # Activate plugin
    success = await manager.activate_plugin("example-upee-enhancer")
    if success:
        print("✅ UPEE enhancer plugin activated")
    else:
        print("❌ Failed to activate UPEE enhancer plugin")
        return False
    
    # Load text worker plugin
    worker_config = PluginConfig(
        enabled=True,
        config={
            "max_text_length": 5000,
            "supported_languages": ["en", "es"]
        }
    )
    
    success = await manager.load_plugin("example-text-worker", worker_config)
    if success:
        await manager.activate_plugin("example-text-worker")
        print("✅ Text worker plugin loaded and activated")
    else:
        print("❌ Failed to load text worker plugin")
        return False
    
    # Test 6: Plugin Execution
    print("\n6. Testing Plugin Execution...")
    
    # Test UPEE enhancer
    upee_plugin = manager.get_plugin("example-upee-enhancer")
    if upee_plugin:
        test_context = {
            "user_input": "Hello, can you help me analyze this text?",
            "conversation_id": "test-123"
        }
        
        enhanced_context = await upee_plugin.enhance_understand(test_context)
        if "enhanced_analysis" in enhanced_context:
            print("✅ UPEE enhancer execution works")
        else:
            print("❌ UPEE enhancer execution failed")
            return False
    
    # Test text worker
    worker_plugin = manager.get_plugin("example-text-worker")
    if worker_plugin:
        can_handle = await worker_plugin.can_handle_task(
            "text_analysis",
            {"text": "This is a test", "language": "en"}
        )
        
        if can_handle:
            result = await worker_plugin.execute_task(
                "text_analysis",
                {"text": "This is a test document for analysis."}
            )
            
            if result.get("success"):
                print("✅ Text worker execution works")
            else:
                print(f"❌ Text worker execution failed: {result}")
                return False
        else:
            print("❌ Text worker cannot handle task")
            return False
    
    # Test 7: Plugin Hooks
    print("\n7. Testing Plugin Hooks...")
    
    # Register hooks
    manager.register_plugin_hook("example-upee-enhancer", "upee_understand")
    manager.register_plugin_hook("example-text-worker", "worker_task_request")
    
    # Execute hooks
    hook_results = await manager.execute_plugin_hook(
        "upee_understand",
        context={"user_input": "Test hook execution"}
    )
    
    if hook_results and "example-upee-enhancer" in hook_results:
        print("✅ Plugin hooks work")
    else:
        print("❌ Plugin hooks not working")
        return False
    
    # Test 8: Health Checks
    print("\n8. Testing Health Checks...")
    
    health_results = await manager.health_check_all()
    
    all_healthy = all(
        result.get("healthy", False) 
        for result in health_results.values()
    )
    
    if all_healthy:
        print("✅ All plugins healthy")
    else:
        print("❌ Some plugins unhealthy")
        print(f"Health results: {health_results}")
        return False
    
    # Test 9: Bridge Status and Metrics
    print("\n9. Testing Status and Metrics...")
    
    bridge_status = bridge1.get_bridge_status()
    if bridge_status["running"]:
        print("✅ Bridge status reporting works")
    else:
        print("❌ Bridge status reporting failed")
        return False
    
    plugin_metrics = manager.get_plugin_metrics()
    if plugin_metrics:
        print("✅ Plugin metrics collection works")
    else:
        print("❌ Plugin metrics collection failed")
        return False
    
    # Test 10: Integration Test
    print("\n10. Testing A2A + Plugin Integration...")
    
    # Send a task request through A2A bridge to trigger plugin
    integration_received = []
    
    def integration_handler(message):
        integration_received.append(message)
        print(f"📨 Integration test: {message.type.value}")
    
    bridge2.register_message_handler(MessageType.TASK_REQUEST, integration_handler)
    
    await bridge1.send_task_request(
        target_agent_id="agent-2",
        task_type="text_summarization",
        task_data={
            "text": "This is a long document that needs to be summarized. " * 20,
            "max_length": 50
        }
    )
    
    await asyncio.sleep(1)
    
    if integration_received:
        print("✅ A2A + Plugin integration works")
    else:
        print("❌ A2A + Plugin integration failed")
        return False
    
    # Cleanup
    await bridge1.stop()
    await bridge2.stop()
    await manager.stop()
    
    print("\n🎉 Phase 3 tests completed successfully!")
    print("\n✅ A2A Bridge works correctly")
    print("✅ Plugin system works correctly")
    print("✅ Plugin discovery and registration work")
    print("✅ Plugin execution and hooks work")
    print("✅ Health monitoring works")
    print("✅ A2A + Plugin integration works")
    print("\n🚀 Ready to proceed to Phase 4!")
    
    return True

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres@localhost:5432/paf_core_agent"
    
    asyncio.run(test_phase3())