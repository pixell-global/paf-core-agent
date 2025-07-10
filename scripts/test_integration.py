#!/usr/bin/env python3
"""
Integration test for memory and file features with the full server.
"""

import asyncio
import json
import os
import sys
import httpx
from datetime import datetime

# Test data
test_chat_request = {
    "message": "Analyze this Python code and explain what it does, considering our previous conversation",
    "files": [
        {
            "file_name": "hello.py",
            "content": "def greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('World'))",
            "file_type": "text/x-python",
            "file_size": 65
        }
    ],
    "history": [
        {
            "role": "user",
            "content": "I'm learning Python programming",
            "timestamp": datetime.now().isoformat()
        },
        {
            "role": "assistant", 
            "content": "That's great! Python is an excellent language to learn. What would you like to explore first?",
            "timestamp": datetime.now().isoformat()
        }
    ],
    "memory_limit": 5,
    "show_thinking": True,
    "temperature": 0.7
}


async def test_server_integration():
    """Test the server integration with new features."""
    print("🧪 Testing Server Integration...")
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test health endpoint
            print("📍 Testing health endpoint...")
            health_response = await client.get(f"{base_url}/api/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health check passed. Status: {health_data.get('status')}")
            else:
                print(f"❌ Health check failed: {health_response.status_code}")
                return False
            
            # Test chat status endpoint
            print("📍 Testing chat status endpoint...")
            status_response = await client.get(f"{base_url}/api/chat/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                features = status_data.get('features', {})
                print(f"✅ Chat status check passed.")
                print(f"  - Short-term memory: {features.get('short_term_memory', False)}")
                print(f"  - Signed URL support: {features.get('signed_url_support', False)}")
                print(f"  - Conversation history: {features.get('conversation_history', False)}")
            else:
                print(f"❌ Chat status check failed: {status_response.status_code}")
                return False
                
            # Test streaming chat with new features
            print("📍 Testing streaming chat with files and memory...")
            async with client.stream(
                "POST",
                f"{base_url}/api/chat/stream",
                json=test_chat_request,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status_code == 200:
                    print("✅ Chat stream started successfully")
                    
                    event_count = 0
                    async for line in response.aiter_lines():
                        if line.strip():
                            event_count += 1
                            if event_count <= 5:  # Show first few events
                                print(f"  Event {event_count}: {line[:100]}...")
                    
                    print(f"✅ Received {event_count} events from stream")
                    return True
                else:
                    error_text = await response.aread()
                    print(f"❌ Chat stream failed: {response.status_code}")
                    print(f"Error: {error_text.decode()}")
                    return False
                    
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


async def test_validation_endpoint():
    """Test validation with the actual endpoint."""
    print("🧪 Testing API Validation...")
    
    base_url = "http://localhost:8000"
    
    # Test invalid requests
    invalid_requests = [
        {
            "message": "",  # Empty message
        },
        {
            "message": "Test",
            "files": [
                {
                    "file_name": "",  # Empty file name
                    "content": "test",
                    "file_type": "text/plain",
                    "file_size": 4
                }
            ]
        },
        {
            "message": "Test",
            "history": [
                {
                    "role": "invalid_role",  # Invalid role
                    "content": "test"
                }
            ]
        },
        {
            "message": "Test", 
            "memory_limit": 25  # Exceeds limit
        }
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, invalid_request in enumerate(invalid_requests):
            try:
                response = await client.post(
                    f"{base_url}/api/chat/stream",
                    json=invalid_request
                )
                if response.status_code == 400:
                    error_data = response.json()
                    print(f"✅ Validation test {i+1} correctly rejected: {error_data.get('detail', 'Unknown error')}")
                else:
                    print(f"❌ Validation test {i+1} should have failed but got: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Validation test {i+1} error: {e}")


def check_server_running():
    """Check if the server is running."""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except:
        return False


async def main():
    """Run integration tests."""
    print("🚀 Starting Integration Tests for Memory and File Features\n")
    
    if not check_server_running():
        print("❌ Server is not running on localhost:8000")
        print("Please start the server with: ./scripts/start.sh")
        return 1
    
    try:
        # Run integration tests
        success = await test_server_integration()
        if not success:
            return 1
            
        await test_validation_endpoint()
        
        print("\n🎉 All integration tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)