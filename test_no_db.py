#!/usr/bin/env python3
"""
Test script to verify PAF Core Agent runs without database dependencies.
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that core components import without database dependencies."""
    print("🧪 Testing core imports...")
    
    try:
        from app.main import app
        from app.api.chat import router as chat_router
        from app.api.health import router as health_router
        from app.api.debug import router as debug_router
        from app.settings import Settings
        print("✅ All core components import successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("🧪 Testing FastAPI app creation...")
    
    try:
        from app.main import app
        # Check that core routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/api/chat/", "/api/health/", "/api/debug/"]
        
        for expected in expected_routes:
            if not any(expected in route for route in routes):
                print(f"❌ Missing route: {expected}")
                return False
        
        print("✅ FastAPI app created successfully with core routes")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

def test_server_startup():
    """Test that the server can start without database."""
    print("🧪 Testing server startup...")
    
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app.main:app", 
            "--host", "127.0.0.1", "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root)
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8001/api/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Server started successfully and health check passed")
            success = True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            success = False
            
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        
        return success
        
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            pass
        return False

def main():
    """Run all tests."""
    print("🚀 Testing PAF Core Agent without database dependencies\n")
    
    tests = [
        ("Import Test", test_imports),
        ("App Creation Test", test_app_creation),
        ("Server Startup Test", test_server_startup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PAF Core Agent works without database dependencies.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())