"""
Test script for Phase 1 implementation.
Verifies database connectivity, model creation, and basic functionality.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import AsyncSessionLocal, init_db, test_connection
from app.auth.models import AuthModels

async def test_phase1():
    """Test Phase 1 implementation."""
    print("🧪 Testing Phase 1: Database Foundation")
    
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
    
    # Test 3: User creation and authentication
    print("\n3. Testing user creation and authentication...")
    async with AsyncSessionLocal() as db:
        try:
            # Check if user already exists
            existing_user = await AuthModels.get_user_by_username(db, "test_phase1")
            if existing_user:
                user = existing_user
                print(f"✅ User already exists: {user.username} ({user.id})")
            else:
                # Create test user
                user = await AuthModels.create_user(
                    db=db,
                    username="test_phase1",
                    email="test@phase1.com",
                    password="testpassword123",
                    role="user"
                )
                print(f"✅ User created: {user.username} ({user.id})")
            
            # Test authentication
            auth_user = await AuthModels.authenticate_user(db, "test_phase1", "testpassword123")
            if auth_user and auth_user.id == user.id:
                print("✅ User authentication works")
            else:
                print("❌ User authentication failed")
                return False
            
            # Test API key creation
            api_key_obj, api_key = await AuthModels.create_api_key(
                db=db,
                user_id=user.id,
                name="Test Key",
                permissions={"read": True, "write": True}
            )
            print(f"✅ API key created: {api_key[:20]}...")
            
            # Test API key verification
            verified_user = await AuthModels.verify_api_key(db, api_key)
            if verified_user and verified_user.id == user.id:
                print("✅ API key verification works")
            else:
                print("❌ API key verification failed")
                return False
            
        except Exception as e:
            print(f"❌ User/API key test failed: {e}")
            return False
    
    # Test 4: Database models
    print("\n4. Testing database models...")
    async with AsyncSessionLocal() as db:
        try:
            from app.db.models.core import Conversation, Message
            from app.db.models.workers import WorkerInstance, WorkerTask
            from app.db.models.jobs import Job
            
            # Test conversation creation
            conversation = Conversation(
                user_id=user.id,
                title="Test Conversation",
                meta_data={"test": True}
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            print(f"✅ Conversation created: {conversation.id}")
            
            # Test message creation
            message = Message(
                conversation_id=conversation.id,
                role="user",
                content="Test message"
            )
            db.add(message)
            await db.commit()
            print(f"✅ Message created: {message.id}")
            
            # Test worker instance
            worker = WorkerInstance(
                name="test_worker",
                endpoint="localhost:50051",
                capabilities={"test": True}
            )
            db.add(worker)
            await db.commit()
            await db.refresh(worker)
            print(f"✅ Worker instance created: {worker.id}")
            
            # Test job creation
            job = Job(
                name="test_job",
                job_type="one_time",
                payload={"test": True}
            )
            db.add(job)
            await db.commit()
            print(f"✅ Job created: {job.id}")
            
        except Exception as e:
            print(f"❌ Database models test failed: {e}")
            return False
    
    print("\n🎉 Phase 1 tests completed successfully!")
    print("\n✅ Database foundation is ready")
    print("✅ User authentication works")
    print("✅ API key system works")
    print("✅ All database models work")
    print("\n🚀 Ready to proceed to Phase 2!")
    
    return True

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres@localhost:5432/paf_core_agent"
    
    asyncio.run(test_phase1())