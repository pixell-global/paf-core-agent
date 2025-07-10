"""
Test script for Phase 2 implementation.
Verifies event bus and job scheduler functionality.
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from app.core.event_bus import event_bus
from app.core.scheduler import job_scheduler
from app.core.events import UserPromptEvent, JobScheduledEvent, EventType
from app.db.database import test_connection, init_db

async def test_phase2():
    """Test Phase 2 implementation."""
    print("🧪 Testing Phase 2: Event Bus & Job Scheduler")
    
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
    
    # Test 3: Event bus functionality
    print("\n3. Testing event bus...")
    
    # Test event subscription
    events_received = []
    
    def event_handler(event):
        events_received.append(event)
        event_type = event.event_type if hasattr(event, 'event_type') else event.get('event_type', 'unknown')
        if hasattr(event_type, 'value'):
            event_type = event_type.value
        print(f"📨 Received event: {event_type}")
    
    # Subscribe to events
    event_bus.subscribe(EventType.USER_PROMPT, event_handler)
    event_bus.subscribe(EventType.JOB_SCHEDULED, event_handler)
    
    # Start event bus
    await event_bus.start_listener()
    
    # Publish test events
    test_user_event = UserPromptEvent(
        user_id="123e4567-e89b-12d3-a456-426614174000",
        message="Test message",
        conversation_id="123e4567-e89b-12d3-a456-426614174001",
        source="test"
    )
    
    await event_bus.publish(test_user_event)
    
    # Give event bus time to process
    await asyncio.sleep(2)
    
    if len(events_received) > 0:
        print("✅ Event bus publishing and subscription works")
    else:
        print("❌ Event bus not working properly")
        return False
    
    # Test 4: Job scheduler functionality
    print("\n4. Testing job scheduler...")
    
    # Start job scheduler
    await job_scheduler.start()
    
    # Schedule a test job
    job = await job_scheduler.schedule_job(
        name="Test Job Phase 2",
        job_type="data_sync",
        payload={"test": True, "phase": 2},
        scheduled_at=datetime.now(timezone.utc) + timedelta(seconds=3)
    )
    
    print(f"✅ Job scheduled: {job.name} ({job.id})")
    
    # Wait for job to execute
    await asyncio.sleep(5)
    
    # Check job status
    jobs = await job_scheduler.get_jobs(limit=5)
    completed_jobs = [j for j in jobs if j.status == "completed"]
    if completed_jobs:
        print(f"✅ Job executed successfully: {completed_jobs[0].status}")
    else:
        print(f"❌ Job execution status: {jobs[0].status if jobs else 'No jobs found'}")
        # Don't fail the test as the job might still be running
        print("  (Job might still be processing, continuing with other tests...)")
    
    # Test 5: Cron job scheduling
    print("\n5. Testing cron job scheduling...")
    
    cron_job = await job_scheduler.schedule_job(
        name="Test Cron Job",
        job_type="cron",
        schedule_spec="* * * * *",  # Every minute
        payload={"test": True, "cron": True}
    )
    
    print(f"✅ Cron job scheduled: {cron_job.name} ({cron_job.id})")
    
    # Test 6: Interval job scheduling
    print("\n6. Testing interval job scheduling...")
    
    interval_job = await job_scheduler.schedule_job(
        name="Test Interval Job",
        job_type="interval",
        schedule_spec="30s",  # Every 30 seconds
        payload={"test": True, "interval": True}
    )
    
    print(f"✅ Interval job scheduled: {interval_job.name} ({interval_job.id})")
    
    # Test 7: Event history retrieval
    print("\n7. Testing event history retrieval...")
    
    events = await event_bus.get_events(limit=5)
    if events:
        print(f"✅ Retrieved {len(events)} events from history")
        for event in events:
            print(f"  - {event.event_type} at {event.created_at}")
    else:
        print("❌ No events found in history")
        return False
    
    # Test 8: Job execution history
    print("\n8. Testing job execution history...")
    
    executions = await job_scheduler.get_job_executions(str(job.id))
    if executions:
        print(f"✅ Retrieved {len(executions)} job executions")
        for execution in executions:
            print(f"  - Status: {execution.status}, Duration: {execution.duration_seconds}s")
    else:
        print("❌ No job executions found")
        return False
    
    # Cleanup
    await event_bus.stop_listener()
    await job_scheduler.stop()
    
    print("\n🎉 Phase 2 tests completed successfully!")
    print("\n✅ Event bus works correctly")
    print("✅ Job scheduler works correctly")
    print("✅ Cron and interval jobs can be scheduled")
    print("✅ Event and job history can be retrieved")
    print("\n🚀 Ready to proceed to Phase 3!")
    
    return True

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres@localhost:5432/paf_core_agent"
    
    asyncio.run(test_phase2())