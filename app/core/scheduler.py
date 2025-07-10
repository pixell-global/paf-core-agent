"""
Job scheduler implementation using APScheduler.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from app.db.database import AsyncSessionLocal
from app.db.models.jobs import Job, JobExecution
from app.core.event_bus import event_bus
from app.core.events import JobScheduledEvent, JobCompletedEvent, JobFailedEvent

logger = logging.getLogger(__name__)


class JobScheduler:
    """Job scheduler using APScheduler with database persistence."""
    
    def __init__(self):
        # Configure scheduler with proper timezone handling
        jobstores = {
            'default': MemoryJobStore()
        }
        job_defaults = {
            'coalesce': False,
            'max_instances': 3,
            'misfire_grace_time': 300
        }
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        self.running = False
        self.job_processors: Dict[str, callable] = {}
        self.queue_processor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the job scheduler."""
        if self.running:
            logger.warning("Job scheduler is already running")
            return
        
        self.scheduler.start()
        self.running = True
        
        # Start queue processor
        self.queue_processor_task = asyncio.create_task(self._process_job_queue())
        
        logger.info("Job scheduler started")
    
    async def stop(self) -> None:
        """Stop the job scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        self.scheduler.shutdown()
        logger.info("Job scheduler stopped")
    
    def register_job_processor(self, job_type: str, processor: callable) -> None:
        """Register a job processor for a specific job type."""
        self.job_processors[job_type] = processor
        logger.info(f"Registered job processor for type '{job_type}'")
    
    async def schedule_job(
        self,
        name: str,
        job_type: str,
        payload: Dict[str, Any],
        schedule_spec: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[int] = None
    ) -> Job:
        """Schedule a new job."""
        async with AsyncSessionLocal() as db:
            # Create job record
            job = Job(
                name=name,
                job_type=job_type,
                schedule_spec=schedule_spec,
                payload=payload,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                scheduled_at=scheduled_at or datetime.now(timezone.utc)
            )
            
            db.add(job)
            await db.commit()
            await db.refresh(job)
            
            # Schedule with APScheduler based on job type
            if job_type == "one_time":
                if scheduled_at:
                    trigger = DateTrigger(run_date=scheduled_at, timezone='UTC')
                else:
                    trigger = DateTrigger(run_date=datetime.now(timezone.utc), timezone='UTC')
                
                self.scheduler.add_job(
                    self._execute_job,
                    trigger=trigger,
                    args=[str(job.id)],
                    id=str(job.id),
                    replace_existing=True,
                    misfire_grace_time=300  # 5 minutes grace time
                )
            
            elif job_type == "cron" and schedule_spec:
                try:
                    trigger = CronTrigger.from_crontab(schedule_spec)
                    self.scheduler.add_job(
                        self._execute_job,
                        trigger=trigger,
                        args=[str(job.id)],
                        id=str(job.id),
                        replace_existing=True,
                        misfire_grace_time=300
                    )
                except ValueError as e:
                    logger.error(f"Invalid cron expression '{schedule_spec}': {e}")
                    job.status = "failed"
                    job.error_message = f"Invalid cron expression: {e}"
                    await db.commit()
                    return job
            
            elif job_type == "interval" and schedule_spec:
                try:
                    # Parse interval specification (e.g., "30s", "5m", "1h")
                    interval_seconds = self._parse_interval(schedule_spec)
                    trigger = IntervalTrigger(seconds=interval_seconds)
                    
                    self.scheduler.add_job(
                        self._execute_job,
                        trigger=trigger,
                        args=[str(job.id)],
                        id=str(job.id),
                        replace_existing=True,
                        misfire_grace_time=300
                    )
                except ValueError as e:
                    logger.error(f"Invalid interval specification '{schedule_spec}': {e}")
                    job.status = "failed"
                    job.error_message = f"Invalid interval specification: {e}"
                    await db.commit()
                    return job
            
            # Publish job scheduled event
            await event_bus.publish(JobScheduledEvent(
                job_id=job.id,
                job_name=job.name,
                job_type=job.job_type,
                source="job_scheduler"
            ))
            
            logger.info(f"Scheduled job '{name}' (ID: {job.id}) of type '{job_type}'")
            return job
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        try:
            self.scheduler.remove_job(job_id)
            
            # Update job status in database
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Job).where(Job.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = "cancelled"
                    await db.commit()
                    logger.info(f"Cancelled job {job_id}")
                    return True
                
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
        
        return False
    
    async def _execute_job(self, job_id: str) -> None:
        """Execute a job."""
        async with AsyncSessionLocal() as db:
            try:
                # Get job from database
                result = await db.execute(
                    select(Job).where(Job.id == job_id)
                )
                job = result.scalar_one_or_none()
                
                if not job:
                    logger.error(f"Job {job_id} not found")
                    return
                
                # Check if job processor is registered
                if job.job_type not in self.job_processors:
                    logger.error(f"No processor registered for job type '{job.job_type}'")
                    job.status = "failed"
                    job.error_message = f"No processor registered for job type '{job.job_type}'"
                    await db.commit()
                    return
                
                # Update job status
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                await db.commit()
                
                # Create job execution record
                execution = JobExecution(
                    job_id=job.id,
                    status="running"
                )
                db.add(execution)
                await db.commit()
                await db.refresh(execution)
                
                # Execute job
                processor = self.job_processors[job.job_type]
                
                try:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(job.payload)
                    else:
                        result = processor(job.payload)
                    
                    # Update job and execution status
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc)
                    job.result = result
                    
                    execution.status = "completed"
                    execution.completed_at = datetime.now(timezone.utc)
                    execution.result = result
                    
                    await db.commit()
                    
                    # Publish job completed event
                    await event_bus.publish(JobCompletedEvent(
                        job_id=job.id,
                        job_name=job.name,
                        job_type=job.job_type,
                        result=result,
                        source="job_scheduler"
                    ))
                    
                    logger.info(f"Job {job_id} completed successfully")
                
                except Exception as e:
                    # Handle job execution failure
                    logger.error(f"Job {job_id} execution failed: {e}")
                    
                    job.retry_count += 1
                    error_message = str(e)
                    
                    execution.status = "failed"
                    execution.completed_at = datetime.now(timezone.utc)
                    execution.error_message = error_message
                    
                    if job.retry_count < job.max_retries:
                        job.status = "queued"
                        job.error_message = None
                        # Reschedule after delay
                        retry_delay = min(300, 30 * (2 ** job.retry_count))  # Exponential backoff
                        retry_time = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
                        
                        self.scheduler.add_job(
                            self._execute_job,
                            trigger=DateTrigger(run_date=retry_time),
                            args=[job.id],
                            id=f"{job.id}_retry_{job.retry_count}",
                            replace_existing=True
                        )
                        
                        logger.info(f"Scheduled retry {job.retry_count} for job {job_id}")
                    else:
                        job.status = "failed"
                        job.error_message = error_message
                        job.completed_at = datetime.now(timezone.utc)
                        
                        # Publish job failed event
                        await event_bus.publish(JobFailedEvent(
                            job_id=job.id,
                            job_name=job.name,
                            job_type=job.job_type,
                            error=error_message,
                            source="job_scheduler"
                        ))
                    
                    await db.commit()
                
            except Exception as e:
                logger.error(f"Error executing job {job_id}: {e}")
    
    async def _process_job_queue(self) -> None:
        """Process queued jobs periodically."""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    # Get queued jobs that are ready to run
                    result = await db.execute(
                        select(Job).where(
                            Job.status == "queued",
                            Job.scheduled_at <= datetime.now(timezone.utc)
                        ).limit(10)
                    )
                    
                    jobs = result.scalars().all()
                    
                    for job in jobs:
                        # Schedule job for immediate execution
                        self.scheduler.add_job(
                            self._execute_job,
                            trigger=DateTrigger(run_date=datetime.now(timezone.utc)),
                            args=[str(job.id)],
                            id=f"{job.id}_queued",
                            replace_existing=True
                        )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing job queue: {e}")
                await asyncio.sleep(5)
    
    def _parse_interval(self, spec: str) -> int:
        """Parse interval specification into seconds."""
        spec = spec.strip().lower()
        
        if spec.endswith('s'):
            return int(spec[:-1])
        elif spec.endswith('m'):
            return int(spec[:-1]) * 60
        elif spec.endswith('h'):
            return int(spec[:-1]) * 3600
        elif spec.endswith('d'):
            return int(spec[:-1]) * 86400
        else:
            # Assume seconds if no unit specified
            return int(spec)
    
    async def get_jobs(
        self, 
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """Get jobs from the database."""
        async with AsyncSessionLocal() as db:
            query = select(Job)
            
            if status:
                query = query.where(Job.status == status)
            if job_type:
                query = query.where(Job.job_type == job_type)
            
            query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)
            
            result = await db.execute(query)
            return result.scalars().all()
    
    async def get_job_executions(self, job_id: str) -> List[JobExecution]:
        """Get execution history for a job."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(JobExecution)
                .where(JobExecution.job_id == job_id)
                .order_by(JobExecution.started_at.desc())
            )
            return result.scalars().all()


# Global job scheduler instance
job_scheduler = JobScheduler()


# Example job processors
async def example_data_sync_processor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example processor for data synchronization jobs."""
    logger.info(f"Processing data sync job with payload: {payload}")
    
    # Simulate work
    await asyncio.sleep(2)
    
    return {
        "success": True,
        "records_processed": 100,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def example_report_generator_processor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example processor for report generation jobs."""
    logger.info(f"Processing report generation job with payload: {payload}")
    
    # Simulate work
    await asyncio.sleep(5)
    
    return {
        "success": True,
        "report_path": "/tmp/report.pdf",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Register example processors
job_scheduler.register_job_processor("data_sync", example_data_sync_processor)
job_scheduler.register_job_processor("report_generation", example_report_generator_processor)