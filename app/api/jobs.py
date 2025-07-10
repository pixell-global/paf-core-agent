"""
Job management API endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.core.scheduler import job_scheduler
from app.schemas import JobCreate, JobResponse

router = APIRouter()


@router.post("/", response_model=JobResponse)
async def create_job(
    job_data: JobCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new job."""
    try:
        job = await job_scheduler.schedule_job(
            name=job_data.name,
            job_type=job_data.job_type,
            payload=job_data.payload,
            schedule_spec=job_data.schedule_spec,
            max_retries=job_data.max_retries,
            timeout_seconds=job_data.timeout_seconds
        )
        
        return JobResponse(
            id=job.id,
            name=job.name,
            job_type=job.job_type,
            schedule_spec=job.schedule_spec,
            status=job.status,
            payload=job.payload,
            created_at=job.created_at,
            scheduled_at=job.scheduled_at,
            max_retries=job.max_retries
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[JobResponse])
async def get_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(100, ge=1, le=1000, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db: AsyncSession = Depends(get_db)
):
    """Get list of jobs."""
    try:
        jobs = await job_scheduler.get_jobs(
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset
        )
        
        return [
            JobResponse(
                id=job.id,
                name=job.name,
                job_type=job.job_type,
                schedule_spec=job.schedule_spec,
                status=job.status,
                payload=job.payload,
                created_at=job.created_at,
                scheduled_at=job.scheduled_at,
                max_retries=job.max_retries
            )
            for job in jobs
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific job."""
    try:
        jobs = await job_scheduler.get_jobs(limit=1)
        job = next((j for j in jobs if j.id == job_id), None)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobResponse(
            id=job.id,
            name=job.name,
            job_type=job.job_type,
            schedule_spec=job.schedule_spec,
            status=job.status,
            payload=job.payload,
            created_at=job.created_at,
            scheduled_at=job.scheduled_at,
            max_retries=job.max_retries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}")
async def cancel_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Cancel a job."""
    try:
        success = await job_scheduler.cancel_job(str(job_id))
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": "Job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/executions")
async def get_job_executions(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get execution history for a job."""
    try:
        executions = await job_scheduler.get_job_executions(str(job_id))
        
        return [
            {
                "id": execution.id,
                "job_id": execution.job_id,
                "status": execution.status,
                "result": execution.result,
                "error_message": execution.error_message,
                "started_at": execution.started_at,
                "completed_at": execution.completed_at,
                "duration_seconds": execution.duration_seconds
            }
            for execution in executions
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def create_test_job(
    db: AsyncSession = Depends(get_db)
):
    """Create a test job for demonstration purposes."""
    try:
        job = await job_scheduler.schedule_job(
            name="Test Job",
            job_type="data_sync",
            payload={"test": True, "timestamp": datetime.now().isoformat()},
            scheduled_at=datetime.now()
        )
        
        return {
            "message": "Test job created successfully",
            "job_id": job.id,
            "status": job.status
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))