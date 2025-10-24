from fastapi import APIRouter
from src.core.activity_manger import ActivityManager

router = APIRouter()


@router.get("/")
async def activity_manager():
	"""Manage activities."""
	return ActivityManager.activities