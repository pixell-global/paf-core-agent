from fastapi import APIRouter, Query
from app.core.activity_manger import ActivityManager, ActivityContents

router = APIRouter()


@router.get("/")
async def activity_manager(organization_id: str = Query(..., description="Organization ID (char(36))")):
	"""List activities for an organization."""
	return [r.model_dump() for r in await ActivityManager.get_activities(organization_id)]


@router.post("/")
async def create_activity(
	organization_id: str = Query(..., description="Organization ID (char(36))"),
	contents: ActivityContents | None = None,
):
	"""Create an activity for an organization."""
	if contents is None:
		contents = ActivityContents(type="none", data=None)
	activity_id = await ActivityManager.add_activity(organization_id, contents)
	return {"id": activity_id}