from typing import Any, List
from pydantic import BaseModel
import json
import uuid

from app.db.database import AsyncSessionLocal
from app.db.models.core import Activity as ActivityModel


class ActivityContents(BaseModel):
	"""Activity contents."""
	type: str
	data: Any


class ActivityRecord(BaseModel):
	"""Pydantic view of an Activity row with parsed contents."""
	id: str
	organization_id: str
	contents: ActivityContents


class ActivityManager:
	"""Activity manager using SQLAlchemy (async)."""

	@staticmethod
	async def add_activity(organization_id: str, contents: ActivityContents) -> str:
		"""Create an activity row for an organization and return its id."""
		activity_id = str(uuid.uuid4())
		contents_str = json.dumps(contents.model_dump())

		async with AsyncSessionLocal() as session:
			try:
				row = ActivityModel(
					id=activity_id,
					organization_id=organization_id,
					activity_contents=contents_str,
				)
				session.add(row)
				await session.commit()
				return activity_id
			except Exception:
				await session.rollback()
				raise

	@staticmethod
	async def get_activities(organization_id: str) -> List[ActivityRecord]:
		"""Return activities for an organization with parsed contents."""
		async with AsyncSessionLocal() as session:
			from sqlalchemy import select

			result = await session.execute(
				select(ActivityModel).where(ActivityModel.organization_id == organization_id)
			)
			rows: List[ActivityModel] = result.scalars().all()

			records: List[ActivityRecord] = []
			for row in rows:
				try:
					parsed = json.loads(row.activity_contents or "{}")
				except Exception:
					parsed = {}
				records.append(
					ActivityRecord(
						id=row.id,
						organization_id=row.organization_id,
						contents=ActivityContents(**parsed),
					)
				)
			return records