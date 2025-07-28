from typing import Any
from pydantic import BaseModel

class ActivityContents(BaseModel):
	"""Activity contents."""
	type: str
	data: Any

class Activity(BaseModel):
	"""Activity."""
	contents: ActivityContents


class ActivityManager:
	"""Activity manager."""
	activities: list[Activity] = []

	def __init__(self):
		pass

	@classmethod
	def add_activity(cls, activity: Activity):
		"""Add an activity."""
		cls.activities.append(activity)