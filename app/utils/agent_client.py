"""A2A (Agent-to-Agent) 클라이언트 구현 - 공식 A2A 프로토콜 표준 준수."""

import asyncio
import httpx
import json
import time
from typing import Any, Optional
from app.utils.logging_config import get_logger
from a2a.types import Message, Part, Role, SendMessageRequest, TextPart, FilePart, MessageSendParams, SendMessageResponse, JSONRPCErrorResponse, SendMessageSuccessResponse
from a2a.client import A2AClient
from app.schemas import FileContent
from a2a.types import FileWithBytes
import uuid


logger = get_logger("a2a_client")

class AgentClient:
	"""A2A 서버와 통신하는 클라이언트 - 공식 A2A 프로토콜 표준 기반."""
	
	def __init__(self, server_url: str):
		self.server_url = server_url
		self.logger = logger
	
	async def discover_agents(self) -> list[dict[str, Any]]:
		"""A2A 서버에서 사용 가능한 에이전트들을 탐색합니다."""
		try:
			# Standard A2A agent discovery endpoint
			async with httpx.AsyncClient(timeout=None) as client:
				response = await client.get(f"{self.server_url}/.well-known/agent.json")
				response.raise_for_status()
				
				agent_card = response.json()
				self.logger.info(f"Discovered agent: {agent_card.get('name', 'Unknown')}")
				return [agent_card]
				
		except httpx.TimeoutException:
			self.logger.error("Timeout while discovering agents")
			return []
		except httpx.HTTPStatusError as e:
			self.logger.error(f"HTTP error while discovering agents: {e.response.status_code}")
			return []
		except Exception as e:
			self.logger.error(f"Error discovering agents: {e}")
			return []
	
	async def fetch_cards(self) -> list[dict[str, Any]]:
		"""A2A 서버로부터 사용 가능한 카드 목록을 가져옵니다."""
		# A2A 표준에 따라 agent discovery 사용
		return await self.discover_agents()
	
	async def get_agent_card(self) -> Optional[dict[str, Any]]:
		"""단일 에이전트의 카드 정보를 가져옵니다."""
		try:
			async with httpx.AsyncClient(timeout=None) as client:
				response = await client.get(f"{self.server_url}/.well-known/agent.json")
				response.raise_for_status()
				
				agent_card = response.json()
				self.logger.info(f"Fetched agent card: {agent_card.get('name', 'Unknown')}")
				return agent_card
				
		except httpx.TimeoutException:
			self.logger.error("Timeout while fetching agent card")
			return None
		except httpx.HTTPStatusError as e:
			self.logger.error(f"HTTP error while fetching agent card: {e.response.status_code}")
			return None
		except Exception as e:
			self.logger.error(f"Error fetching agent card: {e}")
			return None
	
	async def get_card_by_id(self, card_id: str) -> Optional[dict[str, Any]]:
		"""특정 카드 ID로 카드 정보를 가져옵니다."""
		# 단일 에이전트의 경우 카드 ID와 상관없이 동일한 카드 반환
		return await self.get_agent_card()
	
	async def send_message(self, message: dict[str, Any]) -> JSONRPCErrorResponse | SendMessageSuccessResponse:
		"""A2A 서버로 메시지를 전송합니다"""

		if message.get("type") == "skill_request": # TODO: SKILL request가 아닌경우 처리
			try:
				async with httpx.AsyncClient(timeout=None) as httpx_client:
					file: FileContent
					client = A2AClient(url=self.server_url, httpx_client=httpx_client)
				
					message_parts = [Part(TextPart(text=json.dumps(message.get("parameters", {}), ensure_ascii=False)))]
					files = message.get("files", [])
					for file in files or []:
						file_name = file.file_name
						file_content = file.content
						bytes_content = file_content.encode("utf-8")
						message_parts.append(Part(FilePart(file=FileWithBytes(bytes=bytes_content, metadata={"file_name": file_name}))))

					payload = SendMessageRequest(
						id=str(uuid.uuid4()),
						params=MessageSendParams(
							message=Message(
								role=Role.user,
								parts=message_parts,
								messageId=str(uuid.uuid4()),
								metadata={"skill": message.get("skill_id"), "params": message.get("parameters", {})}
							)
						)
					)

					resp = await client.send_message(payload, http_kwargs={"timeout": None})
					return resp.root

			except Exception as e:
				self.logger.error(f"Error sending message via official A2A SDK: {e}")
				return {"status": "error", "error": str(e)}
	
	async def health_check(self) -> bool:
		"""A2A 서버의 상태를 확인합니다."""
		try:
			async with httpx.AsyncClient(timeout=None) as client:
				# A2A 표준 health check 또는 agent card endpoint 확인
				response = await client.get(f"{self.server_url}/.well-known/agent.json")
				return response.status_code == 200
		except Exception:
			return False 