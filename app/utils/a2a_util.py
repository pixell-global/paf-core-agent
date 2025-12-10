from a2a.types   import AgentSkill, MessageSendParams, Message
from a2a.types import Role, Part, TextPart
from a2a.client  import A2AClient
from a2a.client.helpers import create_text_message_object
from uuid import uuid4
import asyncio, uuid, json, httpx

async def run(self, user_id, skill_id, product_id, a2a_url):
	kwargs = {}
	kwargs["user_id"] = user_id
	kwargs["name"] = product_id
	kwargs["description"] = "gooood"
	meta = {"skill": skill_id, "params": kwargs}

	# timeout=None 로 설정해 HTTP 타임아웃 제거
	async with httpx.AsyncClient(timeout=None) as sess:
		# A2A 서버 엔드포인트는 루트(`/`)에 POST 요청을 받도록 구성돼 있으므로 별도 경로를 붙이지 않는다.
		client = A2AClient(url=a2a_url, httpx_client=sess)

		msg = Message(
			role=Role.user,
			parts=[Part(TextPart(text=json.dumps(kwargs, ensure_ascii=False)))],
			messageId=str(uuid4()),
			metadata=meta
		)

		# MessageSendParams: message / metadata 두 가지만 사용
		params = MessageSendParams(message=msg)

		from a2a.types import SendMessageRequest

		request = SendMessageRequest(id=str(uuid4()), params=params)

		# 내부에서도 timeout=None 을 명시적으로 전달
		resp = await client.send_message(request, http_kwargs={"timeout": None})
		return resp.model_dump()