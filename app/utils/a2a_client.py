"""A2A (Agent-to-Agent) 클라이언트 구현 - 공식 A2A 프로토콜 표준 준수."""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, Optional, List
from app.utils.logging_config import get_logger
# --- Official Google A2A SDK (`pip install a2a-sdk`) import 시도 ---

# 공식 Google A2A SDK (`a2a-sdk` → import namespace `a2a`)
from a2a.client import A2AClient as SDKA2AClient
# UPDATED IMPORTS – use official SDK "Types" namespace
from a2a.types import (
    Message as A2AMessage,
    Part as A2APart,
    Role as A2AMessageRole,
    SendMessageRequest as A2ASendMessageRequest,
)
import uuid


logger = get_logger("a2a_client")

class A2AClient:
    """A2A 서버와 통신하는 클라이언트 - 공식 A2A 프로토콜 표준 기반."""
    
    def __init__(self, server_url: str, timeout: int = 10):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logger
        self.agent_card_cache = None
        
        try:
            # 공식 SDK 클라이언트는 동기식이며, SDK가 httpx client 인젝션을 필수로 요구하므로 생성하여 전달
            self._sdk_httpx_client = httpx.Client(timeout=self.timeout)
            self.official_client = SDKA2AClient(
                url=f"{self.server_url.rstrip('/')}/a2a",
                httpx_client=self._sdk_httpx_client,
            )
            self.logger.info("Using official Google A2A SDK for communication")
        except Exception as e:
            # 초기화 실패 시 None 으로 설정하고 로그
            self.logger.error(f"Failed to initialize official A2A client: {e}")
            self.official_client = None
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """A2A 서버에서 사용 가능한 에이전트들을 탐색합니다."""
        try:
            # Standard A2A agent discovery endpoint
            async with httpx.AsyncClient(timeout=self.timeout) as client:
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
    
    async def fetch_cards(self) -> List[Dict[str, Any]]:
        """A2A 서버로부터 사용 가능한 카드 목록을 가져옵니다."""
        # A2A 표준에 따라 agent discovery 사용
        return await self.discover_agents()
    
    async def get_agent_card(self) -> Optional[Dict[str, Any]]:
        """단일 에이전트의 카드 정보를 가져옵니다."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
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
    
    async def get_card_by_id(self, card_id: str) -> Optional[Dict[str, Any]]:
        """특정 카드 ID로 카드 정보를 가져옵니다."""
        # 단일 에이전트의 경우 카드 ID와 상관없이 동일한 카드 반환
        return await self.get_agent_card()
    
    def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 서버로 메시지를 전송합니다 (동기). SDK 가 없으면 표준 JSON-RPC 로 폴백."""

        # 1) SDK 가 설치되어 있고, skill_request 타입인 경우 → 공식 SDK 사용
        if self.official_client and message.get("type") == "skill_request":
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self._send_skill_request_official(
                        message.get("skill_id"),
                        message.get("skill_name"),
                        message.get("parameters", {}),
                        message.get("user_message", ""),
                    )
                )
            except Exception as e:
                self.logger.error(f"Error sending message via official A2A SDK: {e}")
                # 폴백 시도

        # 2) 폴백: 표준 JSON-RPC 흐름 사용
        return self._send_message_a2a_standard(message)

    def _send_message_a2a_standard(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 프로토콜 표준을 따르는 메시지 전송."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # A2A JSON-RPC 2.0 표준 형식으로 메시지 구성
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": int(time.time() * 1000),  # 고유 ID 생성
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": str(message.get("payload", {}).get("message", ""))
                                }
                            ],
                            "messageId": f"msg_{int(time.time() * 1000000)}"
                        },
                        "metadata": {}
                    }
                }
                
                # A2A 프로토콜 표준 엔드포인트
                response = client.post(
                    f"{self.server_url}/",  # A2A 표준에 따른 루트 엔드포인트
                    json=rpc_request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                self.logger.info("Message sent via A2A standard protocol successfully")
                
                # A2A 표준 응답 구조에 따른 파싱
                if "result" in result:
                    return {
                        "status": "success", 
                        "response": result["result"],
                        "protocol": "a2a_standard"
                    }
                else:
                    return {
                        "status": "success", 
                        "response": result,
                        "protocol": "a2a_standard"
                    }
                
        except httpx.TimeoutException:
            self.logger.error("Timeout while sending A2A message")
            return {"status": "error", "error": "Timeout"}
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error while sending A2A message: {e.response.status_code}")
            return {"status": "error", "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            self.logger.error(f"Error sending A2A message: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _send_message_a2a_standard_async(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Async variant of standard A2A JSON-RPC send."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": int(time.time() * 1000),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": str(message.get("payload", {}).get("message", ""))
                                }
                            ],
                            "messageId": f"msg_{int(time.time() * 1000000)}"
                        },
                        "metadata": {}
                    }
                }

                response = await client.post(
                    f"{self.server_url}/",
                    json=rpc_request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()

                result = response.json()
                self.logger.info("Message sent via A2A standard protocol successfully (async)")

                if "result" in result:
                    return {
                        "status": "success",
                        "response": result["result"],
                        "protocol": "a2a_standard"
                    }
                else:
                    return {
                        "status": "success",
                        "response": result,
                        "protocol": "a2a_standard"
                    }

        except httpx.TimeoutException:
            self.logger.error("Timeout while sending A2A message (async)")
            return {"status": "error", "error": "Timeout"}
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error while sending A2A message (async): {e.response.status_code}")
            return {"status": "error", "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            self.logger.error(f"Error sending A2A message (async): {e}")
            return {"status": "error", "error": str(e)}

    # ---------------------------------------------------------------------
    # 내부: 공식 Google A2A SDK 사용한 skill_invocation 전송
    # ---------------------------------------------------------------------
    async def _send_skill_request_official(
        self,
        skill_id: str,
        skill_name: str,
        parameters: Dict[str, Any],
        user_message: str,
    ) -> Dict[str, Any]:
        """Use official Google A2A SDK to send a skill_invocation message. Returns dict response."""

        if not self.official_client:
            raise RuntimeError("Official A2A client not initialized")

        # 1) Build A2A message with skill invocation part (0.3+ spec)
        # 최신 a2a-sdk(0.3+)에서는 Part.kind 는 'text' | 'file' | 'data' 등 제한된 리터럴만 허용됩니다.
        # "skill_invocation" 전용 Part 는 더 이상 존재하지 않으므로, data-part 로 스킬 호출 정보를 전달합니다.
        part = A2APart(
            kind="data",
            data={
                "type": "skill_invocation",
                "skill": {
                    "id": skill_id,
                    "name": skill_name,
                    "parameters": parameters,
                },
                "user_message": user_message,
            },
        )

        # a2a-sdk 0.3+ requires a unique messageId for every Message
        message_id = str(uuid.uuid4())

        msg = A2AMessage(
            role=A2AMessageRole.user,
            parts=[part],
            messageId=message_id,  # snake_case field (alias for messageId)
            metadata={"source_message": user_message},
        )

        # A2ASendMessageRequest (from a2a.types) 는 JSON-RPC 2.0 요청 모델로,
        # 필수 필드(id, params)가 존재합니다. message 만 넘기면 Pydantic ValidationError 가 발생하므로
        # 요구 스펙에 맞게 id 와 params 를 함께 설정해줍니다.

        rpc = A2ASendMessageRequest(
            id=str(uuid.uuid4()),  # 고유 요청 ID
            params={
                "message": msg,
                "metadata": {},  # 스펙상 존재하지만 비워둘 수 있는 필드
            },
        )

        # 2) official_client.send 는 sync => run in executor
        loop = asyncio.get_running_loop()
        from functools import partial

        self.logger.info(
            "Sending skill invocation via official A2A SDK",
            skill_id=skill_id,
            skill_name=skill_name,
            parameters=parameters,
        )

        # SDKA2AClient 의 메서드명은 send_message 이므로 변경
        response = await loop.run_in_executor(None, partial(self.official_client.send_message, rpc))

        # response 는 Pydantic 모델 → dict 로 변환
        return {
            "status": "success",
            "response": response.model_dump() if hasattr(response, "model_dump") else response,
            "protocol": "a2a_official",
        }
    
    async def health_check(self) -> bool:
        """A2A 서버의 상태를 확인합니다."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # A2A 표준 health check 또는 agent card endpoint 확인
                response = await client.get(f"{self.server_url}/.well-known/agent.json")
                return response.status_code == 200
        except Exception:
            return False 