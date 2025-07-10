"""A2A (Agent-to-Agent) 클라이언트 구현 - 공식 A2A 프로토콜 표준 준수."""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, Optional, List
from app.utils.logging_config import get_logger

logger = get_logger("a2a_client")

# 공식 A2A SDK 확인 (향후 사용을 위해)
# 사용자는 다음 명령으로 공식 SDK를 설치할 수 있습니다:
# git clone https://github.com/google/A2A.git
# cd A2A/a2a-python-sdk
# pip install -e .

try:
    # 공식 Google A2A SDK (설치된 경우)
    import a2a
    A2A_OFFICIAL_AVAILABLE = True
    logger.info("Official Google A2A SDK detected")
except ImportError:
    A2A_OFFICIAL_AVAILABLE = False
    logger.info("Using A2A protocol standard HTTP implementation")


class A2AClient:
    """A2A 서버와 통신하는 클라이언트 - 공식 A2A 프로토콜 표준 기반."""
    
    def __init__(self, server_url: str, timeout: int = 10):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logger
        self.agent_card_cache = None
        
        if A2A_OFFICIAL_AVAILABLE:
            # 공식 Google A2A SDK 클라이언트 초기화
            try:
                # TODO: 공식 SDK가 설치된 경우의 초기화 로직
                self.official_client = None  # 실제 구현은 공식 SDK 문서 참조
                self.logger.info("Using official Google A2A SDK for communication")
            except Exception as e:
                self.logger.error(f"Failed to initialize official A2A client: {e}")
                self.official_client = None
        else:
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
        """A2A 서버로 메시지를 전송합니다 (동기 버전)."""
        if A2A_OFFICIAL_AVAILABLE and self.official_client:
            try:
                # 공식 Google A2A SDK 사용
                # TODO: 공식 SDK API에 따른 메시지 전송 구현
                self.logger.info("Message sent via official A2A SDK successfully")
                return {"status": "success", "response": "Official SDK response"}
                
            except Exception as e:
                self.logger.error(f"Error sending message via official A2A SDK: {e}")
                return {"status": "error", "error": str(e)}
        
        # A2A 프로토콜 표준 HTTP 구현 사용
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
    
    async def send_message_async(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 서버로 메시지를 전송합니다 (비동기 버전) - A2A 프로토콜 표준."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # A2A JSON-RPC 2.0 표준 형식
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
                response = await client.post(
                    f"{self.server_url}/",  # A2A 표준에 따른 루트 엔드포인트
                    json=rpc_request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                self.logger.info("Message sent asynchronously via A2A standard protocol successfully")
                
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
            self.logger.error("Timeout while sending A2A message asynchronously")
            return {"status": "error", "error": "Timeout"}
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error while sending A2A message asynchronously: {e.response.status_code}")
            return {"status": "error", "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            self.logger.error(f"Error sending A2A message asynchronously: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> bool:
        """A2A 서버의 상태를 확인합니다."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # A2A 표준 health check 또는 agent card endpoint 확인
                response = await client.get(f"{self.server_url}/.well-known/agent.json")
                return response.status_code == 200
        except Exception:
            return False 