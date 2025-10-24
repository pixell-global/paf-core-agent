import json
import asyncio
import pytest
from typing import Any, Dict, Optional

from src.schemas import ChatRequest, EventType
from src.core.upee_engine import UPEEEngine
from src.settings import Settings
from src.agents.models import A2AResponse
from src.agents.models import AgentDecision


class FakeAgentManager:
	def __init__(self, *, response: A2AResponse):
		self._response = response
		self._should_use_count = 0
		self.last_context: Optional[Dict[str, Any]] = None

	async def should_use_agent(self, phase, context: Dict[str, Any]) -> Optional[AgentDecision]:
		self._should_use_count += 1
		# Return a simple decision to force using the external agent path
		return AgentDecision(
			phase=phase.value,
			agent_id="agent-1",
			capability="ui-demo",
			reasoning="test",
			confidence=0.9,
			alternatives=[],
			estimated_value=0.9,
		)

	async def execute_agent_request(self, decision: AgentDecision, payload: Dict[str, Any], context: Dict[str, Any]) -> A2AResponse:
		self.last_context = context
		return self._response


@pytest.mark.asyncio
async def test_understand_phase_forwards_ui_render_and_passes_capabilities():
	"""Agent metadata["ui"] is forwarded as a ui.render envelope and ui.capabilities is passed into context."""
	settings = Settings()
	upee = UPEEEngine(settings)

	# Fake agent response with metadata["ui"] -> ui.render
	ui_spec = {
		"manifest": {"id": "demo.app", "name": "Demo", "version": "1.0.0", "capabilities": ["page"]},
		"data": {"items": []},
		"view": {"type": "page", "title": "Demo", "children": []},
	}
	fake_response = A2AResponse(
		request_id="req-1",
		agent_id="agent-1",
		status="success",
		result={},
		metadata={"ui": ui_spec},
	)
	upee.agent_manager = FakeAgentManager(response=fake_response)  # type: ignore

	# Create a request that includes ui.capabilities
	request = ChatRequest(
		message="find me skincare subreddits",
		show_thinking=False,
		metadata={"ui.capabilities": {"components": ["page", "table"], "streaming": True, "specVersion": "1.0.0"}},
	)

	# Collect events from the understand phase
	seen_ui_render = False
	async for event in upee._run_understand_phase(request):  # type: ignore
		if event.get("event") == EventType.CONTENT:
			data = event.get("data")
			if isinstance(data, str):
				parsed = json.loads(data)
			else:
				parsed = data
			if isinstance(parsed, dict) and "ui" in parsed:
				ui_env = parsed["ui"]
				assert ui_env.get("type") == "ui.render"
				assert ui_env.get("manifest", {}).get("id") == "demo.app"
				seen_ui_render = True
				break

	assert seen_ui_render, "Expected a CONTENT event containing a ui.render envelope"

	# Verify capabilities were passed to agent context
	assert "ui.capabilities" in upee.agent_manager.last_context  # type: ignore
	assert upee.agent_manager.last_context["ui.capabilities"]["components"] == ["page", "table"]  # type: ignore


@pytest.mark.asyncio
async def test_plan_phase_forwards_ui_patch():
	"""Agent result with type ui.patch is forwarded as a CONTENT event with ui payload."""
	settings = Settings()
	upee = UPEEEngine(settings)

	fake_response = A2AResponse(
		request_id="req-2",
		agent_id="agent-1",
		status="success",
		result={
			"type": "ui.patch",
			"patch": [{"op": "replace", "path": "/data/ui/selected", "value": [1, 2]}],
		},
		metadata={},
	)
	upee.agent_manager = FakeAgentManager(response=fake_response)  # type: ignore

	request = ChatRequest(message="find me skincare subreddits", show_thinking=False)

	seen_ui_patch = False
	async for event in upee._run_plan_phase(request):  # type: ignore
		if event.get("event") == EventType.CONTENT:
			data = event.get("data")
			parsed = json.loads(data) if isinstance(data, str) else data
			if isinstance(parsed, dict) and "ui" in parsed:
				ui_env = parsed["ui"]
				if ui_env.get("type") == "ui.patch":
					seen_ui_patch = True
					break

	assert seen_ui_patch, "Expected a CONTENT event containing a ui.patch envelope"


@pytest.mark.asyncio
async def test_plan_phase_forwards_action_result():
	"""Agent result with type action.result is forwarded in CONTENT event under action_result key."""
	settings = Settings()
	upee = UPEEEngine(settings)

	fake_response = A2AResponse(
		request_id="req-3",
		agent_id="agent-1",
		status="success",
		result={
			"type": "action.result",
			"action": "approve",
			"status": "ok",
			"message": "Approved",
			"details": {"count": 2},
		},
		metadata={},
	)
	upee.agent_manager = FakeAgentManager(response=fake_response)  # type: ignore

	request = ChatRequest(message="find me skincare subreddits", show_thinking=False)

	seen_action_result = False
	async for event in upee._run_plan_phase(request):  # type: ignore
		if event.get("event") == EventType.CONTENT:
			data = event.get("data")
			parsed = json.loads(data) if isinstance(data, str) else data
			if isinstance(parsed, dict) and "action_result" in parsed:
				ar = parsed["action_result"]
				if ar.get("type") == "action.result" and ar.get("action") == "approve":
					seen_action_result = True
					break

	assert seen_action_result, "Expected a CONTENT event containing an action.result envelope"


def test_bridge_ui_event_endpoint_forwards(monkeypatch):
	"""POST /api/bridge/ui/event forwards ui.event envelope to the bridge."""
	from fastapi.testclient import TestClient
	from src.main import app
	from src.api import bridge as bridge_module
	from src.core.bridge.protocol import MessageType

	class FakeBridge:
		def __init__(self):
			self.calls = []
		async def send_message(self, message_type, target_agent_id, payload, priority, conversation_id, ttl_seconds=None, requires_ack=False):
			self.calls.append({
				"message_type": message_type,
				"target_agent_id": target_agent_id,
				"payload": payload,
				"priority": priority,
				"conversation_id": conversation_id,
			})
			class R:
				id = "mid-1"
			return R()

	fake_bridge = FakeBridge()

	def fake_get_bridge():
		return fake_bridge

	monkeypatch.setattr(bridge_module, "get_bridge", fake_get_bridge)

	client = TestClient(app)
	payload = {
		"event": {"type": "ui.event", "intent": "approve", "params": {"x": 1}},
		"target_agent_id": "agent-1",
		"conversation_id": "conv-1",
	}
	resp = client.post("/api/bridge/ui/event", json=payload)
	assert resp.status_code == 200
	data = resp.json()
	assert data["status"] == "sent"
	# Ensure the bridge was called with the correct message type and payload
	assert fake_bridge.calls, "Bridge.send_message was not called"
	call = fake_bridge.calls[0]
	assert call["message_type"] == MessageType.COORDINATION
	assert call["payload"] == payload["event"] 