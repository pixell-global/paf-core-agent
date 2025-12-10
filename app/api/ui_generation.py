from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.settings import get_settings, Settings
from app.llm_providers.openai_provider import OpenAIProvider
import os, json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


class UIGenerationRequest(BaseModel):
	query: str


@router.post("/ui-generation")
async def ui_generation(request: UIGenerationRequest):
	"""Generate UI for the given query."""
	try:
		ui_json = _generate_ui(request.query)
		render_dict = _generate_html(ui_json)

		return {
			"message": "UI generation completed successfully",
			"payload": render_dict
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to start plugin system: {str(e)}")
	

def _generate_ui(query: str):
	"""Generate UI for the given query."""

	# ── 1. JSON Schema 정의 ───────────────────────────────────────────────────────
	schema = {
		"name": "ui",
		"description": "Dynamically generated UI",
		"strict": True,
		"schema": {
			"type": "object",
			"properties": {
				"type": {
					"type": "string",
					"description": "The type of the UI component",
					"enum": [
						"div","span","a","p","h1","h2","h3","h4","h5","h6","ul","ol","li",
						"img","button","input","textarea","select","option","label","form",
						"table","thead","tbody","tr","th","td","nav","header","footer",
						"section","article","aside","main","figure","figcaption","blockquote",
						"q","hr","code","pre","iframe","video","audio","canvas","svg","path",
						"circle","rect","line","polyline","polygon","g","use","symbol"
					]
				},
				"label": {
					"type": "string",
					"description": "The label of the UI component, used for buttons or form fields"
				},
				"children": {
					"type": "array",
					"description": "Nested UI components",
					"items": { "$ref": "#" }
				},
				"attributes": {
					"type": "array",
					"description": "Arbitrary attributes for the UI component, suitable for any element using Tailwind framework",
					"items": {
						"type": "object",
						"properties": {
							"name": {
								"type": "string",
								"description": "The name of the attribute, for example onClick or className"
							},
							"value": {
								"type": "string",
								"description": "The value of the attribute using the Tailwind framework classes"
							}
						},
						"additionalProperties": False,
						"required": ["name", "value"]
					}
				}
			},
			"required": ["type", "label", "children", "attributes"],
			"additionalProperties": False
		}
	}

	# ── 2. LangChain 모델 초기화 ───────────────────────────────────────────────────
	chat = ChatOpenAI(
		model_name="gpt-4o-2024-08-06",
		temperature=0,
		openai_api_key=os.getenv("OPENAI_API_KEY"),
		model_kwargs={
			"response_format": {
				"type": "json_schema",
				"json_schema": schema
			}
		}
	)

	# ── 3. 프롬프트 작성 ──────────────────────────────────────────────────────────
	messages = [
		SystemMessage(
			content="You are a user interface designer and copy writter. "
					"Your job is to help users visualize their website ideas. "
					"You design elegant and simple webs, with professional text. "
					"You use Tailwind framework"
		),
		HumanMessage(content=query)
	]

	# ── 4. 호출 및 결과 처리 ──────────────────────────────────────────────────────
	response = chat.invoke(messages)

	ui_json = json.loads(response.content)

	return ui_json

def _generate_html(ui_json: dict):
	"""Generate HTML for the given UI JSON.
		output: {
			"html": "the page HTML",
			"title": "the page title"
		}
	"""

	chat = ChatOpenAI(
		model_name="gpt-4o-mini",
		temperature=0,
		openai_api_key=os.getenv("OPENAI_API_KEY"),
		model_kwargs={
			"response_format": {
				"type": "json_schema",
				"json_schema": {
					"name": "html_generation",
					"description": "HTML generation from UI JSON",
					"strict": True,
					"schema": {
						"type": "object",
						"properties": {
							"html": {"type": "string"},
							"title": {"type": "string"}
						},
						"required": ["html", "title"],
						"additionalProperties": False
					}
				}
			}
		}
	)

	messages = [
		SystemMessage(
			content="""
You convert a JSON to HTML. 
The JSON output has the following fields:
- html: the page HTML
- title: the page title
"""
		),
		HumanMessage(content=json.dumps(ui_json))
	]

	x = chat.invoke(messages).content

	return json.loads(x)
