"""Debug API endpoints for troubleshooting file uploads and request processing."""

import json
import time
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, FileContent, FileContext
from app.settings import get_settings, Settings
from app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("debug_api")


# -----------------------------------------------------------------------------------
# New multipart-friendly implementation (≤10 MB per file)
# -----------------------------------------------------------------------------------


@router.post("/inspect-request")
async def inspect_chat_request(
    payload: str = Form(..., description="JSON string with ChatRequest-style fields except files"),
    files: List[UploadFile] = File(None, description="One or more files (each < 10 MB)"),
    request: Request = None,
    settings: Settings = Depends(get_settings)
):
    """
    Debug endpoint that now accepts **multipart/form-data**:

    • `payload` – JSON string containing the usual ChatRequest fields (message, model, etc.).
    • `files`   – <10 MB each, sent as separate form parts.
    """

    # ------------------------------------------------------------------
    # Parse payload JSON → dict → ChatRequest (without files for now)
    # ------------------------------------------------------------------
    try:
        payload_dict: Dict[str, Any] = json.loads(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in payload field: {str(e)}")

    # Basic required field check
    if "message" not in payload_dict or not str(payload_dict["message"]).strip():
        raise HTTPException(status_code=400, detail="`message` is required inside payload")

    # ------------------------------------------------------------------
    # Handle uploaded files – enforce 10 MB limit
    # ------------------------------------------------------------------
    max_bytes = 10 * 1024 * 1024  # 10MB
    file_details = []

    if files:
        for idx, up_file in enumerate(files):
            file_bytes = await up_file.read()
            size = len(file_bytes)

            if size > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{up_file.filename}' exceeds 10MB limit ({size} bytes)"
                )

            # Keep small preview (first 100 bytes decoded if possible)
            try:
                preview = file_bytes[:100].decode("utf-8", errors="replace")
            except Exception:
                preview = "<binary>"

            file_details.append({
                "index": idx,
                "file_name": up_file.filename,
                "content_type": up_file.content_type,
                "file_size": size,
                "preview": preview
            })

    # ------------------------------------------------------------------
    # Compose inspection result (similar shape as before)
    # ------------------------------------------------------------------
    inspection_result = {
        "timestamp": time.time(),
        "request_inspection": {
            "message": {
                "content": payload_dict["message"],
                "length": len(payload_dict["message"]),
                "preview": payload_dict["message"][:100] + "…" if len(payload_dict["message"]) > 100 else payload_dict["message"]
            },
            "files": {
                "count": len(file_details),
                "file_details": file_details
            },
            "options": {
                "model": payload_dict.get("model"),
                "temperature": payload_dict.get("temperature"),
                "max_tokens": payload_dict.get("max_tokens")
            }
        },
        "raw_request_info": {
            "content_type": request.headers.get("content-type") if request else None,
            "content_length": request.headers.get("content-length") if request else None,
            "user_agent": request.headers.get("user-agent") if request else None,
        }
    }

    logger.info(
        "Multipart inspection performed",
        files=len(file_details),
        message_length=len(payload_dict["message"])
    )

    return JSONResponse(content=inspection_result)


@router.post("/test-file-processing")
async def test_file_processing(
    chat_request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Test file processing without running full UPEE workflow.
    This helps isolate file processing issues.
    """
    
    if not chat_request.files:
        raise HTTPException(status_code=400, detail="No files provided for testing")
    
    from app.core.file_processor import file_processor
    from app.core.understand import UnderstandPhase
    
    test_results = {
        "timestamp": time.time(),
        "files_received": len(chat_request.files),
        "processing_results": []
    }
    
    try:
        # Test basic file processor
        basic_results = await file_processor.process_files(chat_request.files)
        
        for i, (file_item, basic_result) in enumerate(zip(chat_request.files, basic_results)):
            processing_result = {
                "file_index": i,
                "file_name": getattr(file_item, 'file_name', getattr(file_item, 'path', f'file_{i}')),
                "basic_processing": {
                    "success": basic_result.get("processed", False),
                    "content_length": len(basic_result.get("content", "")),
                    "requires_agentic": basic_result.get("requires_agentic_processing", False),
                    "processing_status": basic_result.get("processing_status", "unknown"),
                    "content_preview": basic_result.get("content", "")[:200] + "..." if basic_result.get("content") and len(basic_result.get("content", "")) > 200 else basic_result.get("content", "")
                }
            }
            
            # Test agentic processing if needed
            if basic_result.get("requires_agentic_processing", False):
                try:
                    from app.core.file_processing_agent import get_file_processing_agent
                    from app.llm_providers import LLMProviderManager
                    
                    llm_manager = LLMProviderManager(settings)
                    agent = get_file_processing_agent(settings, llm_manager)
                    
                    import uuid
                    test_request_id = f"test-{uuid.uuid4().hex[:8]}"
                    
                    agent_result = await agent.process_file(file_item, test_request_id)
                    
                    processing_result["agentic_processing"] = {
                        "success": agent_result.success,
                        "confidence_score": agent_result.confidence_score,
                        "execution_time": agent_result.execution_time,
                        "content_length": len(agent_result.extracted_content) if agent_result.extracted_content else 0,
                        "tools_used": list(agent_result.tool_outputs.keys()),
                        "errors": agent_result.errors,
                        "content_preview": agent_result.extracted_content[:300] + "..." if agent_result.extracted_content and len(agent_result.extracted_content) > 300 else agent_result.extracted_content
                    }
                    
                except Exception as e:
                    processing_result["agentic_processing"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            test_results["processing_results"].append(processing_result)
    
    except Exception as e:
        test_results["error"] = str(e)
        logger.error(f"File processing test failed: {e}", exc_info=True)
    
    return JSONResponse(content=test_results)


async def _check_file_processing_capabilities():
    """Check if file processing dependencies are available."""
    
    capabilities = {
        "pandas_available": False,
        "openpyxl_available": False,
        "file_processor_available": True,
        "agentic_processor_available": True
    }
    
    try:
        import pandas as pd
        capabilities["pandas_available"] = True
        capabilities["pandas_version"] = pd.__version__
    except ImportError:
        pass
    
    try:
        import openpyxl
        capabilities["openpyxl_available"] = True
        capabilities["openpyxl_version"] = openpyxl.__version__
    except ImportError:
        pass
    
    try:
        from app.core.file_processor import file_processor
        capabilities["file_processor_available"] = True
    except ImportError:
        capabilities["file_processor_available"] = False
    
    try:
        from app.core.file_processing_agent import get_file_processing_agent
        capabilities["agentic_processor_available"] = True
    except ImportError:
        capabilities["agentic_processor_available"] = False
    
    return capabilities