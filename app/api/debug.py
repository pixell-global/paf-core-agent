"""Debug API endpoints for troubleshooting file uploads and request processing."""

import json
import time
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, FileContent, FileContext
from app.settings import get_settings, Settings
from app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("debug_api")


@router.post("/inspect-request")
async def inspect_chat_request(
    chat_request: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """
    Debug endpoint to inspect exactly what the core agent receives.
    This helps debug file upload and request formatting issues.
    """
    
    inspection_result = {
        "timestamp": time.time(),
        "request_inspection": {
            "message": {
                "content": chat_request.message,
                "length": len(chat_request.message),
                "preview": chat_request.message[:100] + "..." if len(chat_request.message) > 100 else chat_request.message
            },
            "files": {
                "count": len(chat_request.files) if chat_request.files else 0,
                "files_provided": bool(chat_request.files),
                "file_details": []
            },
            "history": {
                "has_history": bool(chat_request.history),
                "message_count": len(chat_request.history) if chat_request.history else 0,
                "memory_limit": chat_request.memory_limit
            },
            "options": {
                "model": chat_request.model,
                "temperature": chat_request.temperature,
                "max_tokens": chat_request.max_tokens,
                "show_thinking": chat_request.show_thinking,
                "context_window_size": chat_request.context_window_size
            }
        },
        "raw_request_info": {
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "user_agent": request.headers.get("user-agent"),
            "origin": request.headers.get("origin")
        }
    }
    
    # Detailed file inspection
    if chat_request.files:
        for i, file_item in enumerate(chat_request.files):
            file_detail = {
                "index": i,
                "schema_type": "unknown",
                "validation": {
                    "is_valid": True,
                    "errors": []
                }
            }
            
            # Determine schema type and extract details
            try:
                if hasattr(file_item, 'file_name'):
                    # New FileContent schema
                    file_detail.update({
                        "schema_type": "FileContent",
                        "file_name": file_item.file_name,
                        "file_type": file_item.file_type,
                        "file_size": file_item.file_size,
                        "has_content": bool(file_item.content),
                        "content_length": len(file_item.content) if file_item.content else 0,
                        "content_type": type(file_item.content).__name__ if file_item.content else "None",
                        "has_signed_url": bool(file_item.signed_url),
                        "signed_url": file_item.signed_url[:50] + "..." if file_item.signed_url and len(file_item.signed_url) > 50 else file_item.signed_url,
                        "content_preview": file_item.content[:100] + "..." if file_item.content and len(file_item.content) > 100 else file_item.content
                    })
                    
                    # Validation for FileContent
                    if not file_item.file_name:
                        file_detail["validation"]["is_valid"] = False
                        file_detail["validation"]["errors"].append("Missing file_name")
                    
                    if not file_item.content and not file_item.signed_url:
                        file_detail["validation"]["is_valid"] = False
                        file_detail["validation"]["errors"].append("Missing both content and signed_url")
                    
                    if file_item.file_size <= 0:
                        file_detail["validation"]["is_valid"] = False
                        file_detail["validation"]["errors"].append("Invalid file_size")
                
                elif hasattr(file_item, 'path'):
                    # Legacy FileContext schema
                    file_detail.update({
                        "schema_type": "FileContext",
                        "path": file_item.path,
                        "has_content": bool(file_item.content),
                        "content_length": len(file_item.content) if file_item.content else 0,
                        "content_type": type(file_item.content).__name__ if file_item.content else "None",
                        "summary": getattr(file_item, 'summary', None),
                        "content_preview": file_item.content[:100] + "..." if file_item.content and len(file_item.content) > 100 else file_item.content
                    })
                    
                    # Validation for FileContext
                    if not file_item.path:
                        file_detail["validation"]["is_valid"] = False
                        file_detail["validation"]["errors"].append("Missing path")
                    
                    if not file_item.content:
                        file_detail["validation"]["is_valid"] = False
                        file_detail["validation"]["errors"].append("Missing content")
                
                else:
                    file_detail["validation"]["is_valid"] = False
                    file_detail["validation"]["errors"].append("Unknown file schema - missing both file_name and path")
                
            except Exception as e:
                file_detail["validation"]["is_valid"] = False
                file_detail["validation"]["errors"].append(f"Error inspecting file: {str(e)}")
            
            inspection_result["request_inspection"]["files"]["file_details"].append(file_detail)
    
    # File processing capability check
    file_processing_status = await _check_file_processing_capabilities()
    inspection_result["file_processing_capabilities"] = file_processing_status
    
    # Log the inspection for server-side debugging
    logger.info(
        "Request inspection performed",
        files_count=inspection_result["request_inspection"]["files"]["count"],
        message_length=inspection_result["request_inspection"]["message"]["length"],
        valid_files=sum(1 for f in inspection_result["request_inspection"]["files"]["file_details"] if f["validation"]["is_valid"]),
        schema_types=[f["schema_type"] for f in inspection_result["request_inspection"]["files"]["file_details"]]
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