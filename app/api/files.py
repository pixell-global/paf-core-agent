"""File processing API endpoints."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel

from app.schemas import FileContext
from app.utils.file_processor import FileProcessor, ProcessedFile, FileType
from app.utils.text_summarizer import TextSummarizer, SummaryResult
from app.llm_providers import LLMProviderManager
from app.settings import get_settings, Settings
from app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("files_api")

# Global instances
_file_processor: FileProcessor = None
_text_summarizer: TextSummarizer = None


def get_file_processor() -> FileProcessor:
    """Get or create file processor instance."""
    global _file_processor
    if _file_processor is None:
        settings = get_settings()
        _file_processor = FileProcessor(
            max_chunk_size=settings.max_context_tokens // 4,
            overlap_size=200
        )
    return _file_processor


def get_text_summarizer() -> TextSummarizer:
    """Get or create text summarizer instance."""
    global _text_summarizer
    if _text_summarizer is None:
        settings = get_settings()
        llm_manager = LLMProviderManager(settings)
        _text_summarizer = TextSummarizer(settings, llm_manager)
    return _text_summarizer


class FileProcessRequest(BaseModel):
    """Request for file processing."""
    files: List[FileContext]
    include_summaries: bool = True
    max_chunk_size: Optional[int] = None


class FileProcessResponse(BaseModel):
    """Response for file processing."""
    files: List[Dict[str, Any]]
    summary: Dict[str, Any]
    processing_time: float


class FileSummaryRequest(BaseModel):
    """Request for file summarization."""
    file: FileContext
    summary_type: str = "abstractive"  # "abstractive" or "extractive"


@router.post("/process", response_model=FileProcessResponse)
async def process_files(
    request: FileProcessRequest,
    file_processor: FileProcessor = Depends(get_file_processor),
    text_summarizer: TextSummarizer = Depends(get_text_summarizer)
):
    """Process multiple files with intelligent chunking and optional summarization."""
    
    if not request.files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(request.files) > 20:
        raise HTTPException(status_code=400, detail="Too many files (max 20)")
    
    import time
    start_time = time.time()
    
    try:
        processed_files = []
        total_chunks = 0
        total_size = 0
        summarized_count = 0
        
        logger.info(f"Processing {len(request.files)} files")
        
        for file_context in request.files:
            # Process file
            processed_file = file_processor.process_file(file_context)
            total_chunks += len(processed_file.chunks)
            total_size += processed_file.total_size
            
            # Generate summary if requested and file is substantial
            if request.include_summaries and processed_file.total_size > 1000:
                try:
                    summary_result = await text_summarizer.summarize_file(processed_file)
                    if summary_result:
                        processed_file.summary = summary_result.summary
                        summarized_count += 1
                except Exception as e:
                    logger.warning(f"Failed to summarize {processed_file.path}: {e}")
            
            # Convert to serializable format
            processed_files.append({
                "path": processed_file.path,
                "file_type": processed_file.file_type.value,
                "total_size": processed_file.total_size,
                "total_lines": processed_file.total_lines,
                "language": processed_file.language,
                "chunks": [
                    {
                        "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "chunk_type": chunk.chunk_type,
                        "importance": chunk.importance,
                        "summary": chunk.summary
                    } for chunk in processed_file.chunks
                ],
                "summary": processed_file.summary,
                "metadata": processed_file.metadata
            })
        
        processing_time = time.time() - start_time
        
        summary = {
            "total_files": len(request.files),
            "total_chunks": total_chunks,
            "total_size": total_size,
            "summarized_files": summarized_count,
            "file_types": list(set(pf["file_type"] for pf in processed_files)),
            "avg_chunks_per_file": total_chunks / len(request.files),
            "processing_time": processing_time
        }
        
        logger.info(
            "File processing completed",
            files=len(request.files),
            chunks=total_chunks,
            summaries=summarized_count,
            time=processing_time
        )
        
        return FileProcessResponse(
            files=processed_files,
            summary=summary,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"File processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@router.post("/summarize")
async def summarize_file(
    request: FileSummaryRequest,
    file_processor: FileProcessor = Depends(get_file_processor),
    text_summarizer: TextSummarizer = Depends(get_text_summarizer)
):
    """Summarize a single file."""
    
    try:
        # Process the file first
        processed_file = file_processor.process_file(request.file)
        
        if request.summary_type == "extractive":
            # Use extractive summarization
            full_content = '\n'.join(chunk.content for chunk in processed_file.chunks)
            summary_result = text_summarizer.create_extractive_summary(full_content)
        else:
            # Use LLM-based abstractive summarization
            summary_result = await text_summarizer.summarize_file(processed_file)
        
        if not summary_result:
            raise HTTPException(status_code=400, detail="Failed to generate summary")
        
        return {
            "file_path": processed_file.path,
            "file_type": processed_file.file_type.value,
            "original_size": processed_file.total_size,
            "original_lines": processed_file.total_lines,
            "summary": {
                "content": summary_result.summary,
                "type": summary_result.summary_type,
                "confidence": summary_result.confidence,
                "token_reduction": summary_result.token_reduction,
                "metadata": summary_result.metadata
            },
            "chunks_info": {
                "total_chunks": len(processed_file.chunks),
                "chunk_types": list(set(chunk.chunk_type for chunk in processed_file.chunks))
            }
        }
    
    except Exception as e:
        logger.error(f"File summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/analyze")
async def analyze_file_structure(
    file: FileContext,
    file_processor: FileProcessor = Depends(get_file_processor)
):
    """Analyze file structure and provide detailed information."""
    
    try:
        processed_file = file_processor.process_file(file)
        
        # Analyze chunk distribution
        chunk_types = {}
        importance_levels = []
        
        for chunk in processed_file.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            importance_levels.append(chunk.importance)
        
        # Calculate statistics
        avg_importance = sum(importance_levels) / len(importance_levels) if importance_levels else 0
        max_importance = max(importance_levels) if importance_levels else 0
        
        return {
            "file_info": {
                "path": processed_file.path,
                "type": processed_file.file_type.value,
                "size": processed_file.total_size,
                "lines": processed_file.total_lines,
                "language": processed_file.language,
                "encoding": processed_file.encoding
            },
            "chunking_analysis": {
                "total_chunks": len(processed_file.chunks),
                "chunk_types": chunk_types,
                "avg_chunk_size": processed_file.total_size / len(processed_file.chunks) if processed_file.chunks else 0,
                "importance_stats": {
                    "average": avg_importance,
                    "maximum": max_importance,
                    "high_importance_chunks": len([i for i in importance_levels if i > 1.2])
                }
            },
            "metadata": processed_file.metadata,
            "chunks": [
                {
                    "index": chunk.chunk_index,
                    "type": chunk.chunk_type,
                    "lines": f"{chunk.start_line}-{chunk.end_line}",
                    "size": len(chunk.content),
                    "tokens": chunk.token_count,
                    "importance": chunk.importance,
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                } for chunk in processed_file.chunks
            ]
        }
    
    except Exception as e:
        logger.error(f"File analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/types")
async def get_supported_file_types():
    """Get information about supported file types."""
    
    file_type_info = {
        FileType.PYTHON: {
            "name": "Python",
            "extensions": [".py"],
            "description": "Python source code with intelligent function/class chunking",
            "features": ["syntax-aware chunking", "function detection", "class detection", "import analysis"]
        },
        FileType.JAVASCRIPT: {
            "name": "JavaScript/TypeScript", 
            "extensions": [".js", ".ts", ".jsx", ".tsx"],
            "description": "JavaScript and TypeScript code with function-based chunking",
            "features": ["function detection", "arrow function support", "module analysis"]
        },
        FileType.MARKDOWN: {
            "name": "Markdown",
            "extensions": [".md", ".markdown"],
            "description": "Markdown documents with header-based section chunking",
            "features": ["header-based sections", "importance scoring", "link extraction"]
        },
        FileType.JSON: {
            "name": "JSON",
            "extensions": [".json"],
            "description": "JSON data with object-level chunking",
            "features": ["object-level chunking", "structure analysis", "key extraction"]
        },
        FileType.CSV: {
            "name": "CSV",
            "extensions": [".csv"],
            "description": "CSV data with row-based chunking and header preservation",
            "features": ["header preservation", "row-based chunking", "column analysis"]
        },
        FileType.TEXT: {
            "name": "Plain Text",
            "extensions": [".txt"],
            "description": "Plain text with paragraph-aware chunking",
            "features": ["paragraph detection", "sentence boundaries", "topic extraction"]
        },
        FileType.HTML: {
            "name": "HTML",
            "extensions": [".html", ".htm"],
            "description": "HTML markup with element-based processing",
            "features": ["tag detection", "content extraction", "structure analysis"]
        },
        FileType.XML: {
            "name": "XML",
            "extensions": [".xml"],
            "description": "XML markup with element-based processing",
            "features": ["tag detection", "hierarchical structure", "namespace support"]
        }
    }
    
    return {
        "supported_types": [
            {
                "type": file_type.value,
                "info": info
            } for file_type, info in file_type_info.items()
        ],
        "total_types": len(file_type_info),
        "features": {
            "intelligent_chunking": True,
            "type_detection": True,
            "summarization": True,
            "importance_scoring": True,
            "metadata_extraction": True
        }
    } 