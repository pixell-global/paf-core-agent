"""
File processing utilities for handling file content during UPEE phases.
"""

import asyncio
import hashlib
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import httpx
import aiofiles

from app.schemas import FileContent, FileContext, ConversationMessage

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles file processing for chat requests and UPEE phases."""
    
    # File size limits (in bytes)
    INLINE_CONTENT_LIMIT = 100 * 1024  # 100KB for inline content
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB maximum file size
    
    # Supported file types for content processing
    SUPPORTED_TEXT_TYPES = {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.scss', '.sass',
        '.md', '.txt', '.json', '.yaml', '.yml', '.xml', '.csv', '.sql',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.go', '.rs', '.cpp', '.c', '.h', '.hpp', '.java', '.kt', '.swift',
        '.rb', '.php', '.scala', '.clj', '.hs', '.ml', '.f90', '.r',
        '.dockerfile', '.gitignore', '.env', '.ini', '.cfg', '.conf', '.toml'
    }
    
    SUPPORTED_MIME_TYPES = {
        'text/plain', 'text/html', 'text/css', 'text/javascript', 'text/csv',
        'text/markdown', 'text/xml', 'application/json', 'application/xml',
        'application/javascript', 'application/typescript', 'application/yaml',
        'application/x-yaml', 'application/toml', 'application/sql'
    }
    
    def __init__(self):
        self.processed_files_cache: Dict[str, Dict[str, Any]] = {}
    
    async def process_files(
        self, 
        files: Optional[List[Union[FileContent, FileContext]]]
    ) -> List[Dict[str, Any]]:
        """Process files for UPEE phases."""
        if not files:
            return []
        
        processed_files = []
        
        for file_item in files:
            try:
                processed_file = await self._process_single_file(file_item)
                if processed_file:
                    processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error processing file {getattr(file_item, 'file_name', 'unknown')}: {e}")
                # Add error information but continue processing other files
                processed_files.append({
                    "file_name": getattr(file_item, 'file_name', 'unknown'),
                    "error": str(e),
                    "processed": False
                })
        
        return processed_files
    
    async def _process_single_file(
        self, 
        file_item: Union[FileContent, FileContext]
    ) -> Optional[Dict[str, Any]]:
        """Process a single file item."""
        if isinstance(file_item, FileContext):
            # Legacy format - convert to new format
            return await self._process_legacy_file(file_item)
        elif isinstance(file_item, FileContent):
            return await self._process_file_content(file_item)
        else:
            logger.warning(f"Unknown file type: {type(file_item)}")
            return None
    
    async def _process_legacy_file(self, file_context: FileContext) -> Dict[str, Any]:
        """Process legacy FileContext format."""
        file_info = {
            "file_name": os.path.basename(file_context.path),
            "file_path": file_context.path,
            "content": file_context.content,
            "file_type": self._detect_file_type(file_context.path),
            "file_size": len(file_context.content.encode('utf-8')),
            "line_start": file_context.line_start,
            "line_end": file_context.line_end,
            "processed": True,
            "source": "legacy_format"
        }
        
        # Add content analysis
        file_info.update(await self._analyze_content(file_context.content, file_info["file_type"]))
        
        return file_info
    
    async def _process_file_content(self, file_content: FileContent) -> Dict[str, Any]:
        """Process FileContent format."""
        content = None
        
        # Get content either inline or from signed URL
        if file_content.content:
            content = file_content.content
        elif file_content.signed_url:
            content = await self._fetch_from_signed_url(file_content.signed_url)
        else:
            logger.warning(f"No content or signed URL provided for file {file_content.file_name}")
            return None
        
        # Check if this file requires agentic processing
        if self._requires_agentic_processing(file_content.file_type, file_content.file_name):
            logger.info(f"File {file_content.file_name} requires agentic processing")
            
            file_info = {
                "file_name": file_content.file_name,
                "file_path": file_content.file_path,
                "file_type": file_content.file_type,
                "file_size": file_content.file_size,
                "line_start": file_content.line_start,
                "line_end": file_content.line_end,
                "metadata": file_content.metadata or {},
                "processed": False,  # Will be processed by agent
                "requires_agentic_processing": True,
                "source": "signed_url" if file_content.signed_url else "inline_content",
                "processing_status": "pending_agentic_workflow",
                "content": content[:200] + "..." if content and len(content) > 200 else content  # Preview only
            }
            
            # Add basic file analysis
            file_info.update({
                "char_count": len(content) if content else 0,
                "is_text": self._is_text_content(file_content.file_type),
                "content_hash": hashlib.md5(content.encode('utf-8')).hexdigest() if content else None
            })
            
            return file_info
        
        # Standard text processing for simple files
        file_info = {
            "file_name": file_content.file_name,
            "file_path": file_content.file_path,
            "content": content,
            "file_type": file_content.file_type,
            "file_size": file_content.file_size,
            "line_start": file_content.line_start,
            "line_end": file_content.line_end,
            "metadata": file_content.metadata or {},
            "processed": True,
            "requires_agentic_processing": False,
            "source": "signed_url" if file_content.signed_url else "inline_content"
        }
        
        # Add content analysis
        if content:
            file_info.update(await self._analyze_content(content, file_content.file_type))
        
        return file_info
    
    async def _fetch_from_signed_url(self, signed_url: str) -> Optional[str]:
        """Fetch file content from a signed URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(signed_url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not self._is_text_content(content_type):
                    logger.warning(f"Non-text content type from signed URL: {content_type}")
                    return None
                
                # Check content size
                content = response.text
                if len(content.encode('utf-8')) > self.MAX_FILE_SIZE:
                    logger.warning(f"File too large from signed URL: {len(content)} bytes")
                    return None
                
                return content
                
        except Exception as e:
            logger.error(f"Error fetching from signed URL {signed_url}: {e}")
            return None
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from path."""
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        
        # Fall back to extension
        _, ext = os.path.splitext(file_path.lower())
        return ext or 'text/plain'
    
    def _is_text_content(self, content_type_or_path: str) -> bool:
        """Check if content type or file path indicates text content."""
        content_type_or_path = content_type_or_path.lower()
        
        # Check MIME type
        if any(mime_type in content_type_or_path for mime_type in self.SUPPORTED_MIME_TYPES):
            return True
        
        # Check file extension
        _, ext = os.path.splitext(content_type_or_path)
        return ext in self.SUPPORTED_TEXT_TYPES
    
    def _requires_agentic_processing(self, file_type: str, file_name: str) -> bool:
        """Determine if a file requires agentic processing workflow."""
        _, ext = os.path.splitext(file_name.lower())
        
        # Files that need special processing
        complex_types = {
            # Office documents
            '.xlsx', '.xls', '.xlsm', '.xlsb',
            '.docx', '.doc', '.pptx', '.ppt',
            # Data files
            '.parquet', '.h5', '.hdf5', '.sqlite', '.db',
            # Media files
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
            '.mp3', '.wav', '.mp4', '.avi', '.mov',
            # Archive files
            '.zip', '.tar', '.gz', '.rar', '.7z',
            # Binary formats
            '.exe', '.dll', '.so', '.dylib',
        }
        
        complex_mime_types = {
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/pdf',
            'image/jpeg', 'image/png', 'image/gif',
            'application/zip',
            'application/octet-stream'
        }
        
        return ext in complex_types or file_type in complex_mime_types
    
    async def _analyze_content(self, content: str, file_type: str) -> Dict[str, Any]:
        """Analyze file content for metadata."""
        analysis = {
            "line_count": len(content.splitlines()),
            "char_count": len(content),
            "word_count": len(content.split()),
            "is_text": self._is_text_content(file_type),
            "content_hash": hashlib.md5(content.encode('utf-8')).hexdigest()
        }
        
        # Language-specific analysis
        if file_type in ['text/x-python', 'text/x-script.python'] or file_type.endswith('.py'):
            analysis.update(await self._analyze_python_content(content))
        elif file_type in ['text/javascript', 'application/javascript'] or file_type.endswith(('.js', '.ts')):
            analysis.update(await self._analyze_javascript_content(content))
        elif file_type == 'text/markdown' or file_type.endswith('.md'):
            analysis.update(await self._analyze_markdown_content(content))
        
        return analysis
    
    async def _analyze_python_content(self, content: str) -> Dict[str, Any]:
        """Analyze Python file content."""
        analysis = {"language": "python"}
        
        lines = content.splitlines()
        analysis["import_count"] = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
        analysis["class_count"] = sum(1 for line in lines if line.strip().startswith('class '))
        analysis["function_count"] = sum(1 for line in lines if line.strip().startswith('def '))
        analysis["comment_count"] = sum(1 for line in lines if line.strip().startswith('#'))
        
        return analysis
    
    async def _analyze_javascript_content(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file content."""
        analysis = {"language": "javascript"}
        
        lines = content.splitlines()
        analysis["import_count"] = sum(1 for line in lines if 'import ' in line or 'require(' in line)
        analysis["function_count"] = sum(1 for line in lines if 'function ' in line or '=>' in line)
        analysis["class_count"] = sum(1 for line in lines if line.strip().startswith('class '))
        analysis["comment_count"] = sum(1 for line in lines if line.strip().startswith('//') or '/*' in line)
        
        return analysis
    
    async def _analyze_markdown_content(self, content: str) -> Dict[str, Any]:
        """Analyze Markdown file content."""
        analysis = {"language": "markdown"}
        
        lines = content.splitlines()
        analysis["heading_count"] = sum(1 for line in lines if line.strip().startswith('#'))
        analysis["link_count"] = content.count('](')
        analysis["code_block_count"] = content.count('```')
        analysis["list_item_count"] = sum(1 for line in lines if line.strip().startswith(('-', '*', '+')))
        
        return analysis
    
    def process_conversation_history(
        self, 
        history: Optional[List[ConversationMessage]], 
        memory_limit: int = 10
    ) -> Dict[str, Any]:
        """Process conversation history for short-term memory."""
        if not history:
            return {
                "message_count": 0,
                "total_tokens_estimate": 0,
                "processed_messages": [],
                "files_in_history": []
            }
        
        # Limit history to memory_limit messages
        limited_history = history[-memory_limit:] if len(history) > memory_limit else history
        
        processed_messages = []
        total_tokens_estimate = 0
        files_in_history = []
        
        for msg in limited_history:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            tokens_estimate = len(msg.content) // 4
            total_tokens_estimate += tokens_estimate
            
            processed_msg = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "tokens_estimate": tokens_estimate,
                "metadata": msg.metadata or {}
            }
            
            # Process files in message
            if msg.files:
                msg_files = []
                for file_item in msg.files:
                    file_summary = {
                        "file_name": file_item.file_name,
                        "file_type": file_item.file_type,
                        "file_size": file_item.file_size,
                        "has_content": bool(file_item.content),
                        "has_signed_url": bool(file_item.signed_url)
                    }
                    msg_files.append(file_summary)
                    files_in_history.append(file_summary)
                
                processed_msg["files"] = msg_files
            
            processed_messages.append(processed_msg)
        
        return {
            "message_count": len(processed_messages),
            "total_tokens_estimate": total_tokens_estimate,
            "processed_messages": processed_messages,
            "files_in_history": files_in_history,
            "history_truncated": len(history) > memory_limit,
            "original_message_count": len(history)
        }
    
    def create_context_summary(
        self, 
        processed_files: List[Dict[str, Any]], 
        history_info: Dict[str, Any]
    ) -> str:
        """Create a summary of file and conversation context for UPEE phases."""
        context_parts = []
        
        # File context summary
        if processed_files:
            file_count = len(processed_files)
            successful_files = [f for f in processed_files if f.get('processed', False)]
            
            context_parts.append(f"FILES CONTEXT ({file_count} files provided):")
            
            for file_info in successful_files:
                file_summary = f"- {file_info['file_name']} ({file_info['file_type']}"
                if file_info.get('line_count'):
                    file_summary += f", {file_info['line_count']} lines"
                file_summary += ")"
                if file_info.get('language'):
                    file_summary += f" - {file_info['language']} file"
                context_parts.append(file_summary)
        
        # Conversation history summary
        if history_info['message_count'] > 0:
            context_parts.append(f"\nCONVERSATION HISTORY ({history_info['message_count']} messages):")
            context_parts.append(f"- Total estimated tokens: {history_info['total_tokens_estimate']}")
            
            if history_info['files_in_history']:
                context_parts.append(f"- Files referenced in history: {len(history_info['files_in_history'])}")
            
            if history_info['history_truncated']:
                context_parts.append(f"- History truncated from {history_info['original_message_count']} messages")
        
        return "\n".join(context_parts) if context_parts else "No additional context provided."


# Global file processor instance
file_processor = FileProcessor()