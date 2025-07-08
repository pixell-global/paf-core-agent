"""File processing utilities for PAF Core Agent."""

import re
import json
import csv
import io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import mimetypes
from pathlib import Path

from app.schemas import FileContext
from app.utils.logging_config import get_logger

logger = get_logger("file_processor")


class FileType(str, Enum):
    """Supported file types."""
    TEXT = "text"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    YAML = "yaml"
    CONFIG = "config"
    LOG = "log"
    UNKNOWN = "unknown"


@dataclass
class ProcessedFile:
    """Processed file with metadata and chunks."""
    path: str
    file_type: FileType
    total_size: int
    total_lines: int
    language: Optional[str]
    encoding: str
    chunks: List['FileChunk']
    summary: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FileChunk:
    """A chunk of file content."""
    content: str
    start_line: int
    end_line: int
    chunk_index: int
    token_count: int
    chunk_type: str = "content"  # content, header, function, class, etc.
    importance: float = 1.0  # 0.0 to 1.0, higher is more important
    summary: Optional[str] = None


class FileProcessor:
    """Intelligent file processor with format detection and chunking."""
    
    def __init__(self, max_chunk_size: int = 2000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.logger = get_logger("file_processor")
        
        # File type detection patterns
        self.extension_map = {
            '.py': FileType.PYTHON,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.JAVASCRIPT,
            '.jsx': FileType.JAVASCRIPT,
            '.tsx': FileType.JAVASCRIPT,
            '.json': FileType.JSON,
            '.csv': FileType.CSV,
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.html': FileType.HTML,
            '.htm': FileType.HTML,
            '.xml': FileType.XML,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.txt': FileType.TEXT,
            '.log': FileType.LOG,
            '.conf': FileType.CONFIG,
            '.cfg': FileType.CONFIG,
            '.ini': FileType.CONFIG,
        }
    
    def detect_file_type(self, path: str, content: str) -> FileType:
        """Detect file type from path and content."""
        # Check extension first
        ext = Path(path).suffix.lower()
        if ext in self.extension_map:
            return self.extension_map[ext]
        
        # Content-based detection
        content_lower = content.lower().strip()
        
        # Check for common patterns
        if content_lower.startswith('<!doctype html') or '<html' in content_lower:
            return FileType.HTML
        elif content_lower.startswith('<?xml') or content.strip().startswith('<'):
            return FileType.XML
        elif self._is_json_content(content):
            return FileType.JSON
        elif self._is_csv_content(content):
            return FileType.CSV
        elif self._is_python_content(content):
            return FileType.PYTHON
        elif self._is_javascript_content(content):
            return FileType.JAVASCRIPT
        elif self._is_markdown_content(content):
            return FileType.MARKDOWN
        
        return FileType.TEXT
    
    def _is_json_content(self, content: str) -> bool:
        """Check if content appears to be JSON."""
        try:
            json.loads(content.strip())
            return True
        except:
            return False
    
    def _is_csv_content(self, content: str) -> bool:
        """Check if content appears to be CSV."""
        try:
            lines = content.strip().split('\n')[:5]  # Check first 5 lines
            if len(lines) < 2:
                return False
            
            # Try to parse as CSV
            sample = '\n'.join(lines)
            reader = csv.Sniffer()
            dialect = reader.sniff(sample)
            return True
        except:
            return False
    
    def _is_python_content(self, content: str) -> bool:
        """Check if content appears to be Python."""
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'if __name__']
        return any(keyword in content for keyword in python_keywords)
    
    def _is_javascript_content(self, content: str) -> bool:
        """Check if content appears to be JavaScript/TypeScript."""
        js_keywords = ['function ', 'const ', 'let ', 'var ', '=>', 'export ', 'import ']
        return any(keyword in content for keyword in js_keywords)
    
    def _is_markdown_content(self, content: str) -> bool:
        """Check if content appears to be Markdown."""
        md_patterns = [r'^#+ ', r'^\* ', r'^\- ', r'^\d+\. ', r'\[.*\]\(.*\)']
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            if any(re.search(pattern, line, re.MULTILINE) for pattern in md_patterns):
                return True
        return False
    
    def process_file(self, file_context: FileContext) -> ProcessedFile:
        """Process a file and return structured information."""
        self.logger.info(
            "Processing file",
            path=file_context.path,
            content_length=len(file_context.content)
        )
        
        # Detect file type
        file_type = self.detect_file_type(file_context.path, file_context.content)
        
        # Basic file metrics
        lines = file_context.content.split('\n')
        total_lines = len(lines)
        total_size = len(file_context.content)
        
        # Create chunks based on file type
        chunks = self._create_chunks(file_context.content, file_type, file_context.path)
        
        # Generate summary if content is large
        summary = None
        if total_size > self.max_chunk_size * 2:
            summary = self._generate_summary(file_context.content, file_type)
        
        # Extract metadata
        metadata = self._extract_metadata(file_context.content, file_type)
        
        processed_file = ProcessedFile(
            path=file_context.path,
            file_type=file_type,
            total_size=total_size,
            total_lines=total_lines,
            language=self._detect_language(file_type),
            encoding="utf-8",  # Assume UTF-8 for now
            chunks=chunks,
            summary=summary,
            metadata=metadata
        )
        
        self.logger.info(
            "File processing completed",
            path=file_context.path,
            file_type=file_type.value,
            chunks_created=len(chunks),
            has_summary=summary is not None
        )
        
        return processed_file
    
    def _create_chunks(self, content: str, file_type: FileType, path: str) -> List[FileChunk]:
        """Create intelligent chunks based on file type."""
        if file_type == FileType.PYTHON:
            return self._chunk_python_code(content)
        elif file_type == FileType.JAVASCRIPT:
            return self._chunk_javascript_code(content)
        elif file_type == FileType.MARKDOWN:
            return self._chunk_markdown(content)
        elif file_type == FileType.JSON:
            return self._chunk_json(content)
        elif file_type == FileType.CSV:
            return self._chunk_csv(content)
        else:
            return self._chunk_generic_text(content)
    
    def _chunk_python_code(self, content: str) -> List[FileChunk]:
        """Chunk Python code by functions and classes."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_start = 1
        chunk_index = 0
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # Check for function or class boundaries
            if (line.strip().startswith('def ') or 
                line.strip().startswith('class ') or
                (len(current_chunk) > self.max_chunk_size // 50)):  # Lines to chars approximation
                
                if len('\n'.join(current_chunk)) > self.max_chunk_size:
                    # Create chunk
                    chunk_content = '\n'.join(current_chunk)
                    chunk_type = "function" if "def " in chunk_content else "class" if "class " in chunk_content else "content"
                    
                    chunks.append(FileChunk(
                        content=chunk_content,
                        start_line=current_start,
                        end_line=i,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(chunk_content),
                        chunk_type=chunk_type,
                        importance=self._calculate_importance(chunk_content, chunk_type)
                    ))
                    
                    # Start new chunk with overlap
                    overlap_lines = current_chunk[-self.overlap_size // 50:] if len(current_chunk) > self.overlap_size // 50 else []
                    current_chunk = overlap_lines
                    current_start = max(1, i - len(overlap_lines))
                    chunk_index += 1
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(FileChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_index=chunk_index,
                token_count=self._estimate_tokens(chunk_content),
                chunk_type="content"
            ))
        
        return chunks
    
    def _chunk_javascript_code(self, content: str) -> List[FileChunk]:
        """Chunk JavaScript/TypeScript code by functions."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_start = 1
        chunk_index = 0
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # Track braces for function boundaries
            brace_count += line.count('{') - line.count('}')
            
            # Check for function boundaries or size limit
            if (('function ' in line or '=>' in line or brace_count == 0) and 
                len('\n'.join(current_chunk)) > self.max_chunk_size):
                
                chunk_content = '\n'.join(current_chunk)
                chunk_type = "function" if "function " in chunk_content or "=>" in chunk_content else "content"
                
                chunks.append(FileChunk(
                    content=chunk_content,
                    start_line=current_start,
                    end_line=i,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(chunk_content),
                    chunk_type=chunk_type,
                    importance=self._calculate_importance(chunk_content, chunk_type)
                ))
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-self.overlap_size // 50:] if len(current_chunk) > self.overlap_size // 50 else []
                current_chunk = overlap_lines
                current_start = max(1, i - len(overlap_lines))
                chunk_index += 1
                brace_count = 0
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(FileChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_index=chunk_index,
                token_count=self._estimate_tokens(chunk_content),
                chunk_type="content"
            ))
        
        return chunks
    
    def _chunk_markdown(self, content: str) -> List[FileChunk]:
        """Chunk Markdown by headers."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_start = 1
        chunk_index = 0
        current_header_level = 0
        
        for i, line in enumerate(lines, 1):
            # Check for headers
            header_match = re.match(r'^(#+)\s+(.+)', line)
            
            if header_match:
                header_level = len(header_match.group(1))
                
                # If we have content and hit a same or higher level header, create chunk
                if current_chunk and header_level <= current_header_level:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(FileChunk(
                        content=chunk_content,
                        start_line=current_start,
                        end_line=i-1,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(chunk_content),
                        chunk_type="section",
                        importance=1.2 if current_header_level <= 2 else 1.0  # Higher importance for main sections
                    ))
                    
                    current_chunk = [line]
                    current_start = i
                    chunk_index += 1
                else:
                    current_chunk.append(line)
                
                current_header_level = header_level
            else:
                current_chunk.append(line)
                
                # Size-based chunking fallback
                if len('\n'.join(current_chunk)) > self.max_chunk_size:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(FileChunk(
                        content=chunk_content,
                        start_line=current_start,
                        end_line=i,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(chunk_content),
                        chunk_type="section"
                    ))
                    
                    current_chunk = []
                    current_start = i + 1
                    chunk_index += 1
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(FileChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_index=chunk_index,
                token_count=self._estimate_tokens(chunk_content),
                chunk_type="section"
            ))
        
        return chunks
    
    def _chunk_json(self, content: str) -> List[FileChunk]:
        """Chunk JSON content intelligently."""
        try:
            data = json.loads(content)
            
            # If it's a small JSON, return as single chunk
            if len(content) <= self.max_chunk_size:
                return [FileChunk(
                    content=content,
                    start_line=1,
                    end_line=len(content.split('\n')),
                    chunk_index=0,
                    token_count=self._estimate_tokens(content),
                    chunk_type="json_complete"
                )]
            
            # For large JSON, chunk by top-level keys
            chunks = []
            if isinstance(data, dict):
                chunk_index = 0
                for key, value in data.items():
                    chunk_data = {key: value}
                    chunk_content = json.dumps(chunk_data, indent=2)
                    
                    chunks.append(FileChunk(
                        content=chunk_content,
                        start_line=1,  # JSON chunking doesn't preserve exact line numbers
                        end_line=len(chunk_content.split('\n')),
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(chunk_content),
                        chunk_type="json_object",
                        summary=f"JSON object with key: {key}"
                    ))
                    chunk_index += 1
            else:
                # Fallback to text chunking for non-object JSON
                return self._chunk_generic_text(content)
            
            return chunks
            
        except json.JSONDecodeError:
            # Fallback to text chunking if JSON is invalid
            return self._chunk_generic_text(content)
    
    def _chunk_csv(self, content: str) -> List[FileChunk]:
        """Chunk CSV content by rows."""
        lines = content.split('\n')
        
        if len(lines) <= 100:  # Small CSV, single chunk
            return [FileChunk(
                content=content,
                start_line=1,
                end_line=len(lines),
                chunk_index=0,
                token_count=self._estimate_tokens(content),
                chunk_type="csv_complete"
            )]
        
        chunks = []
        header_line = lines[0] if lines else ""
        chunk_size = self.max_chunk_size // (len(header_line) + 50)  # Estimate rows per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = [header_line] + lines[i+1:i+chunk_size+1] if i > 0 else lines[i:i+chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            chunks.append(FileChunk(
                content=chunk_content,
                start_line=i+1,
                end_line=min(i+chunk_size, len(lines)),
                chunk_index=len(chunks),
                token_count=self._estimate_tokens(chunk_content),
                chunk_type="csv_section",
                summary=f"CSV rows {i+1} to {min(i+chunk_size, len(lines))}"
            ))
        
        return chunks
    
    def _chunk_generic_text(self, content: str) -> List[FileChunk]:
        """Generic text chunking with sentence boundary awareness."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""
        current_start = 1
        chunk_index = 0
        current_line = 1
        
        for paragraph in paragraphs:
            paragraph_lines = paragraph.count('\n') + 1
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk + paragraph) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunks.append(FileChunk(
                    content=current_chunk.strip(),
                    start_line=current_start,
                    end_line=current_line - 1,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(current_chunk),
                    chunk_type="text"
                ))
                
                # Start new chunk with overlap
                sentences = re.split(r'[.!?]+\s+', current_chunk)
                overlap = ' '.join(sentences[-2:]) if len(sentences) >= 2 else ""
                current_chunk = overlap + " " + paragraph if overlap else paragraph
                current_start = current_line
                chunk_index += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            
            current_line += paragraph_lines + 1  # +1 for the empty line between paragraphs
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(FileChunk(
                content=current_chunk.strip(),
                start_line=current_start,
                end_line=current_line,
                chunk_index=chunk_index,
                token_count=self._estimate_tokens(current_chunk),
                chunk_type="text"
            ))
        
        return chunks
    
    def _generate_summary(self, content: str, file_type: FileType) -> str:
        """Generate a summary of the file content."""
        lines = content.split('\n')
        total_lines = len(lines)
        total_chars = len(content)
        
        # Type-specific summary generation
        if file_type == FileType.PYTHON:
            functions = len([line for line in lines if line.strip().startswith('def ')])
            classes = len([line for line in lines if line.strip().startswith('class ')])
            imports = len([line for line in lines if line.strip().startswith(('import ', 'from '))])
            
            summary = f"Python file with {functions} functions, {classes} classes, and {imports} imports. Total: {total_lines} lines."
            
        elif file_type == FileType.JAVASCRIPT:
            functions = len([line for line in lines if 'function ' in line or '=>' in line])
            imports = len([line for line in lines if line.strip().startswith(('import ', 'export '))])
            
            summary = f"JavaScript file with {functions} functions and {imports} import/export statements. Total: {total_lines} lines."
            
        elif file_type == FileType.MARKDOWN:
            headers = len([line for line in lines if line.strip().startswith('#')])
            links = len(re.findall(r'\[.*?\]\(.*?\)', content))
            
            summary = f"Markdown document with {headers} headers and {links} links. Total: {total_lines} lines."
            
        elif file_type == FileType.JSON:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    summary = f"JSON object with {len(data)} top-level keys. Size: {total_chars} characters."
                elif isinstance(data, list):
                    summary = f"JSON array with {len(data)} items. Size: {total_chars} characters."
                else:
                    summary = f"JSON file containing {type(data).__name__}. Size: {total_chars} characters."
            except:
                summary = f"JSON file (invalid format). Size: {total_chars} characters."
        
        elif file_type == FileType.CSV:
            rows = total_lines - 1  # Excluding header
            cols = len(lines[0].split(',')) if lines else 0
            summary = f"CSV file with {rows} data rows and {cols} columns. Total: {total_lines} lines."
        
        else:
            # Generic text summary
            words = len(content.split())
            paragraphs = len(re.split(r'\n\s*\n', content))
            summary = f"Text file with {words} words, {paragraphs} paragraphs, and {total_lines} lines."
        
        return summary
    
    def _extract_metadata(self, content: str, file_type: FileType) -> Dict[str, Any]:
        """Extract metadata from file content."""
        metadata = {
            "file_type": file_type.value,
            "total_chars": len(content),
            "total_lines": len(content.split('\n'))
        }
        
        if file_type == FileType.PYTHON:
            metadata.update({
                "functions": len([line for line in content.split('\n') if line.strip().startswith('def ')]),
                "classes": len([line for line in content.split('\n') if line.strip().startswith('class ')]),
                "imports": len([line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))])
            })
        
        elif file_type == FileType.JSON:
            try:
                data = json.loads(content)
                metadata.update({
                    "json_type": type(data).__name__,
                    "json_size": len(data) if isinstance(data, (list, dict)) else 1
                })
            except:
                metadata["json_valid"] = False
        
        elif file_type == FileType.CSV:
            lines = content.split('\n')
            metadata.update({
                "csv_rows": len(lines) - 1,
                "csv_columns": len(lines[0].split(',')) if lines else 0
            })
        
        return metadata
    
    def _detect_language(self, file_type: FileType) -> Optional[str]:
        """Detect programming language from file type."""
        language_map = {
            FileType.PYTHON: "python",
            FileType.JAVASCRIPT: "javascript",
            FileType.MARKDOWN: "markdown",
            FileType.JSON: "json",
            FileType.CSV: "csv",
            FileType.HTML: "html",
            FileType.XML: "xml",
            FileType.YAML: "yaml"
        }
        return language_map.get(file_type)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 4 characters per token
        return len(text) // 4
    
    def _calculate_importance(self, content: str, chunk_type: str) -> float:
        """Calculate importance score for a chunk."""
        base_importance = 1.0
        
        # Boost importance for certain chunk types
        if chunk_type in ["function", "class"]:
            base_importance = 1.2
        elif chunk_type == "header":
            base_importance = 1.3
        
        # Boost for certain keywords
        important_keywords = ["class", "def", "function", "import", "export", "interface", "type"]
        keyword_count = sum(1 for keyword in important_keywords if keyword in content.lower())
        
        return min(2.0, base_importance + (keyword_count * 0.1)) 