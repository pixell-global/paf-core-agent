# Short-term Memory and File Content Support

This document describes the implementation of short-term memory support and enhanced file content handling in the PAF Core Agent.

## Overview

The PAF Core Agent now supports:

1. **Short-term Memory**: Conversation history processing for contextual awareness
2. **Enhanced File Content Support**: Advanced file processing with signed URLs, content analysis, and type detection
3. **UPEE Integration**: Both features are fully integrated into the UPEE (Understand → Plan → Execute → Evaluate) processing loop

## Features Implemented

### 1. Short-term Memory Support

#### ConversationMessage Schema
```python
class ConversationMessage(BaseModel):
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    files: Optional[List[FileContent]] = None
```

#### Memory Processing
- **History Limit**: Configurable limit (1-20 messages, default 10)
- **Token Estimation**: Automatic token counting for context management
- **History Truncation**: Intelligent truncation when limits are exceeded
- **File References**: Track files mentioned in conversation history

#### Integration Points
- **ChatRequest**: New `history` and `memory_limit` fields
- **Understand Phase**: Processes conversation context for intent analysis
- **Plan Phase**: Incorporates memory context into strategy planning
- **Context Summary**: Generates human-readable summaries of conversation history

### 2. Enhanced File Content Support

#### FileContent Schema
```python
class FileContent(BaseModel):
    file_name: str
    file_path: Optional[str] = None
    content: Optional[str] = None  # For small files (<100KB)
    signed_url: Optional[str] = None  # For large files
    file_type: str  # MIME type or extension
    file_size: int  # Size in bytes
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### File Processing Features
- **Type Detection**: Automatic MIME type and extension detection
- **Content Analysis**: Language-specific analysis (Python, JavaScript, Markdown, etc.)
- **Signed URL Support**: HTTP/HTTPS fetching for large files
- **Size Limits**: 100MB maximum, 100KB threshold for inline content
- **Legacy Support**: Backward compatibility with existing FileContext format

#### Supported File Types
- **Programming Languages**: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, etc.
- **Data Formats**: JSON, YAML, CSV, XML
- **Documentation**: Markdown, plain text, HTML
- **Configuration**: .env, .ini, .cfg, .conf, .toml files

### 3. UPEE Integration

#### Understand Phase Enhancements
- **File Processing**: Intelligent content analysis with language detection
- **Memory Analysis**: Conversation history processing and token estimation
- **Context Requirements**: Enhanced token calculation including history and files
- **Complexity Assessment**: Factors in conversation length and file complexity

#### Plan Phase Enhancements
- **File Processing Strategy**: Plans how to handle different file types and sizes
- **Memory Usage Planning**: Determines optimal history inclusion strategy
- **Model Selection**: Considers file and memory context for model recommendations
- **External Call Planning**: Enhanced logic for worker task distribution

## API Changes

### Enhanced ChatRequest
```python
{
    "message": "Your question or request",
    "files": [
        {
            "file_name": "example.py",
            "content": "print('hello')",  # OR signed_url for large files
            "file_type": "text/x-python",
            "file_size": 15
        }
    ],
    "history": [
        {
            "role": "user",
            "content": "Previous message",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    ],
    "memory_limit": 10,
    "show_thinking": true
}
```

### New Validation Rules
- Files must have either `content` or `signed_url`
- File size limit: 100MB
- History limit: 50 messages maximum
- Memory limit: 1-20 messages
- Valid roles: 'user', 'assistant', 'system'

### Enhanced Status Endpoint
The `/api/chat/status` endpoint now reports:
```json
{
    "features": {
        "conversation_history": true,
        "short_term_memory": true,
        "signed_url_support": true,
        "file_context": true
    }
}
```

## Implementation Details

### File Processor (`app/core/file_processor.py`)
- **FileProcessor Class**: Main processing engine
- **Content Analysis**: Language-specific metadata extraction
- **Signed URL Fetching**: HTTP client for remote file access
- **Context Summary Generation**: Human-readable summaries

### UPEE Phase Updates
- **Understand Phase** (`app/core/understand.py`): Enhanced with file and memory processing
- **Plan Phase** (`app/core/plan.py`): Extended planning for context-aware strategies

### Schema Extensions (`app/schemas.py`)
- **FileContent**: New comprehensive file schema
- **ConversationMessage**: Message structure for history
- **ChatRequest**: Enhanced with history and memory fields

## Usage Examples

### Basic Memory Usage
```python
chat_request = ChatRequest(
    message="What did we discuss about Python?",
    history=[
        ConversationMessage(role="user", content="I'm learning Python"),
        ConversationMessage(role="assistant", content="Great! What aspect interests you?")
    ],
    memory_limit=5
)
```

### File Processing with Signed URLs
```python
chat_request = ChatRequest(
    message="Analyze this large codebase",
    files=[
        FileContent(
            file_name="large_app.py",
            signed_url="https://storage.example.com/signed-url-here",
            file_type="text/x-python",
            file_size=500000
        )
    ]
)
```

### Combined Files and Memory
```python
chat_request = ChatRequest(
    message="Based on our previous discussion, review this code",
    files=[
        FileContent(
            file_name="utils.py",
            content="def helper(): pass",
            file_type="text/x-python",
            file_size=20
        )
    ],
    history=[
        ConversationMessage(role="user", content="I need help with code structure")
    ],
    memory_limit=10
)
```

## Testing

### Unit Tests
- File processor functionality: `scripts/test_memory_and_files.py`
- UPEE phase integration
- Schema validation

### Integration Tests
- Server endpoint testing: `scripts/test_integration.py`
- Validation testing
- Feature availability checks

### Running Tests
```bash
# Unit tests
python scripts/test_memory_and_files.py

# Integration tests (requires running server)
./scripts/start.sh  # In one terminal
python scripts/test_integration.py  # In another terminal
```

## Performance Considerations

### Memory Management
- History messages are limited to prevent memory bloat
- Token estimation helps manage context windows
- Large files use signed URLs to avoid memory issues

### Processing Efficiency
- Lazy loading of file content from signed URLs
- Intelligent truncation of conversation history
- Parallel processing support for multiple files

### Caching
- File content caching with MD5 hashing
- Processed file metadata caching
- Context summary caching

## Error Handling

### File Processing Errors
- Invalid signed URLs: Graceful degradation with error logging
- Unsupported file types: Continue processing other files
- Size limit exceeded: Clear error messages

### Memory Processing Errors
- Invalid message roles: Validation at API level
- Malformed history: Sanitization and error reporting
- Token limit exceeded: Automatic truncation with warnings

## Future Enhancements

### Planned Features
- **Long-term Memory**: Persistent conversation storage
- **File Versioning**: Track file changes across conversations
- **Advanced Analytics**: Deeper content analysis and insights
- **Streaming File Processing**: Real-time processing of large files

### Optimization Opportunities
- **Content Compression**: Reduce memory usage for large conversations
- **Smart Caching**: Predictive file content caching
- **Distributed Processing**: Scale file processing across workers

## Configuration

### Environment Variables
```bash
# File processing limits
MAX_FILE_SIZE=104857600  # 100MB
INLINE_CONTENT_LIMIT=102400  # 100KB

# Memory limits
MAX_HISTORY_MESSAGES=50
DEFAULT_MEMORY_LIMIT=10

# Processing timeouts
SIGNED_URL_TIMEOUT=30
FILE_PROCESSING_TIMEOUT=60
```

### Settings Integration
All limits and timeouts are configurable through the existing settings system in `app/settings.py`.

## Security Considerations

### File Access
- Signed URL validation and timeout handling
- Content type verification
- Size limit enforcement

### Memory Safety
- Input sanitization for conversation history
- XSS prevention in file content processing
- Rate limiting for memory-intensive operations

## Migration Guide

### Existing Applications
- Legacy `FileContext` format remains supported
- Gradual migration path to new `FileContent` format
- Backward compatibility maintained for all existing endpoints

### API Clients
- New fields are optional in `ChatRequest`
- Enhanced responses provide additional metadata
- Existing integrations continue to work unchanged