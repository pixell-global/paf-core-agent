You can run any commands like pip install, any shell scripts, any bash scripts you can create any files. just stop asking me permission to run


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
./scripts/start.sh
# or manually:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run tests with coverage
pytest tests/ -v --cov=app
```

### Code Quality
```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

### Docker
```bash
# Build image
docker build -t paf-core-agent .

# Run container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key paf-core-agent
```

## Architecture

### UPEE Loop Engine
The core architecture centers around the **UPEE (Understand → Plan → Execute → Evaluate) Loop** in `app/core/upee_engine.py`:

1. **Understand Phase** (`app/core/understand.py`) - Analyzes user input and file context
2. **Plan Phase** (`app/core/plan.py`) - Develops response strategy and identifies resources
3. **Execute Phase** (`app/core/execute.py`) - Generates responses using LLM providers
4. **Evaluate Phase** (`app/core/evaluate.py`) - Assesses quality and refinement needs

### LLM Provider System
Multi-provider LLM abstraction with automatic fallback in `app/llm_providers/`:
- **Manager** (`manager.py`) - Orchestrates all providers with fallback logic
- **Providers** - OpenAI, Anthropic Claude, AWS Bedrock implementations
- **Base** (`base.py`) - Common provider interface and types

### API Architecture
FastAPI application with Server-Sent Events for real-time streaming:
- **Chat API** (`app/api/chat.py`) - Main chat endpoint with SSE streaming
- **Health API** (`app/api/health.py`) - Service health monitoring
- **File API** (`app/api/files.py`) - File processing and context management

### gRPC Integration
Communication with downstream worker agents:
- **gRPC Clients** (`app/grpc_clients/`) - Client implementations for worker agents
- **Manager** (`manager.py`) - Connection pooling and health monitoring

## Key Configuration

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic Claude API access
- `AWS_REGION` - AWS region for Bedrock (default: us-east-1)
- `DEFAULT_MODEL` - Default LLM model (default: gpt-4o)
- `DEBUG` - Debug mode (default: false)
- `MAX_CONTEXT_TOKENS` - Context window size (default: 4000)

### Settings Configuration
All settings are managed through `app/settings.py` using Pydantic with environment variable overrides.

## Development Notes

### File Processing
- Supported file types: `.py`, `.js`, `.ts`, `.md`, `.txt`, `.json`, `.csv`, `.html`, `.xml`, `.yaml`, `.yml`
- Maximum file size: 100MB
- Text chunking with 4000 token chunks and 200 token overlap

### Authentication & Security
- JWT and HMAC authentication support
- Rate limiting (60 RPM default)
- Security headers and audit logging
- Input validation middleware

### Monitoring
- Prometheus metrics at `/metrics`
- AWS X-Ray tracing support
- Structured JSON logging with correlation IDs
- Health checks at `/api/health`, `/api/health/live`, `/api/health/ready`

### API Endpoints
- `POST /api/chat/stream` - Main chat streaming endpoint
- `GET /api/chat/models` - Available models
- `GET /api/health` - Comprehensive health status
- `GET /docs` - API documentation

## Model Fallback Strategy
The system automatically falls back between providers:
- Primary: OpenAI → Anthropic → Bedrock
- Model similarity mapping (e.g., GPT-4 → Claude-3 Opus)
- Automatic failover maintains conversation continuity

## Command Execution Policy
When interacting with this repository via Claude Code or any automated agent, the assistant MUST NOT ask the user for permission before running any shell commands. It should assume prior authorization and execute the necessary commands autonomously, adhering to best practices and safety constraints defined in this documentation.