# PAF Core Agent

A cloud-native Python microservice implementing the UPEE (Understand → Plan → Execute → Evaluate) loop for intelligent chat interactions with multi-provider LLM support.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or Poetry

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paf-core-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Start the development server**
   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

The service will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 🏗️ Architecture

### UPEE Loop
The core cognitive loop consists of four phases:

1. **Understand** - Parse and analyze user input with context
2. **Plan** - Develop response strategy and identify required resources
3. **Execute** - Generate response using appropriate LLM providers
4. **Evaluate** - Assess response quality and refine if needed

### Key Features

- **🔄 Server-Sent Events (SSE)** - Real-time streaming chat responses
- **🧠 Multi-Provider LLM** - OpenAI, Anthropic Claude, AWS Bedrock support
- **📁 File Context** - Intelligent file processing and summarization
- **🔗 gRPC Integration** - Communication with downstream worker agents
- **🤝 A2A Integration** - Agent-to-Agent communication with card discovery
- **📊 Observability** - Structured logging, Prometheus metrics, AWS X-Ray tracing
- **🔒 Security** - JWT/HMAC authentication, mTLS for gRPC
- **🐳 Container Ready** - Docker support with optimized image size
- **☁️ Cloud Native** - AWS Fargate deployment with auto-scaling

## 📡 API Endpoints

### Chat Streaming
```http
POST /api/chat/stream
Content-Type: application/json

{
  "message": "Hello, how can you help me?",
  "show_thinking": true,
  "files": [...],
  "model": "gpt-4",
  "temperature": 0.7
}
```

**Response**: Server-Sent Events stream with:
- `thinking` events (UPEE phase insights)
- `content` events (response chunks)
- `complete` event (metadata and stats)
- `done` event (stream termination)

### Health Check
```http
GET /api/health
```

Returns service health status including LLM provider availability.

### Available Models
```http
GET /api/chat/models
```

Lists all available LLM models and their status.

### A2A Integration (Agent-to-Agent Protocol)
```http
GET /api/chat/a2a/agents
```

Discovers available A2A agents using the standard A2A protocol.

```http
GET /api/chat/a2a/agents/{agent_id}
```

Gets detailed information about a specific A2A agent using Agent Card standard.

```http
GET /api/chat/a2a/status
```

Checks A2A server health and configuration status.

#### Legacy A2A Endpoints (deprecated)
```http
GET /api/chat/a2a/cards        # Use /api/chat/a2a/agents instead
GET /api/chat/a2a/cards/{id}   # Use /api/chat/a2a/agents/{id} instead
```

## 🛠️ Development

### Project Structure
```
paf-core-agent/
├── app/
│   ├── api/                 # FastAPI routers
│   ├── core/                # UPEE logic
│   ├── llm_providers/       # Multi-provider abstraction
│   ├── grpc_clients/        # gRPC client implementations
│   ├── utils/               # Utilities (logging, auth, metrics)
│   ├── schemas.py           # Pydantic models
│   └── settings.py          # Configuration
├── tests/                   # Test suites
├── scripts/                 # Development scripts
├── proto/                   # Protocol buffer definitions
└── requirements.txt         # Python dependencies
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | No | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | No | - |
| `AWS_REGION` | AWS region for Bedrock | No | us-east-1 |
| `DEBUG` | Enable debug mode | No | false |
| `MAX_CONTEXT_TOKENS` | Maximum context window | No | 4000 |
| `DEFAULT_MODEL` | Default LLM model | No | gpt-4o |
| `A2A_ENABLED` | Enable A2A functionality | No | true |
| `A2A_SERVER_URL` | A2A server endpoint | No | http://localhost:9999 |
| `A2A_AGENT_CARD` | Agent card identifier | No | - |
| `A2A_TIMEOUT` | A2A request timeout (seconds) | No | 10 |

### A2A SDK 설치 (선택사항)

공식 Google A2A SDK를 사용하려면:

```bash
# 공식 Google A2A 저장소 클론
git clone https://github.com/google/A2A.git
cd A2A/a2a-python-sdk

# 개발 모드로 설치
pip install -e .
```

SDK를 설치하지 않으면 표준 A2A 프로토콜 HTTP 구현을 사용합니다.

### A2A 기능 테스트

A2A 기능을 테스트하려면:

```bash
# A2A 테스트 실행
python scripts/test_a2a.py
```

이 스크립트는 다음을 테스트합니다:
- A2A 서버 상태 확인
- 에이전트 탐색 (Agent Discovery)
- 메시지 전송

### Running Tests
```bash
pytest tests/ -v --cov=app
```

### Code Quality
```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint
flake8 app/ tests/

# Type checking
mypy app/
```

## 🐳 Docker

Build and run with Docker:

```bash
# Build image
docker build -t paf-core-agent .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  paf-core-agent
```

## ☁️ Deployment

### AWS Fargate

The service is designed for deployment on AWS Fargate with:
- Application Load Balancer for HTTP/HTTPS traffic
- Auto Scaling based on CPU/memory metrics
- ECS service with health checks
- CloudWatch logging and monitoring

See `terraform/` directory for Infrastructure as Code examples.

### Environment Configuration

For production deployment:

1. Use AWS Secrets Manager for API keys
2. Configure VPC with private subnets for gRPC traffic
3. Set up CloudWatch dashboards for monitoring
4. Enable AWS X-Ray for distributed tracing

## 📊 Monitoring

### Metrics
The service exposes Prometheus metrics at `/metrics`:

- Request latency and throughput
- Token usage per provider
- UPEE phase timing
- Error rates and types

### Logging
Structured JSON logs include:

- Request tracing with correlation IDs
- UPEE phase events
- LLM provider calls
- Performance metrics

### Health Checks
- `/api/health` - Comprehensive health status
- `/api/health/live` - Liveness probe
- `/api/health/ready` - Readiness probe

## 🔧 Configuration

### LLM Providers
Configure multiple providers for redundancy and cost optimization:

```python
# Environment variables
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AWS_REGION=us-east-1

# Default routing
DEFAULT_MODEL=gpt-3.5-turbo
```

### Performance Tuning
- `MAX_CONCURRENT_REQUESTS=150` - Concurrent request limit
- `REQUEST_TIMEOUT=30` - Request timeout in seconds
- `MAX_CONTEXT_TOKENS=4000` - Context window size

## 🔒 Security

- **Authentication**: HMAC signatures or JWT tokens
- **Transport**: HTTPS for client traffic, mTLS for gRPC
- **Secrets**: AWS Secrets Manager integration
- **Network**: VPC isolation for inter-service communication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the health status at `/api/health`

---

**Status**: 🚧 Under Development - Basic scaffold complete, implementing UPEE loop 