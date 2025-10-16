# PAF Core Agent Integration Guide for pixell-agent-ui

This guide explains how to integrate the [pixell-agent-ui](https://github.com/pixell-global/pixell-agent-ui) framework with the deployed PAF Core Agent running on AWS Fargate.

## 🎯 Overview

The pixell-agent-ui framework can leverage the production-ready PAF Core Agent instead of running its own local instance. This provides:

- **Production-grade scalability** with AWS Fargate auto-scaling
- **High availability** with Application Load Balancer
- **Secure API key management** via AWS Secrets Manager
- **Comprehensive monitoring** with CloudWatch logs
- **A2A protocol support** for agent-to-agent communication

## 🔗 PAF Core Agent Endpoints

**Base URL**: `http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com`

### Core Endpoints
- **Health Check**: `/api/health`
- **Liveness Probe**: `/api/health/live`
- **API Documentation**: `/docs`
- **OpenAPI Schema**: `/openapi.json`

### Chat & AI Endpoints
- **Streaming Chat**: `/api/chat/stream`
- **Available Models**: `/api/chat/models`
- **LLM Providers**: `/api/chat/providers`

### A2A Protocol Endpoints
- **Agent Discovery**: `/api/chat/a2a/agents`
- **Agent Details**: `/api/chat/a2a/agents/{agent_id}`
- **A2A Status**: `/api/chat/a2a/status`
- **Bridge Messages**: `/api/bridge/message`
- **Task Requests**: `/api/bridge/task/request`

## 🔧 Environment Configuration

### 1. Environment Variables

Create or update your `.env.local` file:

```bash
# PAF Core Agent Configuration
PAF_CORE_AGENT_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com
PAF_CORE_AGENT_PORT=8000
PAF_CORE_AGENT_TIMEOUT=30000

# A2A Configuration
A2A_ENABLED=true
A2A_SERVER_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com
A2A_AGENT_CARD=paf-core-agent
A2A_AGENT_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com

# API Keys (managed via environment variables in ECS task definition)
# Note: PAF Core Agent uses environment variables instead of AWS Secrets Manager
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# pixell-agent-ui specific variables
NEXT_PUBLIC_API_URL=http://localhost:3001
SUPABASE_URL=http://localhost:54321
SUPABASE_ANON_KEY=your-supabase-key
```

### 2. Docker Compose Configuration

Update your `docker-compose.yml` or `docker-compose.dev.yml`:

```yaml
version: '3.8'
services:
  orchestrator:
    environment:
      # PAF Core Agent Connection
      - PAF_CORE_AGENT_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com
      - PAF_CORE_AGENT_PORT=8000
      - PAF_CORE_AGENT_TIMEOUT=30000
      
      # A2A Configuration
      - A2A_ENABLED=true
      - A2A_SERVER_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com
      - A2A_AGENT_CARD=paf-core-agent
      - A2A_AGENT_URL=http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com
      
      # Other environment variables
      - NODE_ENV=development
      - PORT=3001
```

## 🏗️ Orchestrator Service Implementation

### 1. PAF Core Agent Service Class

Create `orchestrator/src/services/pafCoreAgent.ts`:

```typescript
export interface ChatMessage {
  message: string;
  show_thinking?: boolean;
  model?: string;
  context?: any;
  files?: any[];
}

export interface A2AMessage {
  message_type: string;
  target_agent_id: string;
  payload: any;
  priority?: string;
  correlation_id?: string;
}

export class PAFCoreAgentService {
  private baseUrl: string;
  private timeout: number;

  constructor() {
    this.baseUrl = process.env.PAF_CORE_AGENT_URL!;
    this.timeout = parseInt(process.env.PAF_CORE_AGENT_TIMEOUT || '30000');
    
    if (!this.baseUrl) {
      throw new Error('PAF_CORE_AGENT_URL environment variable is required');
    }
  }

  /**
   * Send a chat message to PAF Core Agent
   */
  async sendChatMessage(message: ChatMessage): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message.message,
        show_thinking: message.show_thinking ?? true,
        model: message.model || 'gpt-4o',
        context: message.context,
        files: message.files
      }),
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`PAF Core Agent request failed: ${response.status} ${response.statusText}`);
    }

    return response;
  }

  /**
   * Get available A2A agents
   */
  async getA2AAgents(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/chat/a2a/agents`, {
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`Failed to get A2A agents: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Send A2A message via bridge
   */
  async sendA2AMessage(message: A2AMessage): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/bridge/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`A2A message failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Check PAF Core Agent health
   */
  async checkHealth(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/health`, {
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get available LLM models
   */
  async getAvailableModels(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/chat/models`, {
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`Failed to get models: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Send task request via bridge
   */
  async sendTaskRequest(task: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/bridge/task/request`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(task),
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`Task request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }
}
```

### 2. Integration with Existing Orchestrator

Update your existing orchestrator to use PAF Core Agent:

```typescript
// orchestrator/src/routes/chat.ts
import { PAFCoreAgentService } from '../services/pafCoreAgent';

const pafService = new PAFCoreAgentService();

export async function handleChatRequest(req: Request, res: Response) {
  try {
    const { message, model, context, files } = req.body;

    // Forward to PAF Core Agent
    const response = await pafService.sendChatMessage({
      message,
      model,
      context,
      files
    });

    // Stream response back to client
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      res.write(value);
    }

    res.end();
  } catch (error) {
    console.error('Chat request failed:', error);
    res.status(500).json({ error: error.message });
  }
}
```

### 3. A2A Integration

```typescript
// orchestrator/src/routes/a2a.ts
import { PAFCoreAgentService } from '../services/pafCoreAgent';

const pafService = new PAFCoreAgentService();

export async function discoverAgents(req: Request, res: Response) {
  try {
    const agents = await pafService.getA2AAgents();
    res.json(agents);
  } catch (error) {
    console.error('Agent discovery failed:', error);
    res.status(500).json({ error: error.message });
  }
}

export async function sendA2AMessage(req: Request, res: Response) {
  try {
    const { message_type, target_agent_id, payload, priority } = req.body;
    
    const result = await pafService.sendA2AMessage({
      message_type,
      target_agent_id,
      payload,
      priority: priority || 'NORMAL'
    });

    res.json(result);
  } catch (error) {
    console.error('A2A message failed:', error);
    res.status(500).json({ error: error.message });
  }
}
```

## 🧪 Testing the Integration

### 1. Health Check

```bash
# Test PAF Core Agent health
curl http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/health

# Expected response:
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "openai": {"status": "healthy"},
    "anthropic": {"status": "healthy"},
    "bedrock": {"status": "healthy"}
  }
}
```

### 2. Chat Integration Test

```bash
# Test chat endpoint
curl -X POST http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello from pixell-agent-ui!",
    "show_thinking": true,
    "model": "gpt-4o"
  }'
```

### 3. A2A Discovery Test

```bash
# Test A2A agent discovery
curl http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/chat/a2a/agents

# Expected response:
{
  "agents": [],
  "count": 0,
  "server_url": "http://localhost:9999",
  "discovery_method": "standard_a2a"
}
```

### 4. Available Models Test

```bash
# Test available models
curl http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/chat/models
```

## 🔄 Migration Strategy

### Phase 1: Parallel Running
1. Keep your existing local PAF Core Agent running
2. Add the external PAF Core Agent as an option
3. Test both configurations

### Phase 2: Gradual Migration
1. Route a percentage of traffic to the external PAF Core Agent
2. Monitor performance and reliability
3. Gradually increase the percentage

### Phase 3: Full Migration
1. Remove local PAF Core Agent from docker-compose
2. Update all references to use the external service
3. Clean up local PAF Core Agent code

## 🛠️ CLI Configuration Updates

Update your pixell CLI to support PAF Core Agent configuration:

```typescript
// packages/cli/src/commands/config.ts
export async function configurePAFCoreAgent() {
  const url = await prompt('PAF Core Agent URL:', {
    default: 'http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com'
  });

  const timeout = await prompt('Request timeout (ms):', {
    default: '30000'
  });

  // Update environment configuration
  await updateEnvFile({
    PAF_CORE_AGENT_URL: url,
    PAF_CORE_AGENT_TIMEOUT: timeout
  });

  console.log('✅ PAF Core Agent configured successfully');
}
```

## 📊 Monitoring and Observability

### 1. Health Monitoring

```typescript
// Add to your orchestrator
setInterval(async () => {
  try {
    const health = await pafService.checkHealth();
    console.log('PAF Core Agent health:', health.status);
  } catch (error) {
    console.error('PAF Core Agent health check failed:', error);
  }
}, 30000); // Check every 30 seconds
```

### 2. Error Handling

```typescript
// Add retry logic with exponential backoff
async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (attempt === maxRetries) throw error;
      
      const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error('Max retries exceeded');
}
```

## 🚀 Benefits of This Integration

1. **Production Ready**: Leverages AWS Fargate auto-scaling and high availability
2. **Cost Effective**: No need to run local PAF Core Agent instances
3. **Secure**: API keys managed via AWS Secrets Manager
4. **Monitored**: CloudWatch logs and metrics included
5. **Scalable**: Handles traffic spikes automatically
6. **A2A Ready**: Full agent-to-agent communication support

## 🔧 Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check if PAF Core Agent URL is correct
   - Verify network connectivity
   - Increase timeout value if needed

2. **Authentication Errors**
   - Verify API keys are set in AWS Secrets Manager
   - Check if PAF Core Agent is healthy

3. **A2A Discovery Issues**
   - Ensure A2A_ENABLED=true
   - Check A2A_SERVER_URL configuration

### Debug Commands

```bash
# Check PAF Core Agent status
curl -v http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/health

# Test specific endpoint
curl -X POST http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}' \
  -v

# Check A2A configuration
curl http://paf-core-agent-prod-alb-62806388.us-east-2.elb.amazonaws.com/api/chat/a2a/status
```

## 📝 Next Steps

1. **Update Environment Configuration**: Set the PAF Core Agent URL in your environment files
2. **Implement Service Integration**: Add the PAF Core Agent service to your orchestrator
3. **Test Integration**: Use the provided test commands to verify connectivity
4. **Update CLI**: Add PAF Core Agent configuration commands
5. **Monitor Performance**: Set up health checks and monitoring
6. **Deploy**: Update your deployment configuration with the new environment variables

This integration allows pixell-agent-ui to leverage a production-ready, scalable PAF Core Agent while maintaining its existing architecture and workflow! 🎉
