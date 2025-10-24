## Deploy to AWS Fargate (AWS CLI)

This guide deploys `paf-core-agent` to Amazon ECS on Fargate behind an Application Load Balancer (ALB), using only AWS CLI and Docker. It assumes your AWS credentials are already configured locally.

### Prerequisites
- Docker 24+
- AWS CLI v2
- jq
- An AWS account with permissions for ECR, ECS, EC2 (VPC/ALB/SG), IAM, CloudWatch Logs, and Secrets Manager

### Service defaults
- Container command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Container port: `8000`
- Health check path: `/api/health/live`
- Logging: CloudWatch Logs (awslogs driver)

### Existing deployment context
This deployment will coexist with the existing `dev.vivid-brand.com` service:
- **Existing service**: `pixell-web-simple` in `pixell-web-cluster`
- **Existing ALB**: `pac-alb` (pac-alb-2089685514.us-east-2.elb.amazonaws.com)
- **Shared VPC**: `vpc-0dc5816f0b041abad` (172.31.0.0/16)
- **Shared subnets**: 3 public subnets across us-east-2a, us-east-2b, us-east-2c
- **Existing target group**: `pac-web-tg` (port 3000, health check `/api/health`)

The new `paf-core-agent` service will use the same VPC but create its own ALB, target group, and ECS cluster for isolation.

---

## 1) Set environment variables
Update values as needed. **Note**: Using us-east-2 region and existing VPC to match dev.vivid-brand.com deployment.

```bash
export AWS_REGION=us-east-2
export APP_NAME=paf-core-agent
export ENV=prod

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=$APP_NAME
export IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# Use existing VPC and subnets from dev.vivid-brand.com deployment
export VPC_ID=vpc-0dc5816f0b041abad
export SUBNET_1=subnet-035d6ed0a581e57df  # us-east-2a
export SUBNET_2=subnet-059d23db977f85843  # us-east-2b
export SUBNET_3=subnet-0fd1d99dab3fdf17b  # us-east-2c

aws configure set region $AWS_REGION
```

---

## 2) Build and push Docker image to ECR

```bash
# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPO >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name $ECR_REPO --image-scanning-configuration scanOnPush=true

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push
docker build -t $ECR_REPO:$IMAGE_TAG -f Dockerfile .
docker tag $ECR_REPO:$IMAGE_TAG $IMAGE_URI
docker push $IMAGE_URI
```

---

## 3) Networking and load balancer (existing VPC)
These commands deploy into the same VPC as dev.vivid-brand.com with existing public subnets and an internet-facing ALB.

```bash
# VPC and subnets are already set in environment variables above
# VPC: vpc-0dc5816f0b041abad (172.31.0.0/16)
# Subnets: 3 public subnets across us-east-2a, us-east-2b, us-east-2c

# Security groups: one for ALB (ingress 80) and one for ECS service (ingress 8000 from ALB)
export ALB_SG_ID=$(aws ec2 create-security-group \
  --group-name ${APP_NAME}-${ENV}-alb-sg \
  --description "ALB SG for ${APP_NAME}" \
  --vpc-id $VPC_ID \
  --query GroupId --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0

export SVC_SG_ID=$(aws ec2 create-security-group \
  --group-name ${APP_NAME}-${ENV}-svc-sg \
  --description "Service SG for ${APP_NAME}" \
  --vpc-id $VPC_ID \
  --query GroupId --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $SVC_SG_ID --protocol tcp --port 8000 --source-group $ALB_SG_ID

# Target group (IP targets, health check on /api/health/live)
export TG_ARN=$(aws elbv2 create-target-group \
  --name ${APP_NAME}-${ENV}-tg \
  --protocol HTTP --port 8000 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /api/health/live \
  --health-check-interval-seconds 15 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# ALB (using all 3 subnets for better availability)
export ALB_ARN=$(aws elbv2 create-load-balancer \
  --name ${APP_NAME}-${ENV}-alb \
  --type application \
  --scheme internet-facing \
  --subnets $SUBNET_1 $SUBNET_2 $SUBNET_3 \
  --security-groups $ALB_SG_ID \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

export LISTENER_ARN=$(aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --query 'Listeners[0].ListenerArn' --output text)

# CloudWatch log group for ECS
aws logs create-log-group --log-group-name /ecs/${APP_NAME} 2>/dev/null || true
aws logs put-retention-policy --log-group-name /ecs/${APP_NAME} --retention-in-days 30
```

---

## 4) IAM roles (task execution + task role)

```bash
# Trust policy for ECS tasks
cat > /tmp/ecsTaskTrust.json << 'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON

# Execution role (pull from ECR, send logs to CloudWatch)
export EXEC_ROLE_NAME=${APP_NAME}-${ENV}-exec
aws iam create-role --role-name $EXEC_ROLE_NAME \
  --assume-role-policy-document file:///tmp/ecsTaskTrust.json >/dev/null 2>&1 || true
aws iam attach-role-policy --role-name $EXEC_ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null 2>&1 || true

# Task role (runtime access, e.g., Secrets Manager)
export TASK_ROLE_NAME=${APP_NAME}-${ENV}-task
aws iam create-role --role-name $TASK_ROLE_NAME \
  --assume-role-policy-document file:///tmp/ecsTaskTrust.json >/dev/null 2>&1 || true

# Optional: grant read access to Secrets Manager (restrict as needed)
cat > /tmp/ecsSecretsAccess.json << 'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "kms:Decrypt"
      ],
      "Resource": "*"
    }
  ]
}
JSON
aws iam put-role-policy --role-name $TASK_ROLE_NAME \
  --policy-name ${APP_NAME}-${ENV}-secrets-access \
  --policy-document file:///tmp/ecsSecretsAccess.json

export EXEC_ROLE_ARN=$(aws iam get-role --role-name $EXEC_ROLE_NAME --query 'Role.Arn' --output text)
export TASK_ROLE_ARN=$(aws iam get-role --role-name $TASK_ROLE_NAME --query 'Role.Arn' --output text)
```

---

## 5) Store provider API keys in Secrets Manager (optional but recommended)

```bash
# Create secrets; replace values as appropriate
aws secretsmanager create-secret --name ${APP_NAME}/OPENAI_API_KEY --secret-string 'sk-your-openai-key' >/dev/null 2>&1 || true
aws secretsmanager create-secret --name ${APP_NAME}/ANTHROPIC_API_KEY --secret-string 'sk-ant-your-anthropic-key' >/dev/null 2>&1 || true

export OPENAI_SECRET_ARN=$(aws secretsmanager describe-secret --secret-id ${APP_NAME}/OPENAI_API_KEY --query 'ARN' --output text)
export ANTHROPIC_SECRET_ARN=$(aws secretsmanager describe-secret --secret-id ${APP_NAME}/ANTHROPIC_API_KEY --query 'ARN' --output text)
```

---

## 6) ECS cluster and task definition

```bash
# ECS cluster
export CLUSTER_NAME=${APP_NAME}-${ENV}
aws ecs create-cluster --cluster-name $CLUSTER_NAME >/dev/null 2>&1 || true

# Task definition JSON
cat > /tmp/taskdef.json << JSON
{
  "family": "${APP_NAME}-${ENV}",
  "networkMode": "awsvpc",
  "cpu": "512",
  "memory": "1024",
  "requiresCompatibilities": ["FARGATE"],
  "executionRoleArn": "${EXEC_ROLE_ARN}",
  "taskRoleArn": "${TASK_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "${APP_NAME}",
      "image": "${IMAGE_URI}",
      "portMappings": [ { "containerPort": 8000, "hostPort": 8000, "protocol": "tcp" } ],
      "essential": true,
      "environment": [
        {"name": "DEBUG", "value": "false"},
        {"name": "DEFAULT_MODEL", "value": "gpt-4o"},
        {"name": "AWS_REGION", "value": "${AWS_REGION}"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "${OPENAI_SECRET_ARN}"},
        {"name": "ANTHROPIC_API_KEY", "valueFrom": "${ANTHROPIC_SECRET_ARN}"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${APP_NAME}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "${APP_NAME}"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:8000/api/health/live || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 2,
        "startPeriod": 15
      }
    }
  ]
}
JSON

# Register task definition
aws ecs register-task-definition --cli-input-json file:///tmp/taskdef.json >/tmp/td.out
cat /tmp/td.out | jq -r '.taskDefinition.taskDefinitionArn' | tee /tmp/TASK_DEF_ARN
export TASK_DEF_ARN=$(cat /tmp/TASK_DEF_ARN)
```

---

## 7) Create ECS service

```bash
aws ecs create-service \
  --cluster $CLUSTER_NAME \
  --service-name ${APP_NAME}-${ENV} \
  --task-definition $TASK_DEF_ARN \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2,$SUBNET_3],securityGroups=[$SVC_SG_ID],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=${APP_NAME},containerPort=8000" \
  --health-check-grace-period-seconds 60

# Wait for service stability
aws ecs wait services-stable --cluster $CLUSTER_NAME --services ${APP_NAME}-${ENV}

# Show ALB DNS
aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN \
  --query 'LoadBalancers[0].DNSName' --output text
```

Visit: http://<ALB_DNS>/api/health and http://<ALB_DNS>/docs

### Service connectivity
Both services are now running in the same VPC and can communicate with each other:
- **paf-core-agent**: Available at your new ALB DNS
- **dev.vivid-brand.com**: Available at pac-alb-2089685514.us-east-2.elb.amazonaws.com
- **Internal communication**: Services can reach each other via private IP addresses within the VPC

## A2A Communication Setup

### How dev.vivid-brand.com can call paf-core-agent via A2A

The paf-core-agent implements the A2A (Agent-to-Agent) protocol and can be called by dev.vivid-brand.com in several ways:

#### 1. Direct HTTP API calls (Recommended)
Since both services are in the same VPC, dev.vivid-brand.com can make direct HTTP calls to paf-core-agent:

```javascript
// From dev.vivid-brand.com (Node.js service)
const pafCoreAgentUrl = 'http://<paf-core-agent-alb-dns>';

// Chat with paf-core-agent
const chatResponse = await fetch(`${pafCoreAgentUrl}/api/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Hello, can you help me analyze this data?",
    show_thinking: true,
    model: "gpt-4o"
  })
});

// Discover available A2A agents
const agentsResponse = await fetch(`${pafCoreAgentUrl}/api/chat/a2a/agents`);
const agents = await agentsResponse.json();

// Send A2A message via bridge
const a2aResponse = await fetch(`${pafCoreAgentUrl}/api/bridge/message`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message_type: "COORDINATION",
    target_agent_id: "paf-core-agent",
    payload: {
      task: "analyze_data",
      data: { /* your data */ }
    },
    priority: "NORMAL"
  })
});
```

#### 2. A2A Protocol Integration
The paf-core-agent exposes A2A endpoints that dev.vivid-brand.com can use:

**Available A2A Endpoints:**
- `GET /api/chat/a2a/agents` - Discover available agents
- `GET /api/chat/a2a/agents/{agent_id}` - Get specific agent details
- `GET /api/chat/a2a/status` - Check A2A server status
- `POST /api/bridge/message` - Send A2A messages
- `POST /api/bridge/task/request` - Send task requests
- `POST /api/bridge/status/broadcast` - Broadcast status updates

#### 3. Environment Configuration
Configure the paf-core-agent to be discoverable by dev.vivid-brand.com:

```bash
# In your paf-core-agent deployment, set these environment variables:
A2A_ENABLED=true
A2A_SERVER_URL=http://<paf-core-agent-alb-dns>
A2A_AGENT_CARD=paf-core-agent
A2A_AGENT_URL=http://<paf-core-agent-alb-dns>
```

#### 4. Service Discovery
Both services can discover each other:

```bash
# From dev.vivid-brand.com, discover paf-core-agent
curl http://<paf-core-agent-alb-dns>/api/chat/a2a/agents

# From paf-core-agent, discover dev.vivid-brand.com (if it implements A2A)
curl http://pac-alb-2089685514.us-east-2.elb.amazonaws.com/.well-known/agent.json
```

#### 5. Internal VPC Communication
Since both services are in the same VPC (vpc-0dc5816f0b041abad), they can communicate directly:

```javascript
// dev.vivid-brand.com can call paf-core-agent using internal ALB DNS
const internalPafUrl = 'http://<paf-core-agent-alb-dns>';

// Or use private IP if you know it
const privateIpUrl = 'http://<private-ip>:8000';
```

### Example Integration Scenarios

#### Scenario 1: Data Analysis Request
```javascript
// dev.vivid-brand.com sends data to paf-core-agent for analysis
const analysisRequest = {
  message: "Please analyze this customer data and provide insights",
  files: [/* file data */],
  show_thinking: true,
  model: "gpt-4o"
};

const response = await fetch(`${pafCoreAgentUrl}/api/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(analysisRequest)
});
```

#### Scenario 2: A2A Task Delegation
```javascript
// dev.vivid-brand.com delegates a task to paf-core-agent
const taskRequest = {
  message_type: "TASK_REQUEST",
  target_agent_id: "paf-core-agent",
  payload: {
    task_type: "data_processing",
    task_data: {
      operation: "summarize",
      content: "Large document content...",
      format: "executive_summary"
    }
  },
  priority: "HIGH"
};

const response = await fetch(`${pafCoreAgentUrl}/api/bridge/task/request`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(taskRequest)
});
```

#### Scenario 3: Health Monitoring
```javascript
// dev.vivid-brand.com monitors paf-core-agent health
const healthCheck = async () => {
  try {
    const response = await fetch(`${pafCoreAgentUrl}/api/health`);
    const health = await response.json();
    
    if (health.status === 'healthy') {
      console.log('paf-core-agent is healthy');
    } else {
      console.log('paf-core-agent is unhealthy:', health);
    }
  } catch (error) {
    console.error('Failed to check paf-core-agent health:', error);
  }
};
```

### Making dev.vivid-brand.com A2A Discoverable

To make dev.vivid-brand.com discoverable by paf-core-agent via A2A protocol, you need to add an agent card endpoint:

#### 1. Add Agent Card Endpoint to dev.vivid-brand.com
Add this endpoint to your Node.js service:

```javascript
// Add to your dev.vivid-brand.com Express app
app.get('/.well-known/agent.json', (req, res) => {
  const agentCard = {
    "name": "vivid-brand-dev",
    "description": "Vivid Brand Development Service",
    "version": "1.0.0",
    "url": "http://pac-alb-2089685514.us-east-2.elb.amazonaws.com",
    "capabilities": {
      "data_processing": true,
      "user_management": true,
      "brand_analytics": true
    },
    "skills": [
      {
        "id": "user_analytics",
        "name": "User Analytics",
        "description": "Analyze user behavior and brand interactions",
        "parameters": {
          "user_id": "string",
          "timeframe": "string"
        }
      },
      {
        "id": "brand_insights",
        "name": "Brand Insights",
        "description": "Generate brand performance insights",
        "parameters": {
          "brand_id": "string",
          "metrics": "array"
        }
      }
    ],
    "provider": {
      "name": "Vivid Brand",
      "contact": "engineering@pixell.global"
    },
    "authentication": {
      "type": "none" // or "api_key", "oauth", etc.
    }
  };
  
  res.json(agentCard);
});
```

#### 2. Configure paf-core-agent to discover dev.vivid-brand.com
Update your paf-core-agent environment variables:

```bash
# In your paf-core-agent deployment
A2A_ENABLED=true
A2A_SERVER_URL=http://pac-alb-2089685514.us-east-2.elb.amazonaws.com
A2A_AGENT_CARD=paf-core-agent
A2A_AGENT_URL=http://<paf-core-agent-alb-dns>
```

#### 3. Test A2A Discovery
After deployment, test the discovery:

```bash
# Test dev.vivid-brand.com agent card
curl http://pac-alb-2089685514.us-east-2.elb.amazonaws.com/.well-known/agent.json

# Test paf-core-agent discovery
curl http://<paf-core-agent-alb-dns>/api/chat/a2a/agents
```

#### 4. Bidirectional A2A Communication
Now both services can discover and communicate with each other:

```javascript
// From paf-core-agent, call dev.vivid-brand.com
const vividBrandUrl = 'http://pac-alb-2089685514.us-east-2.elb.amazonaws.com';

// From dev.vivid-brand.com, call paf-core-agent
const pafCoreUrl = 'http://<paf-core-agent-alb-dns>';

// Both services can now use A2A protocol for communication
```

---

## 8) Auto scaling (optional)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/${CLUSTER_NAME}/${APP_NAME}-${ENV} \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 --max-capacity 5

# Scale on CPU
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/${CLUSTER_NAME}/${APP_NAME}-${ENV} \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name ${APP_NAME}-${ENV}-cpu-50 \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 50.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"},
    "ScaleInCooldown": 60,
    "ScaleOutCooldown": 60
  }'

# Scale on Memory
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/${CLUSTER_NAME}/${APP_NAME}-${ENV} \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name ${APP_NAME}-${ENV}-mem-50 \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 50.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageMemoryUtilization"},
    "ScaleInCooldown": 60,
    "ScaleOutCooldown": 60
  }'
```

---

## 9) HTTPS (optional)
- Create an ACM certificate in the same region.
- Create an HTTPS listener (443) on the ALB using the certificate and forward to the same target group.

```bash
# Example: create HTTPS listener (replace CERT_ARN)
export CERT_ARN=<your-acm-certificate-arn>
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS --port 443 \
  --certificates CertificateArn=$CERT_ARN \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN
```

---

## 10) Troubleshooting
- Describe service events: `aws ecs describe-services --cluster $CLUSTER_NAME --services ${APP_NAME}-${ENV} | jq '.services[0].events[0:10]'`
- Describe tasks: `aws ecs list-tasks --cluster $CLUSTER_NAME --service-name ${APP_NAME}-${ENV}` then `aws ecs describe-tasks ...`
- Container logs: `aws logs tail /ecs/${APP_NAME} --follow --since 1h`
- Target health: `aws elbv2 describe-target-health --target-group-arn $TG_ARN`

---

## 11) Cleanup

**⚠️ Warning**: This will only delete resources created for `paf-core-agent`. The existing `dev.vivid-brand.com` service and its resources will remain untouched.

```bash
# Delete ECS service
aws ecs update-service --cluster $CLUSTER_NAME --service ${APP_NAME}-${ENV} --desired-count 0
aws ecs delete-service --cluster $CLUSTER_NAME --service ${APP_NAME}-${ENV} --force

# Delete cluster (only if no other services are using it)
aws ecs delete-cluster --cluster $CLUSTER_NAME

# Delete ALB + listener + target group (paf-core-agent specific)
aws elbv2 delete-listener --listener-arn $LISTENER_ARN
aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN
aws elbv2 delete-target-group --target-group-arn $TG_ARN

# Delete security groups (paf-core-agent specific)
aws ec2 delete-security-group --group-id $SVC_SG_ID
aws ec2 delete-security-group --group-id $ALB_SG_ID

# Optional: delete log group and secrets (paf-core-agent specific)
aws logs delete-log-group --log-group-name /ecs/${APP_NAME}
aws secretsmanager delete-secret --secret-id ${APP_NAME}/OPENAI_API_KEY --force-delete-without-recovery
aws secretsmanager delete-secret --secret-id ${APP_NAME}/ANTHROPIC_API_KEY --force-delete-without-recovery

# Optional: delete ECR repo (must be empty)
aws ecr delete-repository --repository-name $ECR_REPO --force

# Note: VPC, subnets, and existing dev.vivid-brand.com resources are preserved
```

---

### Notes
- If you deploy in private subnets, provision NAT gateways and disable `assignPublicIp`.
- Health endpoints available:
  - Liveness: `/api/health/live`
  - Readiness: `/api/health/ready`
  - Overall: `/api/health`
- CORS is permissive when `DEBUG=true`; for production, keep it disabled or set `cors_origins` explicitly in `app/settings.py`.


