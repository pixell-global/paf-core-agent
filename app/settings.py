"""Application settings and configuration."""

from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # CORS configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        description="Allowed CORS origins"
    )
    
    # Authentication and Security
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production-32-chars-minimum",
        description="JWT secret key for token signing"
    )
    hmac_secret_key: str = Field(
        default="your-super-secret-hmac-key-for-api-keys-32-chars-minimum",
        description="HMAC secret key for API key hashing"
    )
    
    # Security settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    default_rate_limit_rpm: int = Field(default=60, description="Default requests per minute")
    default_rate_limit_burst: int = Field(default=10, description="Default burst limit")
    security_headers_enabled: bool = Field(default=True, description="Enable security headers")
    audit_logging_enabled: bool = Field(default=True, description="Enable audit logging")
    input_validation_enabled: bool = Field(default=True, description="Enable input validation middleware")
    
    # LLM Provider Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: Optional[str] = Field(default=None, description="OpenAI model name")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    
    # UPEE Configuration
    max_context_tokens: int = Field(default=50000, description="Maximum context tokens")  # Increased from 4000
    default_model: str = Field(default="gpt-4o", description="Default LLM model")
    show_thinking_default: bool = Field(default=False, description="Show thinking events by default")
    
    @property
    def resolved_default_model(self) -> str:
        """Get the resolved default model, preferring OPENAI_MODEL if set."""
        return self.openai_model or self.default_model
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=150, description="Max concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable AWS X-Ray tracing")
    
    # File processing settings
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes (100MB)")
    supported_file_types: List[str] = Field(
        default=[".py", ".js", ".ts", ".md", ".txt", ".json", ".csv", ".html", ".xml", ".yaml", ".yml"],
        description="Supported file types for processing"
    )
    max_chunk_size: int = Field(default=20000, description="Maximum chunk size for text processing")  # Increased from 4000
    chunk_overlap: int = Field(default=500, description="Overlap between chunks")  # Increased from 200
    
    # Text summarization settings
    max_summary_length: int = Field(default=2000, description="Maximum summary length in tokens")  # Increased from 500
    summary_model_preference: List[str] = Field(
        default=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
        description="Preferred models for summarization (fastest first)"
    )
    
    # gRPC Configuration
    worker_agent_endpoint: str = Field(
        default="localhost:50051", 
        description="Worker agent gRPC endpoint"
    )
    grpc_timeout: int = Field(default=10, description="gRPC timeout in seconds")
    grpc_enabled: bool = Field(default=True, description="Enable gRPC client functionality")
    grpc_health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    grpc_connection_timeout: float = Field(default=10.0, description="Connection timeout in seconds")
    grpc_retry_attempts: int = Field(default=3, description="Number of retry attempts for failed calls")


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings() 