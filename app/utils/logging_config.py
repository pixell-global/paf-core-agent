"""Structured logging configuration."""

import logging
import logging.config
import sys
from typing import Dict, Any

import structlog


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            # Add timestamp
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Add request context processor here later
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context to log record."""
        # This will be enhanced when we add request tracing
        return True


def log_request_start(method: str, path: str, request_id: str) -> None:
    """Log request start event."""
    logger = get_logger("request")
    logger.info(
        "Request started",
        method=method,
        path=path,
        request_id=request_id,
        event_type="request_start"
    )


def log_request_end(
    method: str, 
    path: str, 
    request_id: str, 
    status_code: int, 
    duration_ms: float
) -> None:
    """Log request completion event."""
    logger = get_logger("request")
    logger.info(
        "Request completed",
        method=method,
        path=path,
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        event_type="request_end"
    )


def log_upee_phase(
    phase: str, 
    request_id: str, 
    content: str, 
    metadata: Dict[str, Any] = None
) -> None:
    """Log UPEE phase event."""
    logger = get_logger("upee")
    logger.info(
        f"UPEE {phase} phase",
        phase=phase,
        request_id=request_id,
        content=content,
        metadata=metadata or {},
        event_type="upee_phase"
    )


def log_llm_call(
    provider: str,
    model: str,
    request_id: str,
    tokens_used: int = None,
    duration_ms: float = None,
    error: str = None
) -> None:
    """Log LLM API call event."""
    logger = get_logger("llm")
    
    if error:
        logger.error(
            "LLM call failed",
            provider=provider,
            model=model,
            request_id=request_id,
            error=error,
            event_type="llm_error"
        )
    else:
        logger.info(
            "LLM call completed",
            provider=provider,
            model=model,
            request_id=request_id,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            event_type="llm_success"
        ) 