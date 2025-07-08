"""Security middleware for FastAPI application."""

import time
import json
import asyncio
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.auth.models import AuditLog, SecurityEvent
from app.utils.logging_config import get_logger

logger = get_logger("security_middleware")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
        
        # Add server info header
        response.headers["X-Powered-By"] = "PAF-Core-Agent"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with different limits for different endpoints."""
    
    def __init__(
        self,
        app,
        default_requests_per_minute: int = 60,
        default_burst_limit: int = 10,
        **kwargs
    ):
        super().__init__(app)
        self.default_rpm = default_requests_per_minute
        self.default_burst = default_burst_limit
        
        # Per-client request tracking
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.client_last_request: Dict[str, float] = {}
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/chat/stream": {"rpm": 20, "burst": 5},  # Chat is expensive
            "/api/worker/execute": {"rpm": 30, "burst": 5},  # Worker tasks are expensive
            "/api/files/process": {"rpm": 40, "burst": 8},  # File processing
            "/api/auth/login": {"rpm": 10, "burst": 3},  # Login attempts
            "/api/health": {"rpm": 300, "burst": 50},  # Health checks are cheap
        }
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_requests())
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        
        # Try to get authenticated user ID first
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fall back to IP address
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host
        
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_limits_for_path(self, path: str) -> tuple[int, int]:
        """Get rate limits for a specific path."""
        
        # Check for exact match
        if path in self.endpoint_limits:
            limits = self.endpoint_limits[path]
            return limits["rpm"], limits["burst"]
        
        # Check for prefix matches
        for endpoint_path, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_path):
                return limits["rpm"], limits["burst"]
        
        return self.default_rpm, self.default_burst
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        
        current_time = time.time()
        client_id = self._get_client_id(request)
        path = request.url.path
        
        # Get limits for this endpoint
        rpm_limit, burst_limit = self._get_limits_for_path(path)
        
        # Get client's request history
        client_history = self.client_requests[client_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while client_history and client_history[0] < cutoff_time:
            client_history.popleft()
        
        # Check burst limit (requests in last 10 seconds)
        burst_cutoff = current_time - 10
        recent_requests = sum(1 for req_time in client_history if req_time > burst_cutoff)
        
        if recent_requests >= burst_limit:
            logger.warning(
                "Rate limit exceeded (burst)",
                client_id=client_id,
                path=path,
                recent_requests=recent_requests,
                burst_limit=burst_limit
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {burst_limit} requests per 10 seconds",
                    "retry_after": 10
                },
                headers={"Retry-After": "10"}
            )
        
        # Check RPM limit
        if len(client_history) >= rpm_limit:
            logger.warning(
                "Rate limit exceeded (RPM)",
                client_id=client_id,
                path=path,
                requests_count=len(client_history),
                rpm_limit=rpm_limit
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {rpm_limit} requests per minute",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        client_history.append(current_time)
        self.client_last_request[client_id] = current_time
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        remaining_requests = max(0, rpm_limit - len(client_history))
        reset_time = int(current_time + 60)
        
        response.headers["X-RateLimit-Limit"] = str(rpm_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining_requests)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    async def _cleanup_old_requests(self):
        """Periodically clean up old request data."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                current_time = time.time()
                cutoff_time = current_time - 3600  # Keep data for 1 hour
                
                # Clean up old client data
                clients_to_remove = []
                for client_id, last_request in self.client_last_request.items():
                    if last_request < cutoff_time:
                        clients_to_remove.append(client_id)
                
                for client_id in clients_to_remove:
                    del self.client_requests[client_id]
                    del self.client_last_request[client_id]
                
                if clients_to_remove:
                    logger.debug(f"Cleaned up rate limit data for {len(clients_to_remove)} clients")
                
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging of API requests."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Paths to audit (sensitive operations)
        self.audit_paths = {
            "/api/auth/login",
            "/api/auth/logout", 
            "/api/users",
            "/api/api-keys",
            "/api/worker/execute",
            "/api/files/process"
        }
        
        # Exclude health checks and status endpoints from audit
        self.exclude_paths = {
            "/api/health",
            "/api/status",
            "/docs",
            "/openapi.json"
        }
    
    def _should_audit(self, path: str, method: str) -> bool:
        """Determine if this request should be audited."""
        
        # Skip excluded paths
        if any(path.startswith(exclude) for exclude in self.exclude_paths):
            return False
        
        # Audit sensitive paths
        if any(path.startswith(audit) for audit in self.audit_paths):
            return True
        
        # Audit all POST, PUT, DELETE operations
        if method in ["POST", "PUT", "DELETE"]:
            return True
        
        return False
    
    def _get_user_info(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        """Extract user information from request."""
        
        # Try to get user from request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return request.state.user.id, request.state.user.username
        
        return None, None
    
    def _safe_json_body(self, body: bytes) -> str:
        """Safely extract JSON body for logging (excluding sensitive fields)."""
        
        try:
            if not body:
                return ""
            
            data = json.loads(body.decode('utf-8'))
            
            # Remove sensitive fields
            sensitive_fields = {"password", "api_key", "secret", "token"}
            
            def remove_sensitive(obj):
                if isinstance(obj, dict):
                    return {
                        k: "[REDACTED]" if k.lower() in sensitive_fields 
                        else remove_sensitive(v)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [remove_sensitive(item) for item in obj]
                else:
                    return obj
            
            sanitized = remove_sensitive(data)
            return json.dumps(sanitized)[:500]  # Limit size
            
        except Exception:
            return "[INVALID_JSON]"
    
    async def dispatch(self, request: Request, call_next):
        """Log audit information for requests."""
        
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        # Check if we should audit this request
        should_audit = self._should_audit(path, method)
        
        if not should_audit:
            return await call_next(request)
        
        # Get request information
        user_id, username = self._get_user_info(request)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get request body for POST/PUT operations
        request_body = ""
        if method in ["POST", "PUT"] and hasattr(request, "_body"):
            request_body = self._safe_json_body(request._body)
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Determine if operation was successful
        success = 200 <= response.status_code < 400
        
        # Create audit log entry
        audit_entry = {
            "user_id": user_id,
            "username": username,
            "action": f"{method} {path}",
            "resource": path,
            "ip_address": client_ip,
            "user_agent": user_agent,
            "success": success,
            "status_code": response.status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "request_body": request_body,
            "timestamp": time.time()
        }
        
        # Add error information if request failed
        if not success:
            audit_entry["error"] = f"HTTP {response.status_code}"
        
        # Log audit entry
        logger.info(
            "Audit log",
            **audit_entry
        )
        
        # For critical operations, also log as security event
        if path.startswith("/api/auth") or path.startswith("/api/users"):
            severity = "high" if not success else "medium"
            
            security_event = {
                "event_type": "authentication" if "auth" in path else "user_management",
                "severity": severity,
                "user_id": user_id,
                "ip_address": client_ip,
                "description": f"{method} {path} - {'Success' if success else 'Failed'}",
                "metadata": audit_entry
            }
            
            logger.info(
                "Security event",
                **security_event
            )
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for additional input validation and sanitization."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Maximum request sizes by endpoint
        self.size_limits = {
            "/api/chat/stream": 1024 * 1024,  # 1MB for chat
            "/api/files/process": 10 * 1024 * 1024,  # 10MB for file processing
            "/api/worker/execute": 5 * 1024 * 1024,  # 5MB for worker tasks
        }
        
        self.default_size_limit = 1024 * 1024  # 1MB default
    
    def _get_size_limit(self, path: str) -> int:
        """Get size limit for a specific path."""
        
        for endpoint_path, limit in self.size_limits.items():
            if path.startswith(endpoint_path):
                return limit
        
        return self.default_size_limit
    
    async def dispatch(self, request: Request, call_next):
        """Validate and sanitize input."""
        
        path = request.url.path
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                size_limit = self._get_size_limit(path)
                
                if size > size_limit:
                    logger.warning(
                        "Request size exceeded limit",
                        path=path,
                        size=size,
                        limit=size_limit
                    )
                    
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "error": "Request too large",
                            "detail": f"Request size {size} exceeds limit {size_limit}",
                            "max_size": size_limit
                        }
                    )
            except ValueError:
                pass
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT"]:
            content_type = request.headers.get("content-type", "")
            
            # For API endpoints, require JSON content type
            if path.startswith("/api/") and not content_type.startswith("application/json"):
                # Allow form data for auth endpoints
                if not (path.startswith("/api/auth/") and "form" in content_type):
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "error": "Unsupported media type",
                            "detail": "Expected application/json",
                            "received": content_type
                        }
                    )
        
        return await call_next(request) 