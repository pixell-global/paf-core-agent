"""Standalone rate limiter for fine-grained control."""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from app.utils.logging_config import get_logger

logger = get_logger("rate_limiter")


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_minute: int
    burst_limit: int
    window_seconds: int = 60
    burst_window_seconds: int = 10


class RateLimiter:
    """Standalone rate limiter with flexible rules."""
    
    def __init__(self):
        # Client request tracking
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.client_last_request: Dict[str, float] = {}
        
        # Default rules
        self.default_rule = RateLimitRule(
            requests_per_minute=60,
            burst_limit=10
        )
        
        # Endpoint-specific rules
        self.endpoint_rules: Dict[str, RateLimitRule] = {}
        
        # Client-specific rules (for premium users, etc.)
        self.client_rules: Dict[str, RateLimitRule] = {}
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_requests())
    
    def add_endpoint_rule(self, endpoint: str, rule: RateLimitRule):
        """Add rate limit rule for a specific endpoint."""
        self.endpoint_rules[endpoint] = rule
        
        logger.info(
            "Added rate limit rule",
            endpoint=endpoint,
            rpm=rule.requests_per_minute,
            burst=rule.burst_limit
        )
    
    def add_client_rule(self, client_id: str, rule: RateLimitRule):
        """Add rate limit rule for a specific client."""
        self.client_rules[client_id] = rule
        
        logger.info(
            "Added client rate limit rule",
            client_id=client_id,
            rpm=rule.requests_per_minute,
            burst=rule.burst_limit
        )
    
    def _get_rule(self, client_id: str, endpoint: str) -> RateLimitRule:
        """Get the applicable rate limit rule."""
        
        # Check for client-specific rule first
        if client_id in self.client_rules:
            return self.client_rules[client_id]
        
        # Check for endpoint-specific rule
        for endpoint_pattern, rule in self.endpoint_rules.items():
            if endpoint.startswith(endpoint_pattern):
                return rule
        
        # Return default rule
        return self.default_rule
    
    async def check_rate_limit(
        self, 
        client_id: str, 
        endpoint: str
    ) -> Tuple[bool, Optional[int], Dict[str, any]]:
        """
        Check if request is within rate limits.
        
        Returns:
            (allowed, retry_after_seconds, rate_limit_info)
        """
        
        current_time = time.time()
        rule = self._get_rule(client_id, endpoint)
        
        # Get client's request history
        client_history = self.client_requests[client_id]
        
        # Remove requests older than the window
        cutoff_time = current_time - rule.window_seconds
        while client_history and client_history[0] < cutoff_time:
            client_history.popleft()
        
        # Check burst limit
        burst_cutoff = current_time - rule.burst_window_seconds
        recent_requests = sum(1 for req_time in client_history if req_time > burst_cutoff)
        
        if recent_requests >= rule.burst_limit:
            logger.warning(
                "Rate limit exceeded (burst)",
                client_id=client_id,
                endpoint=endpoint,
                recent_requests=recent_requests,
                burst_limit=rule.burst_limit
            )
            
            rate_limit_info = {
                "limit": rule.burst_limit,
                "remaining": 0,
                "reset_time": current_time + rule.burst_window_seconds,
                "retry_after": rule.burst_window_seconds,
                "limit_type": "burst"
            }
            
            return False, rule.burst_window_seconds, rate_limit_info
        
        # Check RPM limit
        if len(client_history) >= rule.requests_per_minute:
            logger.warning(
                "Rate limit exceeded (RPM)",
                client_id=client_id,
                endpoint=endpoint,
                requests_count=len(client_history),
                rpm_limit=rule.requests_per_minute
            )
            
            # Calculate retry after (time until oldest request expires)
            oldest_request = client_history[0]
            retry_after = int(oldest_request + rule.window_seconds - current_time + 1)
            
            rate_limit_info = {
                "limit": rule.requests_per_minute,
                "remaining": 0,
                "reset_time": current_time + rule.window_seconds,
                "retry_after": retry_after,
                "limit_type": "rpm"
            }
            
            return False, retry_after, rate_limit_info
        
        # Request is allowed - record it
        client_history.append(current_time)
        self.client_last_request[client_id] = current_time
        
        # Calculate remaining requests
        remaining_requests = max(0, rule.requests_per_minute - len(client_history))
        remaining_burst = max(0, rule.burst_limit - recent_requests - 1)  # -1 for current request
        
        rate_limit_info = {
            "limit": rule.requests_per_minute,
            "remaining": remaining_requests,
            "reset_time": current_time + rule.window_seconds,
            "retry_after": None,
            "burst_limit": rule.burst_limit,
            "burst_remaining": remaining_burst,
            "burst_reset_time": current_time + rule.burst_window_seconds
        }
        
        return True, None, rate_limit_info
    
    def get_client_stats(self, client_id: str) -> Dict[str, any]:
        """Get statistics for a specific client."""
        
        current_time = time.time()
        client_history = self.client_requests[client_id]
        
        if not client_history:
            return {
                "request_count": 0,
                "last_request": None,
                "requests_last_minute": 0,
                "requests_last_hour": 0
            }
        
        # Count requests in different time windows
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        requests_last_minute = sum(1 for req_time in client_history if req_time > minute_cutoff)
        requests_last_hour = sum(1 for req_time in client_history if req_time > hour_cutoff)
        
        return {
            "request_count": len(client_history),
            "last_request": self.client_last_request.get(client_id),
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour
        }
    
    def reset_client_limits(self, client_id: str):
        """Reset rate limits for a specific client."""
        
        if client_id in self.client_requests:
            del self.client_requests[client_id]
        
        if client_id in self.client_last_request:
            del self.client_last_request[client_id]
        
        logger.info("Reset rate limits", client_id=client_id)
    
    def get_all_client_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics for all clients."""
        
        stats = {}
        for client_id in self.client_requests.keys():
            stats[client_id] = self.get_client_stats(client_id)
        
        return stats
    
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
                    if client_id in self.client_requests:
                        del self.client_requests[client_id]
                    if client_id in self.client_last_request:
                        del self.client_last_request[client_id]
                
                if clients_to_remove:
                    logger.debug(f"Cleaned up rate limit data for {len(clients_to_remove)} clients")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")
    
    def shutdown(self):
        """Shutdown the rate limiter and cleanup."""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
        
        # Configure default endpoint rules
        _global_rate_limiter.add_endpoint_rule(
            "/api/chat/stream",
            RateLimitRule(requests_per_minute=20, burst_limit=5)
        )
        
        _global_rate_limiter.add_endpoint_rule(
            "/api/worker/execute",
            RateLimitRule(requests_per_minute=30, burst_limit=5)
        )
        
        _global_rate_limiter.add_endpoint_rule(
            "/api/files/process",
            RateLimitRule(requests_per_minute=40, burst_limit=8)
        )
        
        _global_rate_limiter.add_endpoint_rule(
            "/api/auth/login",
            RateLimitRule(requests_per_minute=10, burst_limit=3)
        )
        
        _global_rate_limiter.add_endpoint_rule(
            "/api/health",
            RateLimitRule(requests_per_minute=300, burst_limit=50)
        )
    
    return _global_rate_limiter 