"""
NVIDIA API Client with Retry Logic and Error Handling

This module provides a robust API client for interacting with NVIDIA services
with built-in retry logic, rate limit handling, and fallback endpoint support.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


# Configure logging
logger = logging.getLogger(__name__)


class APIErrorType(Enum):
    """Types of API errors"""
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class APIResponse:
    """Represents an API response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[APIErrorType] = None
    status_code: Optional[int] = None
    retry_after: Optional[int] = None
    
    def __repr__(self) -> str:
        if self.success:
            return f"APIResponse(success=True, status={self.status_code})"
        return f"APIResponse(success=False, error={self.error_type})"


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServiceUnavailableError(Exception):
    """Raised when API service is unavailable"""
    pass


class AuthenticationError(Exception):
    """Raised when API authentication fails"""
    pass


class InvalidRequestError(Exception):
    """Raised when API request is invalid"""
    pass


class NVIDIAAPIClient:
    """
    Robust API client for NVIDIA services with retry logic and error handling.
    
    Features:
    - Exponential backoff for rate limits
    - Automatic retry on transient failures
    - Fallback endpoint support
    - Comprehensive error handling
    - Request/response logging
    """
    
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        base_timeout: int = 30,
        enable_logging: bool = True
    ):
        """
        Initialize NVIDIA API client.
        
        Args:
            api_key: NVIDIA API key for authentication
            max_retries: Maximum number of retry attempts
            base_timeout: Base timeout for API requests in seconds
            enable_logging: Enable request/response logging
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.enable_logging = enable_logging
        
        # Fallback endpoints for high availability
        self.fallback_endpoints: Dict[str, List[str]] = {}
        
        # Request statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries": 0,
            "fallback_uses": 0
        }
    
    def call_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> APIResponse:
        """
        Call API endpoint with automatic retry logic.
        
        Implements exponential backoff for rate limits and retries transient failures.
        
        Args:
            endpoint: API endpoint URL
            payload: Request payload
            method: HTTP method (GET, POST, etc.)
            headers: Optional custom headers
            timeout: Optional custom timeout
            
        Returns:
            APIResponse with success status and data or error information
            
        Raises:
            AuthenticationError: If authentication fails
            InvalidRequestError: If request is malformed
            
        Example:
            >>> client = NVIDIAAPIClient(api_key="your-key")
            >>> response = client.call_with_retry(
            ...     "https://api.nvidia.com/v1/embeddings",
            ...     {"input": "test text", "model": "nv-embed-v2"}
            ... )
            >>> if response.success:
            ...     print(response.data)
        """
        self.stats["total_requests"] += 1
        
        # Prepare headers
        request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        request_timeout = timeout or self.base_timeout
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.enable_logging:
                    logger.info(f"API request attempt {attempt + 1}/{self.max_retries}: {method} {endpoint}")
                
                # Make API request
                if method.upper() == "POST":
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=request_headers,
                        timeout=request_timeout
                    )
                elif method.upper() == "GET":
                    response = requests.get(
                        endpoint,
                        params=payload,
                        headers=request_headers,
                        timeout=request_timeout
                    )
                else:
                    raise InvalidRequestError(f"Unsupported HTTP method: {method}")
                
                # Handle response
                return self._handle_response(response)
                
            except RateLimitError as e:
                last_error = e
                self.stats["retries"] += 1
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = e.retry_after or (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit. Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded due to rate limiting")
                    self.stats["failed_requests"] += 1
                    return APIResponse(
                        success=False,
                        error="Rate limit exceeded. Max retries reached.",
                        error_type=APIErrorType.RATE_LIMIT,
                        retry_after=e.retry_after
                    )
            
            except ServiceUnavailableError as e:
                last_error = e
                self.stats["retries"] += 1
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Service unavailable. Retrying... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    # Try fallback endpoint
                    logger.warning("Max retries exceeded. Attempting fallback endpoint...")
                    fallback_response = self._try_fallback_endpoint(endpoint, payload, method, request_headers, request_timeout)
                    if fallback_response:
                        return fallback_response
                    
                    self.stats["failed_requests"] += 1
                    return APIResponse(
                        success=False,
                        error="Service unavailable. All endpoints failed.",
                        error_type=APIErrorType.SERVICE_UNAVAILABLE
                    )
            
            except (Timeout, ConnectionError) as e:
                last_error = e
                self.stats["retries"] += 1
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Network error: {e}. Retrying... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    self.stats["failed_requests"] += 1
                    return APIResponse(
                        success=False,
                        error=f"Network error: {str(e)}",
                        error_type=APIErrorType.NETWORK
                    )
            
            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"Authentication failed: {e}")
                self.stats["failed_requests"] += 1
                raise
            
            except InvalidRequestError as e:
                # Don't retry invalid requests
                logger.error(f"Invalid request: {e}")
                self.stats["failed_requests"] += 1
                raise
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.stats["failed_requests"] += 1
                    return APIResponse(
                        success=False,
                        error=f"Unexpected error: {str(e)}",
                        error_type=APIErrorType.UNKNOWN
                    )
        
        # Should not reach here, but handle gracefully
        self.stats["failed_requests"] += 1
        return APIResponse(
            success=False,
            error=f"Request failed after {self.max_retries} attempts: {str(last_error)}",
            error_type=APIErrorType.UNKNOWN
        )
    
    def _handle_response(self, response: requests.Response) -> APIResponse:
        """
        Handle API response and convert to APIResponse object.
        
        Args:
            response: requests.Response object
            
        Returns:
            APIResponse with parsed data or error information
            
        Raises:
            RateLimitError: If rate limit is exceeded
            ServiceUnavailableError: If service is unavailable
            AuthenticationError: If authentication fails
            InvalidRequestError: If request is invalid
        """
        status_code = response.status_code
        
        # Success
        if 200 <= status_code < 300:
            self.stats["successful_requests"] += 1
            try:
                data = response.json()
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=status_code
                )
            except ValueError:
                return APIResponse(
                    success=True,
                    data={"raw": response.text},
                    status_code=status_code
                )
        
        # Rate limit (429)
        if status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s",
                retry_after=retry_after
            )
        
        # Authentication errors (401, 403)
        if status_code in [401, 403]:
            error_msg = self._extract_error_message(response)
            raise AuthenticationError(f"Authentication failed: {error_msg}")
        
        # Invalid request (400)
        if status_code == 400:
            error_msg = self._extract_error_message(response)
            raise InvalidRequestError(f"Invalid request: {error_msg}")
        
        # Service unavailable (500, 502, 503, 504)
        if status_code in [500, 502, 503, 504]:
            error_msg = self._extract_error_message(response)
            raise ServiceUnavailableError(f"Service unavailable: {error_msg}")
        
        # Other errors
        error_msg = self._extract_error_message(response)
        return APIResponse(
            success=False,
            error=f"API error ({status_code}): {error_msg}",
            error_type=APIErrorType.UNKNOWN,
            status_code=status_code
        )
    
    def _extract_error_message(self, response: requests.Response) -> str:
        """Extract error message from response"""
        try:
            error_data = response.json()
            return error_data.get("error", {}).get("message", response.text)
        except ValueError:
            return response.text or f"HTTP {response.status_code}"
    
    def _try_fallback_endpoint(
        self,
        original_endpoint: str,
        payload: Dict[str, Any],
        method: str,
        headers: Dict[str, str],
        timeout: int
    ) -> Optional[APIResponse]:
        """
        Try fallback endpoints if primary endpoint fails.
        
        Args:
            original_endpoint: Original endpoint that failed
            payload: Request payload
            method: HTTP method
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            APIResponse if fallback succeeds, None otherwise
        """
        fallback_urls = self.fallback_endpoints.get(original_endpoint, [])
        
        for fallback_url in fallback_urls:
            try:
                logger.info(f"Trying fallback endpoint: {fallback_url}")
                self.stats["fallback_uses"] += 1
                
                if method.upper() == "POST":
                    response = requests.post(
                        fallback_url,
                        json=payload,
                        headers=headers,
                        timeout=timeout
                    )
                else:
                    response = requests.get(
                        fallback_url,
                        params=payload,
                        headers=headers,
                        timeout=timeout
                    )
                
                result = self._handle_response(response)
                if result.success:
                    logger.info(f"Fallback endpoint succeeded: {fallback_url}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback endpoint failed: {fallback_url} - {e}")
                continue
        
        return None
    
    def add_fallback_endpoint(self, primary_endpoint: str, fallback_url: str) -> None:
        """
        Add a fallback endpoint for a primary endpoint.
        
        Args:
            primary_endpoint: Primary endpoint URL
            fallback_url: Fallback endpoint URL
            
        Example:
            >>> client = NVIDIAAPIClient(api_key="your-key")
            >>> client.add_fallback_endpoint(
            ...     "https://api.nvidia.com/v1/embeddings",
            ...     "https://backup.api.nvidia.com/v1/embeddings"
            ... )
        """
        if primary_endpoint not in self.fallback_endpoints:
            self.fallback_endpoints[primary_endpoint] = []
        
        if fallback_url not in self.fallback_endpoints[primary_endpoint]:
            self.fallback_endpoints[primary_endpoint].append(fallback_url)
            logger.info(f"Added fallback endpoint for {primary_endpoint}: {fallback_url}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get API client statistics.
        
        Returns:
            Dictionary with request statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset API client statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries": 0,
            "fallback_uses": 0
        }
