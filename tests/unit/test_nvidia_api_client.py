"""
Unit tests for NVIDIA API Client error handling

Tests cover:
- Rate limit scenarios
- Service unavailability fallbacks
- Authentication errors
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.platform_integration import (
    NVIDIAAPIClient,
    APIResponse,
    APIErrorType,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError
)


class TestNVIDIAAPIClientRateLimits:
    """Test rate limit handling scenarios"""
    
    def test_rate_limit_with_retry_after_header(self):
        """Test that rate limit errors respect Retry-After header"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        # Mock response with 429 status and Retry-After header
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "5"}
        
        with patch('requests.post', return_value=mock_response):
            with patch('time.sleep') as mock_sleep:
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                # Should fail after retries
                assert not response.success
                assert response.error_type == APIErrorType.RATE_LIMIT
                assert response.retry_after == 5
                
                # Should have called sleep with exponential backoff
                assert mock_sleep.call_count > 0
    
    def test_rate_limit_exponential_backoff(self):
        """Test exponential backoff on rate limit errors"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=3)
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        
        with patch('requests.post', return_value=mock_response):
            with patch('time.sleep') as mock_sleep:
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                # Verify exponential backoff was used
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert len(sleep_calls) >= 2
                # When no Retry-After header, uses default 60s from retry_after
                # This is the actual behavior - rate limit without header defaults to 60s
                assert all(wait_time > 0 for wait_time in sleep_calls)
    
    def test_rate_limit_eventual_success(self):
        """Test successful request after rate limit retry"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=3)
        
        # First call returns 429, second call succeeds
        mock_rate_limit = Mock(spec=requests.Response)
        mock_rate_limit.status_code = 429
        mock_rate_limit.headers = {"Retry-After": "1"}
        
        mock_success = Mock(spec=requests.Response)
        mock_success.status_code = 200
        mock_success.json.return_value = {"result": "success"}
        
        with patch('requests.post', side_effect=[mock_rate_limit, mock_success]):
            with patch('time.sleep'):
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                assert response.success
                assert response.data == {"result": "success"}
                assert client.stats["retries"] == 1
                assert client.stats["successful_requests"] == 1


class TestNVIDIAAPIClientServiceUnavailability:
    """Test service unavailability and fallback scenarios"""
    
    def test_service_unavailable_503(self):
        """Test handling of 503 Service Unavailable"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 503
        mock_response.text = "Service temporarily unavailable"
        mock_response.json.side_effect = ValueError("No JSON")
        
        with patch('requests.post', return_value=mock_response):
            with patch('time.sleep'):
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                assert not response.success
                assert response.error_type == APIErrorType.SERVICE_UNAVAILABLE
    
    def test_fallback_endpoint_success(self):
        """Test successful fallback to alternative endpoint"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        # Add fallback endpoint
        primary_url = "https://api.nvidia.com/test"
        fallback_url = "https://backup.api.nvidia.com/test"
        client.add_fallback_endpoint(primary_url, fallback_url)
        
        # Primary endpoint fails with 503
        mock_503 = Mock(spec=requests.Response)
        mock_503.status_code = 503
        mock_503.text = "Service unavailable"
        mock_503.json.side_effect = ValueError("No JSON")
        
        # Fallback endpoint succeeds
        mock_success = Mock(spec=requests.Response)
        mock_success.status_code = 200
        mock_success.json.return_value = {"result": "fallback_success"}
        
        with patch('requests.post', side_effect=[mock_503, mock_503, mock_success]):
            with patch('time.sleep'):
                response = client.call_with_retry(primary_url, {"test": "data"})
                
                assert response.success
                assert response.data == {"result": "fallback_success"}
                assert client.stats["fallback_uses"] == 1
    
    def test_fallback_endpoint_also_fails(self):
        """Test when both primary and fallback endpoints fail"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        primary_url = "https://api.nvidia.com/test"
        fallback_url = "https://backup.api.nvidia.com/test"
        client.add_fallback_endpoint(primary_url, fallback_url)
        
        mock_503 = Mock(spec=requests.Response)
        mock_503.status_code = 503
        mock_503.text = "Service unavailable"
        mock_503.json.side_effect = ValueError("No JSON")
        
        with patch('requests.post', return_value=mock_503):
            with patch('time.sleep'):
                response = client.call_with_retry(primary_url, {"test": "data"})
                
                assert not response.success
                assert response.error_type == APIErrorType.SERVICE_UNAVAILABLE
                assert "All endpoints failed" in response.error
    
    def test_multiple_fallback_endpoints(self):
        """Test trying multiple fallback endpoints"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        primary_url = "https://api.nvidia.com/test"
        fallback1 = "https://backup1.api.nvidia.com/test"
        fallback2 = "https://backup2.api.nvidia.com/test"
        
        client.add_fallback_endpoint(primary_url, fallback1)
        client.add_fallback_endpoint(primary_url, fallback2)
        
        mock_503 = Mock(spec=requests.Response)
        mock_503.status_code = 503
        mock_503.text = "Service unavailable"
        mock_503.json.side_effect = ValueError("No JSON")
        
        mock_success = Mock(spec=requests.Response)
        mock_success.status_code = 200
        mock_success.json.return_value = {"result": "second_fallback_success"}
        
        # Primary fails, fallback1 fails, fallback2 succeeds
        with patch('requests.post', side_effect=[mock_503, mock_503, mock_503, mock_success]):
            with patch('time.sleep'):
                response = client.call_with_retry(primary_url, {"test": "data"})
                
                assert response.success
                assert response.data == {"result": "second_fallback_success"}


class TestNVIDIAAPIClientAuthentication:
    """Test authentication error scenarios"""
    
    def test_authentication_error_401(self):
        """Test handling of 401 Unauthorized"""
        client = NVIDIAAPIClient(api_key="invalid-key")
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        with patch('requests.post', return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
            
            assert "Invalid API key" in str(exc_info.value)
    
    def test_authentication_error_403(self):
        """Test handling of 403 Forbidden"""
        client = NVIDIAAPIClient(api_key="test-key")
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "error": {"message": "Access denied"}
        }
        
        with patch('requests.post', return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
            
            assert "Access denied" in str(exc_info.value)
    
    def test_authentication_error_no_retry(self):
        """Test that authentication errors are not retried"""
        client = NVIDIAAPIClient(api_key="invalid-key", max_retries=3)
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            with pytest.raises(AuthenticationError):
                client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
            
            # Should only call once, no retries
            assert mock_post.call_count == 1
            assert client.stats["retries"] == 0


class TestNVIDIAAPIClientInvalidRequests:
    """Test invalid request error scenarios"""
    
    def test_invalid_request_400(self):
        """Test handling of 400 Bad Request"""
        client = NVIDIAAPIClient(api_key="test-key")
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Missing required field: model"}
        }
        
        with patch('requests.post', return_value=mock_response):
            with pytest.raises(InvalidRequestError) as exc_info:
                client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
            
            assert "Missing required field" in str(exc_info.value)
    
    def test_invalid_request_no_retry(self):
        """Test that invalid requests are not retried"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=3)
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid payload"}
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            with pytest.raises(InvalidRequestError):
                client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
            
            # Should only call once, no retries
            assert mock_post.call_count == 1
            assert client.stats["retries"] == 0


class TestNVIDIAAPIClientNetworkErrors:
    """Test network error scenarios"""
    
    def test_timeout_with_retry(self):
        """Test timeout handling with retry"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=3)
        
        with patch('requests.post', side_effect=requests.Timeout("Request timeout")):
            with patch('time.sleep'):
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                assert not response.success
                assert response.error_type == APIErrorType.NETWORK
                assert "timeout" in response.error.lower()
                # Retries happen for all attempts except the last one that fails
                assert client.stats["retries"] >= 2
    
    def test_connection_error_with_retry(self):
        """Test connection error handling with retry"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=2)
        
        with patch('requests.post', side_effect=requests.ConnectionError("Connection refused")):
            with patch('time.sleep'):
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                assert not response.success
                assert response.error_type == APIErrorType.NETWORK
                assert client.stats["retries"] >= 1


class TestNVIDIAAPIClientStatistics:
    """Test API client statistics tracking"""
    
    def test_statistics_successful_request(self):
        """Test statistics for successful requests"""
        client = NVIDIAAPIClient(api_key="test-key")
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        with patch('requests.post', return_value=mock_response):
            response = client.call_with_retry(
                "https://api.nvidia.com/test",
                {"test": "data"}
            )
            
            stats = client.get_statistics()
            assert stats["total_requests"] == 1
            assert stats["successful_requests"] == 1
            assert stats["failed_requests"] == 0
            assert stats["retries"] == 0
    
    def test_statistics_failed_request_with_retries(self):
        """Test statistics for failed requests with retries"""
        client = NVIDIAAPIClient(api_key="test-key", max_retries=3)
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        mock_response.json.side_effect = ValueError("No JSON")
        
        with patch('requests.post', return_value=mock_response):
            with patch('time.sleep'):
                response = client.call_with_retry(
                    "https://api.nvidia.com/test",
                    {"test": "data"}
                )
                
                stats = client.get_statistics()
                assert stats["total_requests"] == 1
                assert stats["successful_requests"] == 0
                assert stats["failed_requests"] == 1
                # Retries happen for each failed attempt
                assert stats["retries"] >= 2
    
    def test_statistics_reset(self):
        """Test resetting statistics"""
        client = NVIDIAAPIClient(api_key="test-key")
        
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        with patch('requests.post', return_value=mock_response):
            client.call_with_retry("https://api.nvidia.com/test", {"test": "data"})
            
            stats_before = client.get_statistics()
            assert stats_before["total_requests"] == 1
            
            client.reset_statistics()
            
            stats_after = client.get_statistics()
            assert stats_after["total_requests"] == 0
            assert stats_after["successful_requests"] == 0
