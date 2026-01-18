"""
Platform Integration Module

Provides integration with external AI platforms including NVIDIA NIM,
Nemotron, and other services required for the RAG evaluation course.
"""

from .nvidia_integration import (
    NVIDIAPlatformIntegration,
    EmbeddingModel,
    LLMEndpoint,
    SyntheticDataModel,
    EmbeddingModelType,
    LLMModelType
)

from .nvidia_api_client import (
    NVIDIAAPIClient,
    APIResponse,
    APIErrorType,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError
)

__all__ = [
    "NVIDIAPlatformIntegration",
    "EmbeddingModel",
    "LLMEndpoint",
    "SyntheticDataModel",
    "EmbeddingModelType",
    "LLMModelType",
    "NVIDIAAPIClient",
    "APIResponse",
    "APIErrorType",
    "RateLimitError",
    "ServiceUnavailableError",
    "AuthenticationError",
    "InvalidRequestError"
]
