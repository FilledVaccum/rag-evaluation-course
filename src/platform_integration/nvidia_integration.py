"""
NVIDIA Platform Integration Module

This module provides integration with NVIDIA's AI platform services including:
- NVIDIA NIM for embeddings and LLM inference
- Nemotron-4-340B for synthetic data generation
- API key management and endpoint configuration
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import os


class EmbeddingModelType(Enum):
    """Supported NVIDIA NIM embedding models"""
    NV_EMBED_V2 = "nvidia/nv-embed-v2"
    NV_EMBED_QA = "nvidia/nv-embed-qa"
    CODE_EMBED = "nvidia/code-embed"
    FINANCE_EMBED = "nvidia/finance-embed"
    HEALTHCARE_EMBED = "nvidia/healthcare-embed"
    MULTILINGUAL_EMBED = "nvidia/multilingual-embed"


class LLMModelType(Enum):
    """Supported NVIDIA NIM LLM models"""
    LLAMA_3_70B = "meta/llama-3-70b-instruct"
    LLAMA_3_8B = "meta/llama-3-8b-instruct"
    MISTRAL_7B = "mistralai/mistral-7b-instruct"
    MIXTRAL_8X7B = "mistralai/mixtral-8x7b-instruct"
    NEMOTRON_340B = "nvidia/nemotron-4-340b-instruct"


@dataclass
class EmbeddingModel:
    """Represents an NVIDIA NIM embedding model"""
    model_name: str
    model_type: EmbeddingModelType
    endpoint: str
    dimension: int
    max_tokens: int
    
    def __repr__(self) -> str:
        return f"EmbeddingModel({self.model_type.value}, dim={self.dimension})"


@dataclass
class LLMEndpoint:
    """Represents an NVIDIA NIM LLM inference endpoint"""
    model_name: str
    model_type: LLMModelType
    endpoint: str
    max_tokens: int
    supports_streaming: bool
    
    def __repr__(self) -> str:
        return f"LLMEndpoint({self.model_type.value})"


@dataclass
class SyntheticDataModel:
    """Represents Nemotron-4-340B model for synthetic data generation"""
    model_name: str
    endpoint: str
    max_tokens: int
    temperature: float
    top_p: float
    
    def __repr__(self) -> str:
        return f"SyntheticDataModel(nemotron-4-340b)"


class NVIDIAPlatformIntegration:
    """
    Main integration class for NVIDIA AI platform services.
    
    Provides methods to:
    - Get embedding models for various domains
    - Get LLM endpoints for inference
    - Get Nemotron synthesizer for synthetic data generation
    - Manage API keys and endpoint configuration
    """
    
    # Default NVIDIA NIM endpoints
    DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_NEMOTRON_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        nim_base_url: Optional[str] = None,
        custom_endpoints: Optional[Dict[str, str]] = None
    ):
        """
        Initialize NVIDIA platform integration.
        
        Args:
            api_key: NVIDIA API key. If None, reads from NVIDIA_API_KEY env var
            nim_base_url: Base URL for NVIDIA NIM services
            custom_endpoints: Custom endpoint URLs for specific models
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.nim_base_url = nim_base_url or self.DEFAULT_NIM_BASE_URL
        self.custom_endpoints = custom_endpoints or {}
        
        # Cache for model configurations
        self._embedding_models_cache: Dict[str, EmbeddingModel] = {}
        self._llm_endpoints_cache: Dict[str, LLMEndpoint] = {}
    
    def get_embedding_model(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.NV_EMBED_V2,
        custom_endpoint: Optional[str] = None
    ) -> EmbeddingModel:
        """
        Get an NVIDIA NIM embedding model.
        
        Args:
            model_type: Type of embedding model to retrieve
            custom_endpoint: Optional custom endpoint URL
            
        Returns:
            EmbeddingModel instance configured for the specified model
            
        Example:
            >>> integration = NVIDIAPlatformIntegration(api_key="your-key")
            >>> model = integration.get_embedding_model(EmbeddingModelType.NV_EMBED_V2)
            >>> print(model.dimension)  # 1024
        """
        cache_key = f"{model_type.value}:{custom_endpoint}"
        
        if cache_key in self._embedding_models_cache:
            return self._embedding_models_cache[cache_key]
        
        # Model configurations
        model_configs = {
            EmbeddingModelType.NV_EMBED_V2: {
                "dimension": 1024,
                "max_tokens": 512,
                "endpoint": f"{self.nim_base_url}/embeddings"
            },
            EmbeddingModelType.NV_EMBED_QA: {
                "dimension": 768,
                "max_tokens": 512,
                "endpoint": f"{self.nim_base_url}/embeddings"
            },
            EmbeddingModelType.CODE_EMBED: {
                "dimension": 768,
                "max_tokens": 1024,
                "endpoint": f"{self.nim_base_url}/embeddings"
            },
            EmbeddingModelType.FINANCE_EMBED: {
                "dimension": 768,
                "max_tokens": 512,
                "endpoint": f"{self.nim_base_url}/embeddings"
            },
            EmbeddingModelType.HEALTHCARE_EMBED: {
                "dimension": 768,
                "max_tokens": 512,
                "endpoint": f"{self.nim_base_url}/embeddings"
            },
            EmbeddingModelType.MULTILINGUAL_EMBED: {
                "dimension": 1024,
                "max_tokens": 512,
                "endpoint": f"{self.nim_base_url}/embeddings"
            }
        }
        
        config = model_configs[model_type]
        endpoint = custom_endpoint or self.custom_endpoints.get(model_type.value) or config["endpoint"]
        
        model = EmbeddingModel(
            model_name=model_type.value,
            model_type=model_type,
            endpoint=endpoint,
            dimension=config["dimension"],
            max_tokens=config["max_tokens"]
        )
        
        self._embedding_models_cache[cache_key] = model
        return model
    
    def get_llm_endpoint(
        self,
        model_type: LLMModelType = LLMModelType.LLAMA_3_70B,
        custom_endpoint: Optional[str] = None
    ) -> LLMEndpoint:
        """
        Get an NVIDIA NIM LLM inference endpoint.
        
        Args:
            model_type: Type of LLM model to retrieve
            custom_endpoint: Optional custom endpoint URL
            
        Returns:
            LLMEndpoint instance configured for the specified model
            
        Example:
            >>> integration = NVIDIAPlatformIntegration(api_key="your-key")
            >>> endpoint = integration.get_llm_endpoint(LLMModelType.LLAMA_3_70B)
            >>> print(endpoint.max_tokens)  # 4096
        """
        cache_key = f"{model_type.value}:{custom_endpoint}"
        
        if cache_key in self._llm_endpoints_cache:
            return self._llm_endpoints_cache[cache_key]
        
        # Model configurations
        model_configs = {
            LLMModelType.LLAMA_3_70B: {
                "max_tokens": 4096,
                "supports_streaming": True,
                "endpoint": f"{self.nim_base_url}/chat/completions"
            },
            LLMModelType.LLAMA_3_8B: {
                "max_tokens": 4096,
                "supports_streaming": True,
                "endpoint": f"{self.nim_base_url}/chat/completions"
            },
            LLMModelType.MISTRAL_7B: {
                "max_tokens": 8192,
                "supports_streaming": True,
                "endpoint": f"{self.nim_base_url}/chat/completions"
            },
            LLMModelType.MIXTRAL_8X7B: {
                "max_tokens": 8192,
                "supports_streaming": True,
                "endpoint": f"{self.nim_base_url}/chat/completions"
            },
            LLMModelType.NEMOTRON_340B: {
                "max_tokens": 4096,
                "supports_streaming": True,
                "endpoint": f"{self.nim_base_url}/chat/completions"
            }
        }
        
        config = model_configs[model_type]
        endpoint = custom_endpoint or self.custom_endpoints.get(model_type.value) or config["endpoint"]
        
        llm_endpoint = LLMEndpoint(
            model_name=model_type.value,
            model_type=model_type,
            endpoint=endpoint,
            max_tokens=config["max_tokens"],
            supports_streaming=config["supports_streaming"]
        )
        
        self._llm_endpoints_cache[cache_key] = llm_endpoint
        return llm_endpoint
    
    def get_nemotron_synthesizer(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        custom_endpoint: Optional[str] = None
    ) -> SyntheticDataModel:
        """
        Get Nemotron-4-340B model for synthetic data generation.
        
        Args:
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            custom_endpoint: Optional custom endpoint URL
            
        Returns:
            SyntheticDataModel instance configured for Nemotron-4-340B
            
        Example:
            >>> integration = NVIDIAPlatformIntegration(api_key="your-key")
            >>> synthesizer = integration.get_nemotron_synthesizer(temperature=0.8)
            >>> print(synthesizer.model_name)  # nvidia/nemotron-4-340b-instruct
        """
        endpoint = custom_endpoint or self.custom_endpoints.get("nemotron") or self.DEFAULT_NEMOTRON_ENDPOINT
        
        return SyntheticDataModel(
            model_name=LLMModelType.NEMOTRON_340B.value,
            endpoint=endpoint,
            max_tokens=4096,
            temperature=temperature,
            top_p=top_p
        )
    
    def configure_endpoint(self, model_identifier: str, endpoint_url: str) -> None:
        """
        Configure a custom endpoint for a specific model.
        
        Args:
            model_identifier: Model name or identifier
            endpoint_url: Custom endpoint URL
            
        Example:
            >>> integration = NVIDIAPlatformIntegration(api_key="your-key")
            >>> integration.configure_endpoint("nvidia/nv-embed-v2", "https://custom.endpoint.com")
        """
        self.custom_endpoints[model_identifier] = endpoint_url
        
        # Clear relevant caches
        self._embedding_models_cache.clear()
        self._llm_endpoints_cache.clear()
    
    def get_api_key(self) -> str:
        """
        Get the configured API key.
        
        Returns:
            The NVIDIA API key (masked for security)
        """
        if len(self.api_key) > 8:
            return f"{self.api_key[:4]}...{self.api_key[-4:]}"
        return "***"
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary with validation results
            
        Example:
            >>> integration = NVIDIAPlatformIntegration(api_key="your-key")
            >>> result = integration.validate_configuration()
            >>> print(result["api_key_configured"])  # True
        """
        return {
            "api_key_configured": bool(self.api_key),
            "nim_base_url": self.nim_base_url,
            "custom_endpoints_count": len(self.custom_endpoints),
            "custom_endpoints": list(self.custom_endpoints.keys()),
            "embedding_models_cached": len(self._embedding_models_cache),
            "llm_endpoints_cached": len(self._llm_endpoints_cache)
        }
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available model types.
        
        Returns:
            Dictionary mapping model categories to available models
        """
        return {
            "embedding_models": [model.value for model in EmbeddingModelType],
            "llm_models": [model.value for model in LLMModelType],
            "synthetic_data_models": [LLMModelType.NEMOTRON_340B.value]
        }
