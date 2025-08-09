from typing import Dict, Any, Optional
from .base import LLMProvider, LLMRequest, LLMResponse
from .providers.local import LocalProvider
from .providers.openai import OpenAIProvider
from .providers.llama import LlamaProvider


class ServiceManager:
    """Manages multiple LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize providers based on config"""
        if "local" in self.config:
            self.providers["local"] = LocalProvider(self.config["local"])
        
        if "openai" in self.config:
            self.providers["openai"] = OpenAIProvider(self.config["openai"])
        
        if "llama" in self.config:
            self.providers["llama"] = LlamaProvider(self.config["llama"])
    
    async def generate(self, request: LLMRequest, provider: str) -> LLMResponse:
        """Generate completion using specified provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        return await self.providers[provider].generate(request)
    
    def get_provider(self, name: str) -> LLMProvider:
        """Get a provider by name"""
        if name not in self.providers:
            raise ValueError(f"Provider {name} not found")
        return self.providers[name]
    
    def list_providers(self) -> Dict[str, LLMProvider]:
        """Get all registered providers"""
        return self.providers.copy()