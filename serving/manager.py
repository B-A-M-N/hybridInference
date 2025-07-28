from typing import Dict, Any, Optional
from .base import LLMProvider, LLMRequest, LLMResponse
from .providers.local import LocalProvider
from .providers.openai import OpenAIProvider
from .providers.llama import LlamaProvider


class ServiceManager:
    """Manages multiple LLM providers and routes requests"""
    
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
    
    async def generate(self, request: LLMRequest, provider: Optional[str] = None) -> LLMResponse:
        """Generate completion using specified provider or default routing"""
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not found")
            return await self.providers[provider].generate(request)
        
        # Simple routing: try local first, fallback to cloud
        if "local" in self.providers:
            try:
                return await self.providers["local"].generate(request)
            except Exception:
                # Fallback to cloud providers
                pass
        
        # Try cloud providers
        for provider_name in ["openai", "llama"]:
            if provider_name in self.providers:
                return await self.providers[provider_name].generate(request)
        
        raise RuntimeError("No available providers")