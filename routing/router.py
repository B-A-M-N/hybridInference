"""Main router that coordinates routing decisions."""

from typing import Dict, Any, Optional
from serving.base import LLMRequest, LLMResponse
from .base import RoutingStrategy


class Router:
    """Routes requests to appropriate providers using routing strategies."""
    
    def __init__(self, strategy: Optional[RoutingStrategy] = None):
        self.strategy = strategy
        self.providers: Dict[str, Any] = {}
    
    def register_provider(self, name: str, provider: Any):
        """Register a provider."""
        self.providers[name] = provider
    
    async def route(self, request: LLMRequest) -> tuple[str, Any]:
        """
        Route a request to a provider.
        
        Returns:
            Tuple of (provider_name, provider_instance)
        """
        if not self.providers:
            raise RuntimeError("No providers registered")
        
        if self.strategy:
            provider_name = await self.strategy.select_provider(request, self.providers)
        else:
            # Simple fallback: use first available provider
            provider_name = next(iter(self.providers))
        
        return provider_name, self.providers[provider_name]