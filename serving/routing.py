from abc import ABC, abstractmethod
from typing import Dict, Any
from .base import LLMRequest


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies"""
    
    @abstractmethod
    async def select_provider(self, request: LLMRequest, providers: Dict[str, Any]) -> str:
        """Select which provider should handle the request"""
        pass