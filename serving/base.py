from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class LLMRequest:
    prompt: str
    model: str
    max_tokens: Optional[int] = 4096
    temperature: float = 0.7
    

@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request"""
        pass