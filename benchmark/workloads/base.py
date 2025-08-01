"""Base interface for workload generators."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any
from dataclasses import dataclass

from serving.base import LLMRequest


@dataclass
class WorkloadRequest:
    """A request with timing information."""
    timestamp_ms: float
    request: LLMRequest
    request_id: str


class BaseWorkload(ABC):
    """Abstract base class for workload patterns."""
    
    @abstractmethod
    async def generate(self) -> AsyncGenerator[WorkloadRequest, None]:
        """Generate requests according to the workload pattern."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this workload."""
        pass