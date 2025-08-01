"""BurstGPT workload implementation."""

import pandas as pd
from typing import AsyncGenerator, Dict, Any, Optional

from .base import BaseWorkload, WorkloadRequest
from serving.base import LLMRequest


class BurstGPTWorkload(BaseWorkload):
    """Generate workload from BurstGPT traces."""
    
    def __init__(self, trace_file: str, prompt_file: Optional[str] = None):
        self.trace_file = trace_file
        self.prompt_file = prompt_file
        self._df: Optional[pd.DataFrame] = None
        
    def _load_trace(self):
        """Load the trace file if not already loaded."""
        if self._df is None:
            self._df = pd.read_csv(self.trace_file)
    
    async def generate(self) -> AsyncGenerator[WorkloadRequest, None]:
        """Generate requests from BurstGPT trace."""
        self._load_trace()
        
        for idx, row in self._df.iterrows():
            # For now, use placeholder prompts
            # TODO: Integrate with prompt manager if needed
            prompt = f"<Prompt for {row['Request tokens']} tokens>"
            
            request = LLMRequest(
                prompt=prompt,
                model=row['Model'],
                max_tokens=int(row['Response tokens'])
            )
            
            yield WorkloadRequest(
                timestamp_ms=int(row['Timestamp']) * 1000,
                request=request,
                request_id=f"burst_{idx}"
            )
            
    def get_stats(self) -> Dict[str, Any]:
        """Get trace statistics."""
        self._load_trace()
        
        if self._df is None or len(self._df) == 0:
            return {}
            
        duration_seconds = self._df['Timestamp'].max() - self._df['Timestamp'].min()
        model_counts = self._df['Model'].value_counts().to_dict()
        
        return {
            'total_requests': len(self._df),
            'duration_seconds': duration_seconds,
            'requests_per_second': len(self._df) / duration_seconds if duration_seconds > 0 else 0,
            'model_distribution': model_counts,
            'avg_prompt_tokens': self._df['Request tokens'].mean(),
            'avg_completion_tokens': self._df['Response tokens'].mean()
        }