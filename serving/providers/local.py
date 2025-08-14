import aiohttp
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..base import LLMProvider, LLMRequest, LLMResponse


@dataclass
class VLLMMetrics:
    """Metrics from a vLLM instance at a point in time."""
    timestamp: float
    instance_id: str
    
    # Request metrics
    requests_running: int = 0
    requests_waiting: int = 0
    requests_swapped: int = 0
    
    # Performance metrics
    avg_prompt_throughput_toks_per_s: float = 0.0
    avg_generation_throughput_toks_per_s: float = 0.0
    
    # GPU metrics
    gpu_cache_usage_perc: float = 0.0
    gpu_memory_usage_bytes: float = 0.0


class LocalProvider(LLMProvider):
    """Provider for local GPU serving (vLLM or SGLang)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.metrics_url = config.get("metrics_url", f"{self.base_url}/metrics")
        self.instance_id = config.get("instance_id", "local")
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "model": request.model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        return LLMResponse(
            text=data["choices"][0]["text"],
            model=request.model,
            provider="local"
        )
    
    async def get_metrics(self) -> Optional[VLLMMetrics]:
        """Fetch metrics from vLLM Prometheus endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.metrics_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return self._parse_prometheus_metrics(text)
        except Exception:
            # Instance might be down or unreachable
            pass
        return None
    
    def _parse_prometheus_metrics(self, text: str) -> VLLMMetrics:
        """Parse Prometheus format metrics from vLLM."""
        metrics = VLLMMetrics(
            timestamp=time.time(),
            instance_id=self.instance_id
        )
        
        for line in text.strip().split('\n'):
            if line.startswith('#') or not line:
                continue
                
            try:
                if 'vllm:num_requests_running' in line:
                    metrics.requests_running = int(float(line.split()[-1]))
                elif 'vllm:num_requests_waiting' in line:
                    metrics.requests_waiting = int(float(line.split()[-1]))
                elif 'vllm:num_requests_swapped' in line:
                    metrics.requests_swapped = int(float(line.split()[-1]))
                elif 'vllm:avg_prompt_throughput_toks_per_s' in line:
                    metrics.avg_prompt_throughput_toks_per_s = float(line.split()[-1])
                elif 'vllm:avg_generation_throughput_toks_per_s' in line:
                    metrics.avg_generation_throughput_toks_per_s = float(line.split()[-1])
                elif 'vllm:gpu_cache_usage_perc' in line:
                    # Convert to percentage (multiply by 100)
                    metrics.gpu_cache_usage_perc = float(line.split()[-1]) * 100
            except (ValueError, IndexError):
                continue
                
        return metrics