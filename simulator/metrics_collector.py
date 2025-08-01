"""Collect metrics from vLLM instances via Prometheus endpoints."""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Optional
import time


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
    

@dataclass 
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    timestamp: float
    model: str
    provider: str
    
    # Latency metrics (in ms)
    time_to_first_token_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    
    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Status
    success: bool = True
    error: Optional[str] = None


class MetricsCollector:
    """Collects metrics from vLLM instances and tracks request performance."""
    
    def __init__(self):
        self.vllm_metrics: List[VLLMMetrics] = []
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self._collection_task: Optional[asyncio.Task] = None
        self._vllm_endpoints: Dict[str, str] = {}
        
    def add_vllm_instance(self, instance_id: str, metrics_url: str):
        """Add a vLLM instance to monitor."""
        self._vllm_endpoints[instance_id] = metrics_url
        
    async def _parse_prometheus_metrics(self, text: str, instance_id: str) -> VLLMMetrics:
        """Parse Prometheus format metrics from vLLM."""
        metrics = VLLMMetrics(
            timestamp=time.time(),
            instance_id=instance_id
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
                    metrics.gpu_cache_usage_perc = float(line.split()[-1])
            except (ValueError, IndexError):
                continue
                
        return metrics
        
    async def _collect_vllm_metrics(self):
        """Continuously collect metrics from all vLLM instances."""
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = []
                for instance_id, url in self._vllm_endpoints.items():
                    tasks.append(self._fetch_metrics(session, instance_id, url))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, VLLMMetrics):
                        self.vllm_metrics.append(result)
                
                await asyncio.sleep(1.0)  # Collect every second
                
    async def _fetch_metrics(self, session: aiohttp.ClientSession, 
                           instance_id: str, url: str) -> Optional[VLLMMetrics]:
        """Fetch metrics from a single vLLM instance."""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    return await self._parse_prometheus_metrics(text, instance_id)
        except Exception:
            # Instance might be down or unreachable
            pass
        return None
        
    def start_collection(self):
        """Start background metrics collection."""
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collect_vllm_metrics())
            
    def stop_collection(self):
        """Stop background metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            self._collection_task = None
            
    def record_request_start(self, request_id: str, model: str, 
                           prompt_tokens: int, provider: str):
        """Record the start of a request."""
        self.request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            timestamp=time.time(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens
        )
        
    def record_request_end(self, request_id: str, completion_tokens: int,
                          ttft_ms: float, total_ms: float, success: bool = True,
                          error: Optional[str] = None):
        """Record the completion of a request."""
        if request_id in self.request_metrics:
            metrics = self.request_metrics[request_id]
            metrics.completion_tokens = completion_tokens
            metrics.time_to_first_token_ms = ttft_ms
            metrics.total_latency_ms = total_ms
            metrics.success = success
            metrics.error = error
            
    def get_summary(self) -> dict:
        """Get summary statistics."""
        successful_requests = [m for m in self.request_metrics.values() if m.success]
        
        if not successful_requests:
            return {}
            
        latencies = [m.total_latency_ms for m in successful_requests if m.total_latency_ms]
        ttfts = [m.time_to_first_token_ms for m in successful_requests if m.time_to_first_token_ms]
        
        # Calculate percentiles
        latencies.sort()
        ttfts.sort()
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] if f == c else data[f] * (c - k) + data[c] * (k - f)
        
        return {
            "total_requests": len(self.request_metrics),
            "successful_requests": len(successful_requests),
            "failed_requests": len(self.request_metrics) - len(successful_requests),
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p90_ms": percentile(latencies, 0.9),
            "latency_p99_ms": percentile(latencies, 0.99),
            "ttft_p50_ms": percentile(ttfts, 0.5),
            "ttft_p90_ms": percentile(ttfts, 0.9),
            "ttft_p99_ms": percentile(ttfts, 0.99),
        }