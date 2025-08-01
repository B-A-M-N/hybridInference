"""Run experiments with different routing strategies."""

import asyncio
from typing import Dict, List, Optional, Type
import time
from dataclasses import dataclass

from serving.base import LLMRequest, LLMResponse, LLMProvider
from serving.routing import RoutingStrategy
from serving.manager import LLMManager
from benchmark.runner import BenchmarkRunner
from benchmark.workloads.burstgpt import BurstGPTWorkload
from .metrics_collector import MetricsCollector


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    trace_file: str
    routing_strategy: RoutingStrategy
    time_scale: float = 1.0
    duration_limit_seconds: Optional[float] = None


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    metrics_summary: dict
    start_time: float
    end_time: float
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class ExperimentRunner:
    """Runs experiments with different configurations and collects results."""
    
    def __init__(self, manager: LLMManager):
        self.manager = manager
        self.metrics_collector = MetricsCollector()
        self._setup_metrics_collection()
        
    def _setup_metrics_collection(self):
        """Setup metrics collection for all providers."""
        # Add vLLM instances to metrics collector
        for name, provider in self.manager.providers.items():
            if hasattr(provider, 'metrics_url'):
                self.metrics_collector.add_vllm_instance(name, provider.metrics_url)
                
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        print(f"\nStarting experiment: {config.name}")
        print(f"  Trace file: {config.trace_file}")
        print(f"  Routing strategy: {config.routing_strategy.__class__.__name__}")
        print(f"  Time scale: {config.time_scale}x")
        
        # Set routing strategy
        self.manager.routing_strategy = config.routing_strategy
        
        # Reset metrics
        self.metrics_collector.request_metrics.clear()
        self.metrics_collector.vllm_metrics.clear()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Create request handler
        async def handle_request(request: LLMRequest, trace_id: str):
            try:
                # Record request start
                provider = await self.manager.routing_strategy.select_provider(
                    request, self.manager.providers
                )
                self.metrics_collector.record_request_start(
                    trace_id, request.model, request.max_tokens, provider
                )
                
                # Send request
                start_time = time.time()
                response = await self.manager.generate(request)
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record completion
                # Note: In real usage, we'd get actual TTFT from vLLM
                self.metrics_collector.record_request_end(
                    trace_id, 
                    completion_tokens=len(response.text.split()),  # Approximate
                    ttft_ms=total_time_ms * 0.1,  # Approximate TTFT as 10% of total
                    total_ms=total_time_ms,
                    success=True
                )
                
            except Exception as e:
                self.metrics_collector.record_request_end(
                    trace_id, 0, 0, 0, success=False, error=str(e)
                )
        
        # Create benchmark runner
        workload = BurstGPTWorkload(config.trace_file)
        runner = BenchmarkRunner(workload, config.time_scale)
        
        # Show workload stats
        stats = workload.get_stats()
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Duration: {stats.get('duration_seconds', 0):.1f}s")
        
        start_time = time.time()
        
        # Run benchmark
        await runner.run(handle_request, config.duration_limit_seconds)
            
        end_time = time.time()
        
        # Wait a bit for final requests to complete
        await asyncio.sleep(2.0)
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Get results
        result = ExperimentResult(
            config=config,
            metrics_summary=self.metrics_collector.get_summary(),
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"  Completed in {result.duration_seconds:.1f}s")
        print(f"  Total requests: {result.metrics_summary.get('total_requests', 0)}")
        print(f"  Success rate: {result.metrics_summary.get('successful_requests', 0) / max(1, result.metrics_summary.get('total_requests', 1)) * 100:.1f}%")
        
        return result
        
    async def run_experiments(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run multiple experiments sequentially."""
        results = []
        
        for config in configs:
            result = await self.run_experiment(config)
            results.append(result)
            
            # Cool down between experiments
            await asyncio.sleep(5.0)
            
        return results
        
    def print_comparison(self, results: List[ExperimentResult]):
        """Print a comparison table of results."""
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Strategy':<20} {'Requests':<10} {'Success':<10} "
              f"{'P50 (ms)':<10} {'P90 (ms)':<10} {'P99 (ms)':<10}")
        print("-"*80)
        
        # Results
        for result in results:
            metrics = result.metrics_summary
            success_rate = metrics.get('successful_requests', 0) / max(1, metrics.get('total_requests', 1)) * 100
            
            print(f"{result.config.name:<20} "
                  f"{metrics.get('total_requests', 0):<10} "
                  f"{success_rate:<10.1f} "
                  f"{metrics.get('latency_p50_ms', 0):<10.1f} "
                  f"{metrics.get('latency_p90_ms', 0):<10.1f} "
                  f"{metrics.get('latency_p99_ms', 0):<10.1f}")
                  
        print("="*80)