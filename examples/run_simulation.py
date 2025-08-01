"""Example of running a complete simulation experiment."""

import asyncio
import sys
sys.path.append('.')

from benchmark.workloads.burstgpt import BurstGPTWorkload
from simulator.metrics_collector import MetricsCollector
from simulator.experiment_runner import ExperimentRunner, ExperimentConfig
from serving.manager import LLMManager
from serving.routing import RoutingStrategy
from serving.base import LLMRequest


# Example routing strategy - researchers would implement their own
class SimpleRoundRobinStrategy(RoutingStrategy):
    """Example routing strategy for demonstration."""
    
    def __init__(self):
        self._index = 0
        
    async def select_provider(self, request: LLMRequest, providers):
        if not providers:
            raise ValueError("No providers available")
        provider_names = list(providers.keys())
        selected = provider_names[self._index % len(provider_names)]
        self._index += 1
        return selected


async def main():
    # 1. Setup LLM Manager with providers
    print("Setting up LLM providers...")
    manager = LLMManager()
    
    # Add your vLLM instances here
    # Example configuration:
    manager.add_provider("local_gpu_0", {
        "type": "local",
        "endpoint": "http://localhost:8001",
        "metrics_url": "http://localhost:8001/metrics"
    })
    
    manager.add_provider("local_gpu_1", {
        "type": "local", 
        "endpoint": "http://localhost:8002",
        "metrics_url": "http://localhost:8002/metrics"
    })
    
    # 2. Create experiment runner
    runner = ExperimentRunner(manager)
    
    # 3. Define experiment configurations
    experiments = [
        ExperimentConfig(
            name="RoundRobin_10x",
            trace_file="data/BurstGPT_1.csv",
            routing_strategy=SimpleRoundRobinStrategy(),
            time_scale=10.0,  # 10x speed
            duration_limit_seconds=60  # Run for max 60 seconds
        ),
        # Add more experiments with different routing strategies here
    ]
    
    # 4. Run experiments
    print("\nStarting experiments...")
    results = await runner.run_experiments(experiments)
    
    # 5. Show comparison
    runner.print_comparison(results)
    
    # 6. Detailed results
    print("\nDetailed Results:")
    for result in results:
        print(f"\n{result.config.name}:")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        metrics = result.metrics_summary
        print(f"  Total requests: {metrics.get('total_requests', 0)}")
        print(f"  Successful: {metrics.get('successful_requests', 0)}")
        print(f"  Failed: {metrics.get('failed_requests', 0)}")
        print(f"  Latency P50: {metrics.get('latency_p50_ms', 0):.1f}ms")
        print(f"  Latency P90: {metrics.get('latency_p90_ms', 0):.1f}ms")
        print(f"  Latency P99: {metrics.get('latency_p99_ms', 0):.1f}ms")


if __name__ == "__main__":
    # Run the simulation
    asyncio.run(main())