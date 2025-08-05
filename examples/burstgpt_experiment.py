"""BurstGPT experiment on local deployment."""

import asyncio
import time
from serving.base import LLMRequest, LLMResponse
from serving.manager import ServiceManager
from serving.config import get_config
from client.loaders import BurstGPTLoader
from client.metrics import MetricsCollector


async def mock_local_handler(request: LLMRequest, request_id: str, service_manager: ServiceManager, metrics: MetricsCollector):
    """Handle request using local vLLM service."""
    start_time = time.time()
    
    # Record request start
    metrics.record_request_start(
        request_id=request_id,
        provider="local",
        model=request.model,
        prompt_tokens=len(request.prompt.split())  # Rough estimate
    )
    
    try:
        # Call local service
        response = await service_manager.generate(request, "local")
        
        # Record success
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request_end(
            request_id=request_id,
            completion_tokens=len(response.text.split()) if response.text else 0,
            request_latency_ms=latency_ms,
            success=True
        )
        
        return response
        
    except Exception as e:
        # Record failure  
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_request_end(
            request_id=request_id,
            completion_tokens=0,
            request_latency_ms=latency_ms,
            success=False,
            error=str(e)
        )
        print(f"Request {request_id} failed: {e}")


async def run_burstgpt_experiment():
    """Run BurstGPT experiment."""
    print("Starting BurstGPT experiment...")
    
    # Setup
    config = get_config()
    service_manager = ServiceManager(config)
    metrics = MetricsCollector()
    
    # Load BurstGPT data
    loader = BurstGPTLoader(
        trace_file="data/BurstGPT_1.csv",
        time_scale=100.0,  # 100x faster
        max_requests=50    # Limit for testing
    )
    
    # Run experiment
    request_count = 0
    start_time = time.time()
    
    async def handle_request(request: LLMRequest):
        nonlocal request_count
        request_id = f"burst_{request_count}"
        request_count += 1
        
        await mock_local_handler(request, request_id, service_manager, metrics)
    
    # Process requests
    tasks = []
    async for request in loader.load_requests():
        task = asyncio.create_task(handle_request(request))
        tasks.append(task)
    
    # Wait for all requests to complete
    print("Waiting for all requests to complete...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Show results
    duration = time.time() - start_time
    print(f"\nExperiment completed in {duration:.2f}s")
    print(f"Summary: {metrics.get_summary()}")
    
    # Show provider metrics
    local_metrics = metrics.get_provider_metrics("local")
    if local_metrics:
        print(f"\nLocal Provider Metrics:")
        print(f"  - Total requests: {local_metrics.total_requests}")
        print(f"  - Success rate: {(local_metrics.successful_requests/local_metrics.total_requests)*100:.1f}%")
        print(f"  - Avg latency: {local_metrics.avg_latency_ms:.1f}ms")
        print(f"  - P90 latency: {local_metrics.p90_latency_ms:.1f}ms")
        print(f"  - Throughput: {local_metrics.requests_per_second:.1f} req/s")


if __name__ == "__main__":
    asyncio.run(run_burstgpt_experiment())