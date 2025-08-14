"""BurstGPT experiment on local deployment."""

import asyncio
import time
from serving.base import LLMRequest, LLMResponse, LLMProvider
from serving.config import get_config
from serving.providers.local import LocalProvider
from client.loader import BurstGPTLoader
from client.metrics import MetricsCollector


async def mock_local_handler(request: LLMRequest, request_id: str, provider: LLMProvider, metrics: MetricsCollector):
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
        response = await provider.generate(request)
        
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
    print("Connecting to vLLM at http://localhost:8000")
    
    # Setup
    config = get_config()
    local_provider = LocalProvider(config["local"])
    metrics = MetricsCollector()
    
    # Load BurstGPT data
    loader = BurstGPTLoader(
        trace_file="data/BurstGPT_1.csv",
        time_scale=50,  # 50x faster for testing
        max_requests=30    # Process 30 requests
    )
    
    # Collect vLLM metrics periodically
    vllm_metrics_history = []
    
    async def collect_vllm_metrics():
        """Collect vLLM metrics every second during the experiment."""
        while True:
            vllm_metrics = await local_provider.get_metrics()
            if vllm_metrics:
                vllm_metrics_history.append(vllm_metrics)
                # Only print when there's meaningful activity
                if vllm_metrics.requests_running > 0 or vllm_metrics.requests_waiting > 0:
                    print(f"vLLM Status - Running: {vllm_metrics.requests_running}, "
                          f"Waiting: {vllm_metrics.requests_waiting}, "
                          f"GPU Cache: {vllm_metrics.gpu_cache_usage_perc:.1f}%")
            await asyncio.sleep(1)
    
    # Start metrics collection
    metrics_task = asyncio.create_task(collect_vllm_metrics())
    
    # Run experiment
    request_count = 0
    start_time = time.time()
    
    async def handle_request(request: LLMRequest):
        nonlocal request_count
        request_id = f"burst_{request_count}"
        request_count += 1
        
        request.model = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B"
        request.max_tokens = 20  # Limit tokens for faster testing
        
        await mock_local_handler(request, request_id, local_provider, metrics)
    
    # Process requests
    tasks = []
    async for request in loader.load_requests():
        task = asyncio.create_task(handle_request(request))
        tasks.append(task)
    
    # Wait for all requests to complete
    print("Waiting for all requests to complete...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop metrics collection
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
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
    
    # Show vLLM metrics summary
    if vllm_metrics_history:
        print(f"\nvLLM Metrics Summary:")
        max_running = max(m.requests_running for m in vllm_metrics_history)
        max_waiting = max(m.requests_waiting for m in vllm_metrics_history)
        avg_gpu_cache = sum(m.gpu_cache_usage_perc for m in vllm_metrics_history) / len(vllm_metrics_history)
        avg_prompt_throughput = sum(m.avg_prompt_throughput_toks_per_s for m in vllm_metrics_history) / len(vllm_metrics_history)
        avg_gen_throughput = sum(m.avg_generation_throughput_toks_per_s for m in vllm_metrics_history) / len(vllm_metrics_history)
        
        print(f"  - Max concurrent requests: {max_running}")
        print(f"  - Max waiting requests: {max_waiting}")
        print(f"  - Avg GPU cache usage: {avg_gpu_cache:.1f}%")
        print(f"  - Avg prompt throughput: {avg_prompt_throughput:.1f} toks/s")
        print(f"  - Avg generation throughput: {avg_gen_throughput:.1f} toks/s")


if __name__ == "__main__":
    asyncio.run(run_burstgpt_experiment())