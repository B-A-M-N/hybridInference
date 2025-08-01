"""Benchmark runner that coordinates workload execution."""

import asyncio
import time
from typing import Optional, Callable

from .workloads.base import BaseWorkload


class BenchmarkRunner:
    """Runs benchmarks with specified workloads."""
    
    def __init__(self, workload: BaseWorkload, time_scale: float = 1.0):
        """
        Args:
            workload: The workload pattern to execute
            time_scale: Speed multiplier (e.g., 10.0 = 10x faster)
        """
        self.workload = workload
        self.time_scale = time_scale
        
    async def run(self, 
                  request_handler: Callable,
                  duration_limit: Optional[float] = None):
        """
        Run the benchmark.
        
        Args:
            request_handler: Async function to handle each request
            duration_limit: Optional time limit in seconds
        """
        start_time = time.time()
        first_timestamp = None
        
        async for workload_request in self.workload.generate():
            # Initialize first timestamp
            if first_timestamp is None:
                first_timestamp = workload_request.timestamp_ms
                
            # Calculate when to send this request
            relative_ms = workload_request.timestamp_ms - first_timestamp
            target_time = start_time + (relative_ms / 1000.0 / self.time_scale)
            
            # Check duration limit
            if duration_limit and (time.time() - start_time) > duration_limit:
                print(f"Duration limit reached ({duration_limit}s)")
                break
                
            # Wait if needed
            current_time = time.time()
            if target_time > current_time:
                await asyncio.sleep(target_time - current_time)
                
            # Send request asynchronously
            asyncio.create_task(
                request_handler(workload_request.request, workload_request.request_id)
            )