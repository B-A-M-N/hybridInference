# Hybrid Inference Simulator

This simulator helps evaluate routing algorithms for hybrid cloud/local LLM inference using real traces.

## Components

### 1. Trace Replay (`trace_replay.py`)
- Loads BurstGPT traces from CSV files
- Replays requests with accurate timing
- Supports time scaling (e.g., 10x speed)

### 2. Metrics Collector (`metrics_collector.py`)
- Collects metrics from vLLM Prometheus endpoints
- Tracks per-request latency and throughput
- Calculates percentile statistics

### 3. Experiment Runner (`experiment_runner.py`)
- Orchestrates experiments with different routing strategies
- Manages trace replay and metrics collection
- Generates comparison reports

## Usage

1. **Start vLLM instances** with metrics enabled:
```bash
# Terminal 1
USE_ALL_GPU=false PORT=8001 ./scripts/start_vllm.sh

# Terminal 2  
USE_ALL_GPU=false PORT=8002 ./scripts/start_vllm.sh
```

2. **Implement your routing strategy**:
```python
from serving.routing import RoutingStrategy
from serving.base import LLMRequest

class MyRoutingStrategy(RoutingStrategy):
    async def select_provider(self, request: LLMRequest, providers):
        # Your routing logic here
        return selected_provider_name
```

3. **Run experiments**:
```python
from simulator.experiment_runner import ExperimentRunner, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    name="MyStrategy",
    trace_file="data/BurstGPT_1.csv",
    routing_strategy=MyRoutingStrategy(),
    time_scale=10.0,  # 10x speed
    duration_limit_seconds=300  # 5 minutes max
)

# Run and compare
results = await runner.run_experiments([config])
runner.print_comparison(results)
```

## Metrics

The simulator collects:
- **Latency**: P50, P90, P99 for request completion
- **Throughput**: Requests/second, tokens/second  
- **Reliability**: Success rate, error tracking
- **Resource Usage**: GPU utilization, queue lengths

## Trace Format

BurstGPT traces should have columns:
- Timestamp: Time in seconds
- Model: Model name (e.g., "ChatGPT", "GPT-4")
- Request tokens: Input token count
- Response tokens: Output token count