# Cost-Latency Simulator for Hybrid LLM Inference

This project simulates inference workloads across **local GPUs** and **cloud providers**, allowing you to compare **cost-latency tradeoffs** under different deployment strategies (local, cloud, hybrid).

## Structure
```
.
├── data/
│   └── BurstGPT_1.csv        # Sample workload trace
├── main.py                   # Entry point
├── src
│   ├── analyzer.py
│   ├── data_loader.py
│   ├── main.py
│   ├── plotter.py
│   └── simulate.py           # Main simulation logic
└── res/                      # Output plots
```

## Requirements

Install dependencies:

```bash
pip install matplotlib numpy pandas pyyaml
```

## Configuration

System parameters are defined in [`infra_config.yaml`](./infra_config.yaml):

For example:
```yaml
hardware:
  A100:
    cost_per_hour: 3.0
  L4:
    cost_per_hour: 0.6

token_rates:
  ChatGPT:
    L4:
      prefill: 300
      decode: 600
    A100:
      prefill: 500
      decode: 1000

api_providers:
  OpenAI:
    latency: 0.5
    cost_input:
      ChatGPT: 0.000005
    cost_output:
      ChatGPT: 0.000015
```

You can add more models, hardware, and providers as needed.


## How to Run
Edit `main.py` to specify:

```python
model = 'ChatGPT'
local_hw = 'L4'
cloud_provider = 'OpenAI'
```

Run the simulator with:

```bash
python src/main.py
```

The simulator will:

* Load the trace from `data/BurstGPT_1.csv`
* Run the simulation across different strategies (local / hybrid / cloud)
* Analyze latency stats
* Plot cost-latency tradeoffs into `res/`

## Output

Plots are saved in the `res/` directory:

* `res/tradeoff_ChatGPT_L4_cloud-OpenAI.pdf`

These show total cost vs. p99 latency for different deployment modes.


## Simulate on New Traces

To test with your own data, place a CSV file in `data/` with at least the following columns:

```csv
Timestamp,Model,Request tokens,Response tokens
0.0,ChatGPT,100,200
1.5,ChatGPT,120,150
...
```

Then update the file path in `main.py`:

```python
path = 'data/your_trace.csv'
```
