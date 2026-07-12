# h200-idle-proxy

Idle reverse proxy for **DeepSeek-V4-Flash (FP8)** on a multi-GPU **H200** box.
Reuses [`local_deployment_proxy.py`](../local_deployment_proxy/local_deployment_proxy.py)
with a dedicated model profile and port.

| Setting | Value |
|---|---|
| Listen port | **8003** (8001 = local RTX Qwen, 8002 = Spark) |
| Model | `deepseek-v4-flash` → `sgl-project/DeepSeek-V4-Flash-FP8` |
| Engine | sglang, `pipeline_parallel_size: 3` |
| GPUs | **0,2,3** (GPU **1** left free for other tenants) |
| Max context | **1,048,576** tokens (1M — model's YARN-extended architectural max) |
| Idle stop | 24 min (`IDLE_TIMEOUT=1440`) |

## How it works

```
Client → staging/prod host:8003 ──SSH tunnel──→ H200 box :8003 (proxy)
                                                  └─ model="deepseek-v4-flash" → :18003 (sglang, PP=3 on GPUs 0,2,3)
```

1. Proxy listens on port **8003**.
2. First request for `deepseek-v4-flash` starts the sglang container on GPUs 0, 2 and 3.
3. After 24 minutes with no traffic the container stops; the proxy stays up.

## Quick start

### Foreground

```bash
MODELS_CONFIG=ops/h200_idle_proxy/models.json LISTEN_PORT=8003 \
  python ops/local_deployment_proxy/local_deployment_proxy.py
```

### Background daemon + tunnels to staging and prod

```bash
# spark2 = staging gateway; jason@internal.freeinference.org = production
SSH_HOST='spark2|jason@internal.freeinference.org' REMOTE_PORT=8003 \
  ./ops/h200_idle_proxy/h200_idle_service.sh start

./ops/h200_idle_proxy/h200_idle_service.sh status
./ops/h200_idle_proxy/h200_idle_service.sh stop
```

Logs: `/tmp/h200_idle_proxy_<uid>_8003/proxy.log` (per-user run dir, mode 700).

### systemd (recommended)

```bash
sudo ./ops/h200_idle_proxy/install.sh
# remove:
sudo ./ops/h200_idle_proxy/uninstall.sh
```

Default tunnels: `spark2` and `jason@internal.freeinference.org` on remote port 8003.

## Gateway configuration

On **both** staging and production, set:

```bash
H200_DEPLOYMENT_URL=http://host.docker.internal:8003/v1
LOCAL_API_KEY=freeinference_api   # must match the proxy
```

`config/models.yaml` registers an optional `kind: sglang` route for
`deepseek-v4-flash` on `${H200_DEPLOYMENT_URL}` (skipped when the env var is blank).

## Model config

See [`models.json`](models.json):

| Field | Value |
|---|---|
| `gpu_index` | `"0,2,3"` — pins the 3 pipeline stages; GPU 1 is never claimed |
| `tensor_parallel_size` | `1` |
| `pipeline_parallel_size` | `3` |
| `backend_port` | `18003` |
| `model_dir` | `/netscratch/juncheng/models/DeepSeek-V4-Flash-FP8` |
| `max_model_len` | `1048576` (1M — the model's YARN architectural max; KV cache holds ~5.7M tokens so context is not VRAM-bound) |
| `mem_fraction` | `0.90` |

> **Why PP=3, not TP=2?** The FP8 weights are ~274 GiB. At TP=2 each rank would
> need ~137 GiB, which does not fit alongside KV/activations on a 143 GiB H200
> (the two GPUs' combined 281 GiB barely exceed the weights) — cold-start OOMs.
> TP=3 is illegal (64 attention heads are not divisible by 3). Pipeline
> parallelism splits the model by *layer*, so PP=3 puts ~91 GiB per GPU on GPUs
> 0, 2 and 3 with ~35–49 GiB free each for KV cache — all while GPU 1 stays free
> for other tenants. Measured KV capacity at PP=3: `max_total_num_tokens`
> ≈ 5.7M (FP8 KV cache), i.e. ~5.4× the 1M context.

## Requirements

- 4× NVIDIA H200 (or at least GPUs 0, 2 and 3 free)
- Docker + NVIDIA Container Toolkit
- `lmsysorg/sglang:latest`
- Weights at `model_dir` (or `hf_repo` download on first request)
- SSH to staging/prod with `GatewayPorts clientspecified` (or `yes`)
- `autossh` for durable tunnels (installed by `install.sh`)
