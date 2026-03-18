#!/usr/bin/env python3
"""External LLM latency prober.

Probes configured API endpoints every probe_interval_seconds and exposes
per-target TTFT, throughput, and availability as Prometheus metrics.

Metrics are served on :9116/metrics (configurable via config.yml or METRICS_PORT).

Usage:
    python prober.py [config.yml]             # run continuously
    python prober.py [config.yml] --run-once  # probe once and exit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import yaml
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)

logger = logging.getLogger("latency_prober")

# ---------------------------------------------------------------------------
# Prometheus metrics (isolated registry to avoid collisions with other libs)
# ---------------------------------------------------------------------------

_registry = CollectorRegistry(auto_describe=True)

_SUCCESS = Gauge(
    "llm_probe_success",
    "1 if the last probe succeeded, 0 otherwise",
    ["target", "model"],
    registry=_registry,
)
_TTFT = Gauge(
    "llm_probe_ttft_seconds",
    "Time to first output token in seconds",
    ["target", "model"],
    registry=_registry,
)
_E2E = Gauge(
    "llm_probe_e2e_latency_seconds",
    "End-to-end request latency in seconds",
    ["target", "model"],
    registry=_registry,
)
_THROUGHPUT = Gauge(
    "llm_probe_throughput_tokens_per_second",
    "Output token throughput (tokens/sec) after first token",
    ["target", "model"],
    registry=_registry,
)
_TOKENS = Gauge(
    "llm_probe_output_tokens",
    "Number of output tokens in the last probe",
    ["target", "model"],
    registry=_registry,
)
_LAST_RUN = Gauge(
    "llm_probe_last_run_timestamp_seconds",
    "Unix timestamp of the last probe attempt",
    ["target", "model"],
    registry=_registry,
)

# ---------------------------------------------------------------------------
# Fixed probe prompt — ~100 output tokens for reliable throughput measurement.
# ---------------------------------------------------------------------------

_PROBE_MESSAGES = [
    {
        "role": "user",
        "content": "Count from 1 to 50, writing each number on its own line. Do not add any other text.",
    }
]

# ---------------------------------------------------------------------------
# Probe logic
# ---------------------------------------------------------------------------


async def _stream_request(
    client: httpx.AsyncClient, url: str, api_key: str, model: str, timeout: float
) -> tuple[float, float, int]:
    """Stream one chat completion and return (ttft_s, e2e_s, output_tokens).

    output_tokens prefers usage.completion_tokens from the server; falls back
    to a character-count estimate (~4 chars/token).
    """
    body = {
        "model": model,
        "messages": _PROBE_MESSAGES,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    t_start = time.perf_counter()
    ttft: float | None = None
    output_tokens = 0
    content_chars = 0  # running count for fallback token estimation

    async with client.stream("POST", url, headers=headers, json=body, timeout=timeout) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    content_chars += len(content)

            if chunk.get("usage"):
                output_tokens = chunk["usage"].get("completion_tokens") or 0

    e2e = time.perf_counter() - t_start

    if output_tokens == 0 and content_chars:
        output_tokens = max(1, content_chars // 4)
    if ttft is None:
        ttft = e2e

    return ttft, e2e, output_tokens


async def run_probe(client: httpx.AsyncClient, target: dict[str, Any]) -> None:
    """Probe one target and write results to Prometheus metrics."""
    name: str = target["name"]
    model: str = target["model"]
    labels = {"target": name, "model": model}

    api_key = os.environ.get(target["api_key_env"], "")
    if not api_key:
        logger.warning("Skipping %s: env var %s is not set", name, target["api_key_env"])
        _SUCCESS.labels(**labels).set(0)
        _LAST_RUN.labels(**labels).set(time.time())
        return

    logger.info("Probing %s (%s) …", name, model)
    try:
        ttft, e2e, tokens = await _stream_request(
            client,
            target["url"],
            api_key,
            model,
            float(target.get("timeout_seconds", 60)),
        )
        tps = tokens / max(e2e - ttft, 0.001) if tokens else 0.0

        _SUCCESS.labels(**labels).set(1)
        _TTFT.labels(**labels).set(ttft)
        _E2E.labels(**labels).set(e2e)
        _THROUGHPUT.labels(**labels).set(tps)
        _TOKENS.labels(**labels).set(tokens)
        logger.info(
            "  %s: ttft=%.3fs e2e=%.3fs tokens=%d tps=%.1f",
            name,
            ttft,
            e2e,
            tokens,
            tps,
        )
    except Exception as exc:
        logger.warning("  %s FAILED: %s", name, exc)
        _SUCCESS.labels(**labels).set(0)
    finally:
        _LAST_RUN.labels(**labels).set(time.time())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


async def probe_once(client: httpx.AsyncClient, targets: list[dict]) -> None:
    """Probe all targets concurrently."""
    await asyncio.gather(*(run_probe(client, t) for t in targets), return_exceptions=True)


async def probe_loop(targets: list[dict], interval: float) -> None:
    """Run probe_once every interval seconds, accounting for probe duration."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            t0 = time.monotonic()
            await probe_once(client, targets)
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, interval - elapsed)
            logger.info("Cycle done in %.1fs — sleeping %.0fs.", elapsed, sleep_for)
            await asyncio.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Prometheus metrics HTTP server (background daemon thread)
# ---------------------------------------------------------------------------


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/metrics", "/"):
            output = generate_latest(_registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_: Any) -> None:  # suppress access logs
        pass


def start_metrics_server(port: int) -> None:
    """Start the Prometheus /metrics HTTP server in a background daemon thread."""
    server = HTTPServer(("", port), _MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Metrics server listening on :%d/metrics", port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, start the metrics server, and run the probe loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    args = sys.argv[1:]
    run_once = "--run-once" in args
    config_path = next((a for a in args if not a.startswith("-")), "config.yml")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    targets: list[dict] = config.get("targets", [])
    if not targets:
        logger.error("No targets defined in %s", config_path)
        sys.exit(1)

    interval = float(config.get("probe_interval_seconds", 300))
    metrics_port = int(os.environ.get("METRICS_PORT", config.get("metrics_port", 9116)))

    if run_once:

        async def _once() -> None:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                await probe_once(client, targets)

        asyncio.run(_once())
        return

    start_metrics_server(metrics_port)
    logger.info(
        "Starting prober: %d target(s), %.0fs interval, metrics on :%d",
        len(targets),
        interval,
        metrics_port,
    )
    asyncio.run(probe_loop(targets, interval))


if __name__ == "__main__":
    main()
