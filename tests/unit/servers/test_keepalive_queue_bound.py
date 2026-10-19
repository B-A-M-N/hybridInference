"""Regression tests for the bounded hand-off queue in
``_streaming_response_with_keepalive`` (forced-stream buffering).

Before the fix the hand-off queue was unbounded: with the consumer stalled
(slow/stopped client applying transport backpressure), the upstream reader
kept buffering every chunk — a second full copy of the generation in
memory, growing for as long as the client stalls. Reproduced pre-fix:
2000-chunk upstream + 0.3 s consumer stall left 1941 chunks queued.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from serving.servers.routers import completions as comp


def _chunk(text: str) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "m",
                "choices": [{"delta": {"content": text}}],
            }
        )
        + "\n\n"
    )


async def _fast_upstream(count: int, close_flag: list[bool] | None = None):
    try:
        for _ in range(count):
            await asyncio.sleep(0)  # let the event loop interleave reader/consumer
            yield _chunk("x")
    except GeneratorExit:
        if close_flag is not None:
            close_flag.append(True)
        raise


class _Headers(dict):
    pass


def _make_gen(upstream: Any) -> Any:
    return comp._streaming_response_with_keepalive(
        upstream, request_id="rid", model="m", request_headers=_Headers()
    )


async def _spy_queues(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.Queue]:
    """Capture every asyncio.Queue the function creates during the test."""
    captured: list[asyncio.Queue] = []
    real_queue = asyncio.Queue

    class SpyQueue(real_queue):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(comp.asyncio, "Queue", SpyQueue)
    return captured


@pytest.mark.asyncio
async def test_slow_consumer_bounds_queued_chunks(monkeypatch):
    """A stalled consumer must not leave the reader buffering without limit."""
    queues = await _spy_queues(monkeypatch)
    gen = _make_gen(_fast_upstream(2000))

    comp._FORCE_STREAMING_KEEPALIVE_S = 0.01  # type: ignore[attr-defined]
    try:
        first = await gen.__anext__()  # first keepalive byte
        assert first == b" "
        # Consumer stalls here (client stopped reading); upstream keeps
        # producing into the queue.
        await asyncio.sleep(0.3)

        q = queues[0]
        assert q.qsize() <= comp._KEEPALIVE_QUEUE_MAXSIZE

        # The bound is backpressure, not data loss: draining the generator
        # to completion still yields the whole generation in the final body.
        items: list[bytes] = [first]
        async for item in gen:
            items.append(item)
    finally:
        comp._FORCE_STREAMING_KEEPALIVE_S = 15  # type: ignore[attr-defined]

    body = json.loads(items[-1].decode())
    assert body["choices"][0]["message"]["content"] == "x" * 2000


@pytest.mark.asyncio
async def test_closing_while_reader_blocked_on_full_queue_does_not_hang(monkeypatch):
    """Closing the generator with the reader blocked on a full queue must end it.

    The old reader's ``finally: await queue.put(None)`` would block forever on
    a full bounded queue once the consumer was gone, leaking the task and the
    upstream connection.
    """
    await _spy_queues(monkeypatch)
    close_flag: list[bool] = []
    gen = _make_gen(_fast_upstream(5000, close_flag))

    comp._FORCE_STREAMING_KEEPALIVE_S = 0.01  # type: ignore[attr-defined]
    try:
        await gen.__anext__()
        await asyncio.sleep(0.1)  # fill the queue; reader blocks on put
    finally:
        comp._FORCE_STREAMING_KEEPALIVE_S = 15  # type: ignore[attr-defined]

    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    assert close_flag == [True]  # the upstream generator was closed, not leaked


@pytest.mark.asyncio
async def test_upstream_exception_still_reaches_the_consumer_with_bound(monkeypatch):
    """The error path through the bounded queue is unchanged."""
    await _spy_queues(monkeypatch)

    async def _fail_after_first():
        yield _chunk("hello")
        raise RuntimeError("upstream blew up")

    comp._FORCE_STREAMING_KEEPALIVE_S = 15  # type: ignore[attr-defined]
    # Run with the default keepalive window: the exception arrives well within
    # the first wait_for timeout, so the clean re-raise path is exercised.
    slow = _fail_after_first()

    async def run() -> None:
        agen = comp._streaming_response_with_keepalive(
            slow, request_id="rid", model="m", request_headers=_Headers()
        )
        # Nothing was yielded before the failure, so the clean re-raise path
        # surfaces from the very first item pulled.
        with pytest.raises(RuntimeError):
            await agen.__anext__()

    await asyncio.wait_for(run(), timeout=5.0)
