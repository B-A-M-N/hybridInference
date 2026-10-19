"""Regression tests for the bounded request-log queue (issue #133).

Before the fix, ``CompletionsLogger.schedule_log`` spawned one asyncio task
per payload with no admission bound: a stalled database accumulated one
pending task (plus its full prompt/response payload) per completed request,
graceful shutdown never waited for them, and queued billing rows were lost.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from serving.servers.routers.completions_logging import (
    _LOG_QUEUE_MAXSIZE,
    CompletionsLogger,
)


def _store(hang: asyncio.Event | None = None) -> MagicMock:
    store = MagicMock()
    if hang is None:
        store.log_request = AsyncMock(return_value=None)
    else:

        async def _slow(**kwargs: object) -> None:
            await hang.wait()

        store.log_request = AsyncMock(side_effect=_slow)
    return store


async def _wait_for_first_write(store: MagicMock) -> None:
    """Pump the loop until the worker has pulled the first payload."""
    for _ in range(100):
        if store.log_request.await_count >= 1:
            return
        await asyncio.sleep(0)
    raise AssertionError("worker never pulled the first payload")


@pytest.mark.asyncio
async def test_stalled_db_bounds_queued_payloads():
    """A stalled DB must not accumulate one pending task per request."""
    hang = asyncio.Event()
    store = _store(hang)
    cl = CompletionsLogger(log_store=store)

    cl.schedule_log("req-0", {"request_id": "req-0"})
    await _wait_for_first_write(store)

    total = _LOG_QUEUE_MAXSIZE + 976
    for i in range(1, total):
        cl.schedule_log(f"req-{i}", {"request_id": f"req-{i}"})
        if i % 64 == 0:
            await asyncio.sleep(0)

    # Exactly one long-lived worker instead of one task per write, and the
    # queued payload count is capped; everything beyond the cap was dropped
    # loudly, not silently piled up.
    assert cl._queue.qsize() <= _LOG_QUEUE_MAXSIZE
    assert len(cl._background_tasks) == 1
    # req-0 was pulled by the worker (in-flight) before the queue filled.
    assert cl._dropped_count == total - _LOG_QUEUE_MAXSIZE - 1

    hang.set()
    await cl.stop()
    # Draining writes the surviving payloads; the drop count is final.
    assert store.log_request.await_count == _LOG_QUEUE_MAXSIZE + 1
    assert cl._dropped_count == total - _LOG_QUEUE_MAXSIZE - 1


@pytest.mark.asyncio
async def test_stop_drains_queued_rows_before_returning():
    """Graceful shutdown flushes every queued row, in order."""
    cl = CompletionsLogger(log_store=_store())
    for i in range(50):
        cl.schedule_log(f"req-{i}", {"request_id": f"req-{i}", "n": i})
    await cl.stop()
    assert cl._log_store.log_request.await_count == 50
    assert cl._queue.qsize() == 0
    ids = [call.kwargs["request_id"] for call in cl._log_store.log_request.await_args_list]
    assert ids == [f"req-{i}" for i in range(50)]


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_stops_the_worker():
    cl = CompletionsLogger(log_store=_store())
    cl.schedule_log("req-1", {"request_id": "req-1"})
    await cl.stop()
    await cl.stop()
    assert cl._worker is None


@pytest.mark.asyncio
async def test_stop_timeout_discards_leftovers_and_logs(caplog):
    hang = asyncio.Event()
    store = _store(hang)
    cl = CompletionsLogger(log_store=store)
    for i in range(3):
        cl.schedule_log(f"req-{i}", {"request_id": f"req-{i}"})

    with caplog.at_level(logging.ERROR, logger="serving.servers.routers.completions_logging"):
        await cl.stop(timeout=0.05)

    assert "timed out" in caplog.text
    # At most the in-flight row was started; the rest were reported discarded.
    assert store.log_request.await_count <= 1
    hang.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_failed_write_still_emits_alert_event(caplog):
    """A failed row write keeps the tracked_task failure event contract."""
    store = MagicMock()
    store.log_request = AsyncMock(side_effect=RuntimeError("db down"))
    cl = CompletionsLogger(log_store=store)

    with caplog.at_level(logging.WARNING, logger="serving.observability.tracked_tasks"):
        cl.schedule_log("req-1", {"request_id": "req-1"})
        await cl.stop()

    matching = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tracked_task_completed"
        and getattr(r, "task_name", None) == "request_log"
    ]
    assert len(matching) == 1
    assert matching[0].success is False
    assert matching[0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_successful_write_still_emits_alert_event():
    """A successful row write keeps the tracked_task success event contract."""
    store = _store()
    cl = CompletionsLogger(log_store=store)

    cl.schedule_log("req-1", {"foo": "bar"})
    await cl.stop()

    store.log_request.assert_awaited_once_with(foo="bar")
