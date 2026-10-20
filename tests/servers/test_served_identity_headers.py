"""X-FI-* served-identity response headers (issue #1504).

The gateway resolves the served adapter, endpoint and provider before it
dispatches, but that identity never reached callers: ``owned_by`` is only a
provider label and the small-budget Anthropic reroute can serve a different
model than the one the client named. These tests pin the sanitized header set
on both inference surfaces, streaming and non-streaming, and the rule that
auto-selection's deliberate concealment of the served model extends to the
headers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import completions
from serving.stream import done_sentinel, make_final_usage_chunk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

NATIVE_MODEL = "claude-opus-4.7"
CHAT_MODEL = "glm-4.7"


def _anthropic_auth():
    from tests.servers.conftest import ANTHROPIC_TEST_API_KEY

    return {"x-api-key": ANTHROPIC_TEST_API_KEY}


def _anthropic_body(**overrides):
    body = {
        "model": NATIVE_MODEL,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


class DummyAdapter(BaseAdapter):
    async def chat_completion(self, messages: list[dict[str, Any]], **params) -> dict[str, Any]:
        resp = self.format_response(content="Test response", model=self.config.id)
        resp["_routing"] = {
            "provider": self.config.provider,
            "endpoint_id": self.config.endpoint_id,
            "base_url": self.config.base_url,
        }
        return resp

    async def stream_chat_completion(self, messages, **params):
        yield self.format_stream_chunk(model=self.config.id, content="Test ")
        yield make_final_usage_chunk(
            model=self.config.id, messages=messages, total_content="Test response"
        )
        yield done_sentinel()


def _chat_cfg(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        name=model_id,
        provider="zai",
        base_url="https://example-zai.test",
        chat_path="/chat/completions",
        endpoint_id=f"{model_id}:zai-api",
        context_length=8192,
        max_output_length=4096,
        supported_params=["temperature", "top_p", "max_tokens", "stream", "tools"],
    )


@pytest.fixture
def chat_app(monkeypatch, mock_db_logger, mock_log_store) -> FastAPI:
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")
    router = RouteExecutor()
    router.register_route(CHAT_MODEL, [(DummyAdapter(_chat_cfg(CHAT_MODEL)), 1.0)])

    app = FastAPI(title="Test Served Identity App")
    app.state.services = AppServices(
        router=router, db_logger=mock_db_logger, log_store=mock_log_store
    )
    install_error_handlers(app)
    app.include_router(completions.router)
    return app


@pytest.fixture
async def chat_client(chat_app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def anthropic_client(anthropic_test_app) -> AsyncClient:
    transport = ASGITransport(app=anthropic_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _patch_anthropic_upstream(monkeypatch, model: str = "claude-opus-4-7"):
    from serving.http import AsyncHTTPClient

    upstream_resp = {
        "id": "msg_hdr",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }

    async def fake_post(self, url, json=None, headers=None, timeout=None, retries=2):
        return upstream_resp

    monkeypatch.setattr(AsyncHTTPClient, "json_post_with_retry", fake_post)


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_response_carries_served_identity_headers(
    anthropic_client, anthropic_test_app, monkeypatch
):
    _patch_anthropic_upstream(monkeypatch)
    adapter = anthropic_test_app.state.services.router.routes[NATIVE_MODEL].adapters[0][0]
    # The route's registry-minted identity, as production configs carry it.
    adapter.config.endpoint_id = f"{NATIVE_MODEL}:anthropic-api"

    r = await anthropic_client.post(
        "/v1/messages", json=_anthropic_body(), headers=_anthropic_auth()
    )
    assert r.status_code == 200
    assert r.headers["x-fi-requested-model"] == NATIVE_MODEL
    assert r.headers["x-fi-served-model"] == NATIVE_MODEL
    assert r.headers["x-fi-served-endpoint"] == f"{NATIVE_MODEL}:anthropic-api"
    assert r.headers["x-fi-served-provider"] == adapter.config.provider


@pytest.mark.asyncio
async def test_messages_stream_carries_the_same_served_identity_headers(
    anthropic_client, anthropic_test_app, monkeypatch
):
    """Streaming and non-streaming expose the same identity (issue #1504)."""

    class _FakeContent:
        def iter_any(self):
            async def _chunks():
                yield (
                    b"event: message_start\n"
                    b'data: {"type":"message_start","message":{"id":"msg_h","model":"claude-opus-4-7",'
                    b'"role":"assistant","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}\n\n'
                    b"event: content_block_delta\n"
                    b'data: {"type":"content_block_delta","index":0,'
                    b'"delta":{"type":"text_delta","text":"hi"}}\n\n'
                    b"event: message_stop\n"
                    b'data: {"type":"message_stop"}\n\n'
                )

            return _chunks()

    class _FakeResp:
        status = 200
        content = _FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakeSession:
        def post(self, url, json=None, headers=None, timeout=None):
            return _FakeResp()

    async def fake_ensure_session(self):
        return _FakeSession()

    from serving.http import AsyncHTTPClient

    monkeypatch.setattr(AsyncHTTPClient, "_ensure_session", fake_ensure_session)

    adapter = anthropic_test_app.state.services.router.routes[NATIVE_MODEL].adapters[0][0]
    adapter.config.endpoint_id = f"{NATIVE_MODEL}:anthropic-api"
    async with anthropic_client.stream(
        "POST", "/v1/messages", json=_anthropic_body(stream=True), headers=_anthropic_auth()
    ) as r:
        assert r.status_code == 200
        assert r.headers["x-fi-requested-model"] == NATIVE_MODEL
        assert r.headers["x-fi-served-model"] == NATIVE_MODEL
        assert r.headers["x-fi-served-endpoint"] == adapter.config.endpoint_id
        assert r.headers["x-fi-served-provider"] == adapter.config.provider
        body = b"".join([chunk async for chunk in r.aiter_bytes()])
    assert b"message_stop" in body


# ---------------------------------------------------------------------------
# OpenAI /v1/chat/completions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_response_carries_served_identity_headers(chat_client):
    r = await chat_client.post(
        "/v1/chat/completions",
        json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-fi-requested-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-endpoint"] == f"{CHAT_MODEL}:zai-api"
    assert r.headers["x-fi-served-provider"] == "zai"
    assert r.headers["x-fi-served-endpoint"].count("http") == 0


@pytest.mark.asyncio
async def test_chat_completions_stream_carries_requested_identity(chat_client):
    """Adapter selection is lazy on the streaming path, so only pre-stream
    identity is promised there; the served endpoint stays in the logs."""
    r = await chat_client.post(
        "/v1/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["x-fi-requested-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-endpoint"] == f"{CHAT_MODEL}:zai-api"
    assert r.headers["x-fi-served-provider"] == "zai"


@pytest.mark.asyncio
async def test_forced_streaming_branch_carries_the_headers(chat_app):
    """The keepalive branch returns its StreamingResponse directly, so the
    headers must ride on that object — injected-parameter headers are not
    merged into a returned Response."""

    class _Forced:
        async def get_bool(self, key: str) -> bool:
            return key == "force_chat_completions_streaming"

    from serving.config.runtime_settings import get_runtime_settings

    chat_app.dependency_overrides[get_runtime_settings] = lambda: _Forced()

    transport = ASGITransport(app=chat_app)
    from httpx import AsyncClient as _AC

    async with _AC(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["x-fi-requested-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-endpoint"] == f"{CHAT_MODEL}:zai-api"
    assert r.headers["x-fi-served-provider"] == "zai"


@pytest.mark.asyncio
async def test_multi_backend_streaming_withholds_unresolved_served_fields(chat_app):
    """With two backends the serving adapter is chosen after the headers are
    on the wire, so the served endpoint/provider are withheld, not guessed."""

    class _SecondAdapter(DummyAdapter):
        pass

    second = _SecondAdapter(_chat_cfg(CHAT_MODEL + "-b"))
    chat_app.state.services.router.register_route(
        CHAT_MODEL,
        [(chat_app.state.services.router.routes[CHAT_MODEL].adapters[0][0], 0.5), (second, 0.5)],
    )
    transport = ASGITransport(app=chat_app)
    from httpx import AsyncClient as _AC

    async with _AC(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
    assert r.status_code == 200
    assert r.headers["x-fi-requested-model"] == CHAT_MODEL
    assert r.headers["x-fi-served-model"] == CHAT_MODEL
    assert "x-fi-served-endpoint" not in r.headers
    assert "x-fi-served-provider" not in r.headers


# ---------------------------------------------------------------------------
# Header sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint_id", "provider", "served_model", "expected"),
    [
        ("glm-4.6:local-12003", "sglang", "glm-4.6", "glm-4.6:local-12003"),
        ("glm-4.6:zai-api", "zai", "glm-4.6", "glm-4.6:zai-api"),
        ("glm-4.6:local", "vllm", "glm-4.6", "glm-4.6:local"),
        # LAN octet stamps, hostnames and admin route_ids never reach the
        # header: they degrade to the provider label.
        ("glm-4.6:10-api", "zai", "glm-4.6", "zai"),
        ("glm-4.6:gpu-7-api", "sglang", "glm-4.6", "sglang"),
        ("glm-4.6:my-internal-box", "zai", "glm-4.6", "zai"),
        ("glm-4.6:.local-api", "zai", "glm-4.6", "zai"),
        # A trailing newline must not smuggle an unsanitized shape through.
        ("glm-4.6:zai-api\n", "zai", "glm-4.6", "zai"),
        # A safe-looking location is not authority for the whole identifier:
        # only a prefix that IS the trusted served model may be republished.
        # "private.internal:local-12003" ends in a registry-minted location,
        # but the prefix must come from the registry-resolved served model,
        # not ride along with the location's shape.
        ("private.internal:local-12003", "zai", "glm-4.6", "zai"),
        ("private.internal:local-12003", "zai", "private.internal", "private.internal:local-12003"),
        # No trusted served model to anchor the prefix: degrade, fail closed.
        ("glm-4.6:local-12003", "sglang", None, "sglang"),
        # A colon-less id is not a registry construction at all: the old code
        # fullmatched the whole string as the candidate and published a bare
        # "local"; the anchor check requires the {model}:{location} shape.
        ("local", "vllm", "glm-4.6", "vllm"),
        (None, "zai", "glm-4.6", "zai"),
        (None, None, None, None),
        ("", "zai", "glm-4.6", "zai"),
    ],
)
def test_header_endpoint_id_sanitization(endpoint_id, provider, served_model, expected):
    from serving.servers.routers.served_identity import header_endpoint_id

    assert header_endpoint_id(endpoint_id, provider, served_model) == expected


def test_served_identity_headers_omit_missing_fields():
    from serving.servers.routers.served_identity import served_identity_headers

    assert served_identity_headers(requested_model="m") == {"X-FI-Requested-Model": "m"}
    headers = served_identity_headers(
        requested_model="auto",
        served_model="glm-4.6",
        endpoint_id="glm-4.6:zai-api",
        provider="zai",
    )
    assert headers == {
        "X-FI-Requested-Model": "auto",
        "X-FI-Served-Model": "glm-4.6",
        "X-FI-Served-Endpoint": "glm-4.6:zai-api",
        "X-FI-Served-Provider": "zai",
    }
