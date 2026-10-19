"""Embedding models honor the model-level access gate (required_role/admin_only).

``docs/developer/adding-models.md`` documents ``required_role`` as "the lowest
user role that can see and call the model" — a model-level field with no
chat-only qualifier. But embedding models never enter the weighted route table
where the chat surface reads that field: the registry's embedding branch
registered only the adapters, so the declaration was dead config and
/v1/embeddings served any authenticated caller regardless of the declared
tier. These tests pin the fix: the registry records the same fail-closed gate
it parses for chat models, and /v1/embeddings enforces it through the shared
``model_access_denial`` check (uniform 404, aliases held to the model's rules,
runtime visibility overrides applied).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from routing.routers import FixedRouter
from serving.config.runtime_settings import get_runtime_settings
from serving.servers.auth import verify_api_key
from serving.servers.concurrency import enforce_user_concurrency
from serving.servers.deps import (
    get_completions_logger,
    get_embedding_adapters,
    get_log_store,
    get_model_visibility_resolver,
    get_operational_store,
    get_router,
)
from serving.servers.registry import register_from_models_yaml
from serving.servers.routers import embeddings

# ---------------------------------------------------------------------------
# Registry: the embedding branch records the gate the chat branch parses.
# ---------------------------------------------------------------------------


def _write_models_yaml(tmp_path, body: str):
    p = tmp_path / "models.yaml"
    p.write_text(body)
    return p


def _embedding_yaml(*, role: str | None = "pro", route_extra: str = "") -> str:
    gate = f"    required_role: {role}\n" if role is not None else ""
    return f"""
models:
  - id: emb-x
    name: Embedding X
    provider: openai_compat
    model_type: embedding
{gate}    aliases: [emb-x-alias]
    route:
      - kind: openai_compat
        base_url: https://gateway.example/api/v2/
        api_keys: [pool-key]
        embeddings_path: /embeddings{route_extra}
"""


@pytest.mark.unit
def test_embedding_model_records_required_role_gate(tmp_path):
    p = _write_models_yaml(tmp_path, _embedding_yaml(role="pro"))

    exe = FixedRouter()
    count, _infos = register_from_models_yaml(exe, p, {})

    assert count == 2  # model + alias
    gate = exe.embedding_routes["emb-x"]
    assert gate.required_role == "pro"
    # The alias is held to the model's rules: same object, same gate.
    assert exe.embedding_routes["emb-x-alias"] is gate
    # And the model must NOT appear in the weighted chat table, where an
    # embedding adapter would be dispatchable as a chat route.
    assert "emb-x" not in exe.routes
    assert "emb-x-alias" not in exe.routes


@pytest.mark.unit
def test_embedding_model_without_role_defaults_free_like_chat(tmp_path):
    p = _write_models_yaml(tmp_path, _embedding_yaml(role=None))

    exe = FixedRouter()
    register_from_models_yaml(exe, p, {})

    assert exe.embedding_routes["emb-x"].required_role == "free"


@pytest.mark.unit
def test_embedding_model_invalid_role_fails_closed(tmp_path):
    p = _write_models_yaml(tmp_path, _embedding_yaml(role="god"))

    exe = FixedRouter()
    register_from_models_yaml(exe, p, {})

    assert exe.embedding_routes["emb-x"].required_role == "admin"


@pytest.mark.unit
def test_embedding_model_zero_weight_route_still_gated(tmp_path):
    """A zero-weight route must not skip the gate record.

    The weight table plays no role in embedding dispatch (the surface walks
    its adapter chain directly), so unlike register_route there is nothing
    to protect by ignoring zero-weight routes — only a gate to lose.
    """
    p = _write_models_yaml(tmp_path, _embedding_yaml(role="pro", route_extra="\n        weight: 0"))

    exe = FixedRouter()
    register_from_models_yaml(exe, p, {})

    gate = exe.embedding_routes["emb-x"]
    assert gate.required_role == "pro"
    assert "emb-x" not in exe.routes


# ---------------------------------------------------------------------------
# Surface: /v1/embeddings enforces the gate the way chat enforces its route's.
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self):
        self.config = SimpleNamespace(
            id="emb-pro",
            provider="fake-provider",
            pricing={"prompt": "1", "completion": "0"},
        )

    async def embeddings(self, input_data, **params):
        return {
            "object": "list",
            "model": "emb-pro",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }


class _CapturingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def schedule_log(self, request_id: str, log_data: dict[str, Any]) -> None:
        self.calls.append((request_id, log_data))


class _FakeRuntimeSettings:
    async def get_bool(self, key: str) -> bool:
        return False


class _StubVisibilityResolver:
    """Runtime override standing in for the admin visibility table."""

    def __init__(self, required: str) -> None:
        self._required = required

    async def get_effective_required_role(self, canonical: str, required: str) -> str:
        return self._required


def _gated_router(required_role: str = "pro") -> FixedRouter:
    """A router whose embedding_routes gates emb-pro (and its alias)."""
    exe = FixedRouter()
    exe.register_embedding_route(
        "emb-pro",
        [(_FakeAdapter(), 1.0)],
        aliases=["emb-pro-alias"],
        required_role=required_role,
    )
    return exe


def _build_app(
    router_exec,
    logger: _CapturingLogger,
    *,
    user_role: str = "free",
    visibility_resolver=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(embeddings.router)
    app.dependency_overrides[verify_api_key] = lambda: {
        "user_id": "user-emb",
        "role": user_role,
        "authenticated": True,
    }
    app.dependency_overrides[enforce_user_concurrency] = lambda: None
    app.dependency_overrides[get_router] = lambda: router_exec
    app.dependency_overrides[get_embedding_adapters] = lambda: {
        "emb-pro": _FakeAdapter(),
        "emb-pro-alias": _FakeAdapter(),
    }
    app.dependency_overrides[get_model_visibility_resolver] = lambda: visibility_resolver
    app.dependency_overrides[get_log_store] = lambda: object()  # truthy → logging enabled
    app.dependency_overrides[get_completions_logger] = lambda: logger
    app.dependency_overrides[get_operational_store] = lambda: None
    app.dependency_overrides[get_runtime_settings] = lambda: _FakeRuntimeSettings()
    return app


async def _post(app: FastAPI, model: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/embeddings", json={"model": model, "input": "hi"})
    await asyncio.sleep(0.05)
    return resp


@pytest.mark.asyncio
async def test_required_role_denies_free_caller():
    logger = _CapturingLogger()
    app = _build_app(_gated_router("pro"), logger, user_role="free")

    resp = await _post(app, "emb-pro")

    assert resp.status_code == 404
    # The denial is logged as a 404 row, the same shape as an unknown model.
    assert len(logger.calls) == 1
    assert logger.calls[0][1]["status_code"] == 404
    assert logger.calls[0][1]["error"] == "Embedding model 'emb-pro' not found"


@pytest.mark.asyncio
async def test_required_role_admits_sufficient_role():
    logger = _CapturingLogger()
    app = _build_app(_gated_router("pro"), logger, user_role="pro")

    resp = await _post(app, "emb-pro")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_alias_held_to_model_rules():
    app = _build_app(_gated_router("pro"), _CapturingLogger(), user_role="free")

    resp = await _post(app, "emb-pro-alias")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_only_gate_denies_free_caller():
    exe = FixedRouter()
    exe.register_embedding_route("emb-pro", [(_FakeAdapter(), 1.0)], admin_only=True)
    app = _build_app(exe, _CapturingLogger(), user_role="free")

    resp = await _post(app, "emb-pro")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_visibility_override_wins_over_baseline():
    app = _build_app(
        _gated_router("pro"),
        _CapturingLogger(),
        user_role="free",
        visibility_resolver=_StubVisibilityResolver("free"),
    )

    resp = await _post(app, "emb-pro")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_router_without_gate_keeps_serving():
    """A router predating the gate gates nothing — the old behavior."""

    class _LegacyRouter:
        def __init__(self) -> None:
            self.routes = {}

    logger = _CapturingLogger()
    app = _build_app(_LegacyRouter(), logger, user_role="free")

    resp = await _post(app, "emb-pro")

    assert resp.status_code == 200
