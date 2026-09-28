"""Runtime serving regressions for the #1337 routing handoff."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI, Request
from httpx import ASGITransport, AsyncClient

from routing.executor import RouteExecutor
from routing.prefill_load import (
    PRIORITY_ELEPHANT,
    PRIORITY_INTERACTIVE,
    PRIORITY_LARGE,
)
from serving.adapters.base import BaseAdapter, ModelConfig
from serving.servers.deps import AppServices
from serving.servers.middleware.error import install_error_handlers
from serving.servers.routers import completions
from serving.utils.context import notify_traffic_admitted
from serving.utils.traffic_classifier import TrafficClass, TrafficClassification
from serving.utils.traffic_state import get_traffic_observation_state

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _CompletionsTestAdapter(BaseAdapter):
    async def chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        return self.format_response(content="ok", model=self.config.id)

    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncGenerator[str, None]:
        if False:  # pragma: no cover - the integration case is non-streaming
            yield ""


class _StreamingCompletionsTestAdapter(_CompletionsTestAdapter):
    async def stream_chat_completion(
        self, messages: list[dict[str, Any]], **params: Any
    ) -> AsyncGenerator[str, None]:
        yield self.format_stream_chunk(content="ok", model=self.config.id)


class _IgnoringAdmissionRouter(RouteExecutor):
    """Typed-options router that intentionally ignores admission callbacks."""

    async def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        routing_options: Any = None,
        **params: Any,
    ) -> dict[str, Any]:
        return await super().chat_completion(
            model_id,
            messages,
            routing_options=None,
            **params,
        )

    def stream_chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        routing_options: Any = None,
        **params: Any,
    ) -> Any:
        return super().stream_chat_completion(
            model_id,
            messages,
            routing_options=None,
            **params,
        )


@pytest_asyncio.fixture
async def traffic_completions_app(mock_db_logger, mock_log_store, monkeypatch):
    """Self-contained app fixture; sibling test-module fixtures are not global."""
    monkeypatch.setenv("USER_AUTH_ENABLED", "0")
    from serving.config.runtime_settings import get_runtime_settings
    from serving.config.settings import get_settings
    from serving.servers.concurrency import enforce_user_concurrency
    from serving.servers.deps import (
        get_completions_logger,
        get_cost_tracker,
        get_log_store,
        get_model_router_registry,
        get_model_visibility_resolver,
        get_operational_store,
        get_pricing_lookup,
        get_router,
        get_services,
    )
    from serving.servers.middleware.request_id import RequestIdMiddleware
    from serving.servers.routers.completions_cost import CostTracker, PricingLookup
    from serving.servers.routers.completions_logging import CompletionsLogger

    get_settings.cache_clear()
    config = ModelConfig(
        id="gpt-4",
        name="gpt-4",
        provider="test",
        base_url="http://test",
        context_length=8192,
        max_output_length=4096,
        supported_params=["temperature", "top_p", "max_tokens"],
    )
    router = RouteExecutor()
    router.register_route("gpt-4", [(_CompletionsTestAdapter(config), 1.0)])
    app = FastAPI(title="Traffic Classifier Completions Test App")
    services = AppServices(
        router=router,
        db_logger=mock_db_logger,
        log_store=mock_log_store,
    )
    services.completions_logger = CompletionsLogger(
        log_store=mock_log_store,
        model_router_registry=None,
    )
    services.pricing_lookup = PricingLookup(router=router)
    services.cost_tracker = CostTracker(
        op_store=services.operational_store,
        pricing=services.pricing_lookup,
    )
    app.state.services = services

    async def _get_services():
        return services

    async def _get_router():
        return services.router

    async def _get_log_store():
        return services.log_store

    async def _get_operational_store():
        return services.operational_store

    async def _get_model_router_registry():
        return services.model_router_registry

    async def _get_model_visibility_resolver():
        return services.model_visibility_resolver

    async def _get_runtime_settings():
        return services.runtime_settings

    async def _get_completions_logger():
        return services.completions_logger

    async def _get_pricing_lookup():
        return services.pricing_lookup

    async def _get_cost_tracker():
        return services.cost_tracker

    async def _pass_concurrency_gate():
        return None

    app.dependency_overrides.update(
        {
            get_services: _get_services,
            get_router: _get_router,
            get_log_store: _get_log_store,
            get_operational_store: _get_operational_store,
            get_model_router_registry: _get_model_router_registry,
            get_model_visibility_resolver: _get_model_visibility_resolver,
            get_runtime_settings: _get_runtime_settings,
            get_completions_logger: _get_completions_logger,
            get_pricing_lookup: _get_pricing_lookup,
            get_cost_tracker: _get_cost_tracker,
            enforce_user_concurrency: _pass_concurrency_gate,
        }
    )
    install_error_handlers(app)
    app.include_router(completions.router)
    app.add_middleware(RequestIdMiddleware)
    try:
        yield app
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def traffic_completions_client(traffic_completions_app):
    transport = ASGITransport(app=traffic_completions_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def traffic_anthropic_app(anthropic_test_app, anthropic_compat_router):
    """Use async dependency overrides to keep Messages tests self-contained."""
    from serving.servers.concurrency import enforce_user_concurrency
    from serving.servers.deps import (
        get_log_store,
        get_model_visibility_resolver,
        get_operational_store,
        get_router,
    )
    from serving.servers.middleware.request_id import RequestIdMiddleware

    async def _get_router():
        return anthropic_compat_router

    async def _get_log_store():
        return anthropic_test_app.state.services.log_store

    async def _get_operational_store():
        return anthropic_test_app.state.services.operational_store

    async def _get_model_visibility_resolver():
        return None

    async def _pass_concurrency_gate():
        return None

    anthropic_test_app.dependency_overrides.update(
        {
            get_router: _get_router,
            get_log_store: _get_log_store,
            get_operational_store: _get_operational_store,
            get_model_visibility_resolver: _get_model_visibility_resolver,
            enforce_user_concurrency: _pass_concurrency_gate,
        }
    )
    anthropic_test_app.add_middleware(RequestIdMiddleware)
    return anthropic_test_app


@pytest_asyncio.fixture
async def traffic_anthropic_client(traffic_anthropic_app):
    transport = ASGITransport(app=traffic_anthropic_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _auth() -> dict[str, str]:
    from tests.servers.conftest import ANTHROPIC_TEST_API_KEY

    return {"x-api-key": ANTHROPIC_TEST_API_KEY}


def _messages_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "glm-4.7",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _capture_upstream(monkeypatch):
    sent: dict[str, Any] = {}
    response = {
        "id": "chatcmpl-traffic-test",
        "object": "chat.completion",
        "model": "glm-4.7",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }

    async def fake_post(self, url, payload):
        sent.clear()
        sent.update(payload or {})
        return response

    from serving.adapters.openai_compat import OpenAICompatAdapter

    monkeypatch.setattr(OpenAICompatAdapter, "_post_with_pool", fake_post)
    return sent


def _human_classification() -> TrafficClassification:
    return TrafficClassification(
        automation_score=0.12,
        confidence=0.80,
        class_hint=TrafficClass.LIKELY_HUMAN,
        reasons=("slow_cadence", "session_continuity"),
    )


def _capture_traffic_timestamps(monkeypatch, arrival_timestamp: float):
    """Record preview/commit time while the ASGI middleware has a fixed clock."""
    from types import SimpleNamespace

    from serving.servers.middleware import request_id

    monkeypatch.setattr(
        request_id,
        "time",
        SimpleNamespace(monotonic=lambda: arrival_timestamp),
    )
    state = get_traffic_observation_state()
    preview_times: list[float | None] = []
    commit_times: list[float | None] = []
    original_preview = state.preview_request
    original_record = state.record_request

    def capture_preview(*args: Any, **kwargs: Any) -> dict[str, Any]:
        preview_times.append(kwargs.get("observed_at"))
        return original_preview(*args, **kwargs)

    def capture_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        commit_times.append(kwargs.get("observed_at"))
        return original_record(*args, **kwargs)

    monkeypatch.setattr(state, "preview_request", capture_preview)
    monkeypatch.setattr(state, "record_request", capture_record)
    return preview_times, commit_times


@pytest.fixture
def no_log_store(monkeypatch):
    from serving.servers.routers import anthropic_messages as amod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        amod,
        "_schedule_log_store_task",
        lambda log_store, **kwargs: captured.update(kwargs),
    )
    return captured


@pytest.fixture
def priority_route(anthropic_compat_router):
    adapter, _weight = anthropic_compat_router.routes["glm-4.7"].adapters[0]
    original = adapter.config.priority_scheduling
    adapter.config.priority_scheduling = True
    yield adapter
    adapter.config.priority_scheduling = original


@pytest.fixture(autouse=True)
def reset_traffic_state():
    state = get_traffic_observation_state()
    state.reset()
    yield
    state.reset()


@pytest.mark.asyncio
async def test_messages_runtime_classifier_preserves_normal_dispatch(
    traffic_anthropic_client,
    monkeypatch,
    no_log_store,
    priority_route,
):
    """The real Messages handler consumes the classifier and still returns 200."""
    from serving.servers.routers import anthropic_messages as amod

    sent = _capture_upstream(monkeypatch)
    monkeypatch.setattr(amod, "classify_traffic", lambda evidence: _human_classification())

    response = await traffic_anthropic_client.post(
        "/v1/messages", json=_messages_body(), headers=_auth()
    )

    assert response.status_code == 200
    assert sent["priority"] == PRIORITY_INTERACTIVE
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_messages_keep_asgi_arrival_time_through_admitted_dispatch(
    traffic_anthropic_app,
    traffic_anthropic_client,
    monkeypatch,
    no_log_store,
):
    """Messages keeps the middleware timestamp through preview and admission."""
    from serving.servers.deps import get_router
    from serving.utils.context import REQUEST_ARRIVAL_TIMESTAMP

    arrival_timestamp = 3210.25
    preview_times, commit_times = _capture_traffic_timestamps(monkeypatch, arrival_timestamp)
    dependency_times: list[float | None] = []
    router_override = traffic_anthropic_app.dependency_overrides[get_router]

    async def observe_middleware_timestamp(request: Request):
        dependency_times.append(getattr(request.state, REQUEST_ARRIVAL_TIMESTAMP, None))
        return await router_override()

    traffic_anthropic_app.dependency_overrides[get_router] = observe_middleware_timestamp
    _capture_upstream(monkeypatch)

    response = await traffic_anthropic_client.post(
        "/v1/messages",
        json=_messages_body(),
        headers=_auth(),
    )

    assert response.status_code == 200
    assert dependency_times == [arrival_timestamp]
    assert preview_times == [arrival_timestamp]
    assert commit_times == [arrival_timestamp]


@pytest.mark.asyncio
async def test_messages_human_hint_stays_within_large_prefill_tier(
    traffic_anthropic_client,
    monkeypatch,
    no_log_store,
    priority_route,
):
    from serving.servers.routers import anthropic_messages as amod

    sent = _capture_upstream(monkeypatch)
    monkeypatch.setattr(amod, "classify_traffic", lambda evidence: _human_classification())
    large_system = "x" * 240_000

    response = await traffic_anthropic_client.post(
        "/v1/messages",
        json=_messages_body(system=large_system),
        headers=_auth(),
    )

    assert response.status_code == 200
    assert sent["priority"] == PRIORITY_LARGE + 1
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_messages_human_hint_cannot_promote_elephant(
    traffic_anthropic_client,
    monkeypatch,
    no_log_store,
    priority_route,
):
    from serving.servers.routers import anthropic_messages as amod

    sent = _capture_upstream(monkeypatch)
    monkeypatch.setattr(amod, "classify_traffic", lambda evidence: _human_classification())
    elephant_system = "x" * 900_000

    response = await traffic_anthropic_client.post(
        "/v1/messages",
        json=_messages_body(system=elephant_system),
        headers=_auth(),
    )

    assert response.status_code == 200
    assert sent["priority"] == PRIORITY_ELEPHANT
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_messages_trusted_synthetic_probe_does_not_update_traffic_state(
    traffic_anthropic_app,
    traffic_anthropic_client,
):
    """Deployment probes must not become authenticated Messages history."""
    from serving.servers.deps import get_router

    router = RouteExecutor()
    config = ModelConfig(
        id="glm-4.7",
        name="GLM-4.7",
        provider="test",
        base_url="http://test",
        max_output_length=1024,
        supported_params=["max_tokens"],
    )
    router.register_route("glm-4.7", [(_CompletionsTestAdapter(config), 1.0)])
    original_router_override = traffic_anthropic_app.dependency_overrides[get_router]

    async def use_test_router():
        return router

    traffic_anthropic_app.dependency_overrides[get_router] = use_test_router
    state = get_traffic_observation_state()

    try:
        response = await traffic_anthropic_client.post(
            "/v1/messages",
            json=_messages_body(),
            headers={**_auth(), "X-Probe": "synthetic"},
        )
    finally:
        traffic_anthropic_app.dependency_overrides[get_router] = original_router_override

    assert response.status_code == 200
    assert state.get_identity_count() == 0


@pytest.mark.asyncio
async def test_messages_rejected_model_does_not_update_traffic_state(
    traffic_anthropic_client,
):
    """Model rejection happens before behavioral state mutation."""
    state = get_traffic_observation_state()

    response = await traffic_anthropic_client.post(
        "/v1/messages",
        json=_messages_body(model="not-published"),
        headers=_auth(),
    )

    assert response.status_code == 404
    assert state.get_identity_count() == 0


@pytest.mark.asyncio
async def test_messages_lost_half_open_probe_does_not_update_traffic_state(
    traffic_anthropic_client,
    anthropic_compat_router,
):
    """A request rejected by a lost probe race cannot advance user history."""
    from routing.endpoint_health import _CircuitState
    from routing.endpoints import endpoint_id_for_adapter

    state = get_traffic_observation_state()
    registry = anthropic_compat_router.endpoint_health_registry
    adapter, _weight = anthropic_compat_router.routes["claude-opus-4.7"].adapters[0]
    endpoint_id = endpoint_id_for_adapter(adapter)

    for _ in range(20):
        registry.record_failure(endpoint_id, reason="seeded")
        if registry.snapshot()[endpoint_id]["circuit_state"] == _CircuitState.OPEN:
            break
    else:  # pragma: no cover - defensive
        raise AssertionError("circuit never opened")
    circuit = registry._circuits[endpoint_id]
    circuit.last_opened -= circuit.cooldown_seconds + 1

    entered = asyncio.Event()
    release = asyncio.Event()
    original_messages = adapter.messages

    async def hold_messages(body, **kwargs):
        # This double represents the adapter-level admission callback. The
        # production Anthropic adapter fires it after its outbound slot and
        # response context are acquired; a custom adapter that omits it is
        # covered by the handler's post-success fallback instead.
        notify_traffic_admitted()
        entered.set()
        await release.wait()
        return {
            "id": "msg_probe_race",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    adapter.messages = hold_messages
    first = asyncio.create_task(
        traffic_anthropic_client.post(
            "/v1/messages",
            json=_messages_body(model="claude-opus-4.7"),
            headers=_auth(),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        identity_key = state._identity_key("user", "test-user")
        assert state._identities[identity_key].request_count == 1

        second = await traffic_anthropic_client.post(
            "/v1/messages",
            json=_messages_body(model="claude-opus-4.7"),
            headers=_auth(),
        )
        assert second.status_code == 503
        assert state._identities[identity_key].request_count == 1
    finally:
        release.set()
        first_response = await first
        adapter.messages = original_messages

    assert first_response.status_code == 200


@pytest.mark.asyncio
async def test_completions_pass_typed_traffic_hint_to_active_router(
    traffic_completions_client,
    traffic_completions_app,
    monkeypatch,
):
    """The real completions handler transports the generated hint to routing."""
    from serving.servers.routers import completions

    expected = TrafficClassification(
        automation_score=0.34,
        confidence=0.80,
        class_hint=TrafficClass.UNKNOWN,
        reasons=("mixed_evidence",),
    )
    monkeypatch.setattr(completions, "classify_traffic", lambda evidence: expected)

    active_router = traffic_completions_app.state.services.router
    original_chat_completion = active_router.chat_completion
    observed: dict[str, Any] = {}

    async def spy_chat_completion(model_id, messages, **params):
        options = params.get("routing_options")
        observed["options"] = options
        return await original_chat_completion(model_id, messages, **params)

    monkeypatch.setattr(active_router, "chat_completion", spy_chat_completion)
    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    options = observed["options"]
    assert options.traffic_classification == "unknown"
    assert options.traffic_confidence == 0.80
    assert options.traffic_reasons == ("mixed_evidence",)


@pytest.mark.asyncio
async def test_completions_rejected_pin_does_not_update_traffic_state(
    traffic_completions_client,
    traffic_completions_app,
):
    """Preflight rejection cannot advance authenticated traffic history."""
    from serving.servers.routers.completions import verify_api_key

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-test-user",
            "role": "internal",
            "is_admin": True,
        }

    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key
    state = get_traffic_observation_state()

    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Route-Pin": "not-a-route"},
    )

    assert response.status_code == 400
    assert state.get_identity_count() == 0


@pytest.mark.asyncio
async def test_trusted_synthetic_probe_does_not_update_traffic_state(
    traffic_completions_client,
    traffic_completions_app,
):
    """Deployment probes must not become authenticated user history."""
    from serving.servers.routers.completions import verify_api_key

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-test-user",
            "role": "internal",
            "is_admin": True,
        }

    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key
    state = get_traffic_observation_state()

    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "probe"}]},
        headers={"X-Probe": "synthetic"},
    )

    assert response.status_code == 200
    assert state.get_identity_count() == 0


@pytest.mark.asyncio
async def test_completions_rejected_open_circuit_does_not_update_traffic_state(
    traffic_completions_client,
    traffic_completions_app,
):
    """A router admission failure cannot advance authenticated history."""
    from routing.endpoint_health import _CircuitState
    from routing.endpoints import endpoint_id_for_adapter
    from serving.servers.routers.completions import verify_api_key

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-test-user",
            "role": "internal",
            "is_admin": True,
        }

    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key
    router = traffic_completions_app.state.services.router
    adapter, _weight = router.routes["gpt-4"].adapters[0]
    endpoint_id = endpoint_id_for_adapter(adapter)
    registry = router.endpoint_health_registry
    for _ in range(20):
        registry.record_failure(endpoint_id, reason="seeded")
        if registry.snapshot()[endpoint_id]["circuit_state"] == _CircuitState.OPEN:
            break
    else:  # pragma: no cover - defensive
        raise AssertionError("circuit never opened")

    state = get_traffic_observation_state()
    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json=_messages_body(model="gpt-4"),
    )

    assert response.status_code == 503
    assert state.get_identity_count() == 0


@pytest.mark.asyncio
async def test_completions_admitted_upstream_timeout_records_traffic_history(
    traffic_completions_client,
    traffic_completions_app,
    monkeypatch,
):
    """An admitted request remains observable when its upstream times out."""
    from serving.servers.routers.completions import verify_api_key

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-test-user",
            "role": "internal",
            "is_admin": True,
        }

    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key
    adapter, _weight = traffic_completions_app.state.services.router.routes["gpt-4"].adapters[0]

    async def fail_upstream(messages, **params):
        notify_traffic_admitted()
        raise asyncio.TimeoutError("upstream timed out")

    monkeypatch.setattr(adapter, "chat_completion", fail_upstream)
    state = get_traffic_observation_state()

    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json=_messages_body(model="gpt-4"),
    )

    assert response.status_code == 500
    identity_key = state._identity_key("user", "traffic-test-user")
    assert state._identities[identity_key].request_count == 1


@pytest.mark.asyncio
async def test_chat_completions_keep_asgi_arrival_time_through_dispatch(
    traffic_completions_client,
    traffic_completions_app,
    monkeypatch,
):
    """Chat completions uses the ASGI-boundary time, not handler time."""
    from serving.servers.deps import get_router
    from serving.servers.routers.completions import verify_api_key
    from serving.utils.context import REQUEST_ARRIVAL_TIMESTAMP

    arrival_timestamp = 5678.5
    preview_times, commit_times = _capture_traffic_timestamps(monkeypatch, arrival_timestamp)
    dependency_times: list[float | None] = []

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-arrival-user",
            "role": "internal",
            "is_admin": True,
        }

    async def observe_middleware_timestamp(request: Request):
        dependency_times.append(getattr(request.state, REQUEST_ARRIVAL_TIMESTAMP, None))
        return traffic_completions_app.state.services.router

    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key
    traffic_completions_app.dependency_overrides[get_router] = observe_middleware_timestamp

    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert dependency_times == [arrival_timestamp]
    assert preview_times == [arrival_timestamp]
    assert commit_times == [arrival_timestamp]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_completions_fallback_records_when_typed_router_ignores_callback(
    traffic_completions_client,
    traffic_completions_app,
    stream,
):
    """Successful custom routers still record traffic without callback support."""
    from serving.servers.deps import get_router
    from serving.servers.routers.completions import verify_api_key

    async def fake_verify_api_key():
        return {
            "authenticated": True,
            "user_id": "traffic-test-user",
            "role": "internal",
            "is_admin": True,
        }

    config = ModelConfig(
        id="gpt-4",
        name="gpt-4",
        provider="test",
        base_url="http://test",
        context_length=8192,
        max_output_length=4096,
        supported_params=["temperature", "top_p", "max_tokens", "stream"],
    )
    router = _IgnoringAdmissionRouter()
    adapter_type = _StreamingCompletionsTestAdapter if stream else _CompletionsTestAdapter
    router.register_route("gpt-4", [(adapter_type(config), 1.0)])
    traffic_completions_app.dependency_overrides[verify_api_key] = fake_verify_api_key

    async def use_test_router():
        return router

    traffic_completions_app.dependency_overrides[get_router] = use_test_router

    response = await traffic_completions_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )

    assert response.status_code == 200
    identity_key = get_traffic_observation_state()._identity_key("user", "traffic-test-user")
    state = get_traffic_observation_state()
    assert state._identities[identity_key].request_count == 1


@pytest.mark.asyncio
async def test_concurrency_dependency_reports_admitted_count_and_releases():
    """The evidence count comes from the same admission slot lifecycle."""
    from serving.servers.auth import verify_api_key
    from serving.servers.concurrency import (
        UserConcurrencyLimiter,
        enforce_user_concurrency,
        static_limits_provider,
    )
    from serving.servers.deps import (
        get_embedding_adapters,
        get_model_concurrency_resolver,
        get_router,
        get_user_concurrency_limiter,
    )

    app = FastAPI()
    limiter = UserConcurrencyLimiter(static_limits_provider({"free": 2}))
    entered: list[int] = []
    release = asyncio.Event()

    async def fake_verify_api_key():
        return {"user_id": "traffic-test-user", "role": "free", "is_admin": False}

    async def fake_get_embedding_adapters() -> dict[str, Any]:
        return {}

    async def fake_get_limiter():
        return limiter

    async def no_router():
        return None

    async def no_concurrency_resolver():
        return None

    app.dependency_overrides[verify_api_key] = fake_verify_api_key
    app.dependency_overrides[get_user_concurrency_limiter] = fake_get_limiter
    app.dependency_overrides[get_router] = no_router
    app.dependency_overrides[get_model_concurrency_resolver] = no_concurrency_resolver
    app.dependency_overrides[get_embedding_adapters] = fake_get_embedding_adapters

    @app.get("/probe")
    async def probe(admitted_count: int | None = Depends(enforce_user_concurrency)):
        assert admitted_count is not None
        entered.append(admitted_count)
        await release.wait()
        return {"admitted_count": admitted_count}

    async def wait_for_entered(count: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while len(entered) < count:
            if loop.time() >= deadline:
                raise AssertionError(f"only {len(entered)}/{count} requests entered")
            await asyncio.sleep(0.005)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.get("/probe"))
        await wait_for_entered(1)
        second = asyncio.create_task(client.get("/probe"))
        await wait_for_entered(2)

        assert entered == [1, 2]
        assert limiter.in_flight("traffic-test-user") == 2
        release.set()
        first_response, second_response = await asyncio.gather(first, second)

    assert first_response.json()["admitted_count"] == 1
    assert second_response.json()["admitted_count"] == 2
    assert limiter.in_flight("traffic-test-user") == 0
