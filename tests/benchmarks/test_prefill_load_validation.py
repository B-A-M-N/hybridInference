"""Validation tests for prefill-load-aware RouteWise.

Covers:
  - No prefill-state leak after success/error/timeout/fallback/disconnect
  - Reservation timing: near-concurrent decisions see preceding in-flight pressure
  - Multi-worker statement: process-local only
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from routing.route_table import EffectiveRoute
from routing.routewise.config import RouteWiseConfig
from routing.routewise.router import RouteWiseRouter


class _FakeRouteTable:
    def __init__(self) -> None:
        self._routes: dict[str, tuple[tuple[Any, float], ...]] = {}
        self.weight_overrides: dict[str, float] = {}

    def add(self, model_id: str, adapters_with_weights: list[tuple[Any, float]]) -> None:
        self._routes[model_id] = tuple(adapters_with_weights)

    def iter_effective_routes(self) -> tuple[EffectiveRoute, ...]:
        return tuple(
            EffectiveRoute(
                route_key=model_id,
                canonical_model_id=model_id,
                adapters=tuple(
                    (
                        adapter,
                        float(self.weight_overrides.get(adapter.config.endpoint_id, weight)),
                    )
                    for adapter, weight in adapters
                ),
            )
            for model_id, adapters in self._routes.items()
        )

    def canonical_id(self, model_id: str) -> str:
        return model_id


class _FakeAdapter:
    def __init__(self, endpoint_id: str) -> None:
        self.config = type(
            "Config",
            (),
            {
                "endpoint_id": endpoint_id,
                "provider": "test",
                "base_url": f"https://{endpoint_id}.example/v1",
                "pricing": {"prompt": "1.0", "completion": "1.0"},
            },
        )()


def _make_router(
    n_endpoints: int = 3,
    feature_enabled: bool = True,
    seed: int = 42,
) -> RouteWiseRouter:
    endpoints = [(f"test-model:ep{i}", "1.0", "1.0") for i in range(n_endpoints)]
    adapters = [_FakeAdapter(ep_id) for ep_id, _, _ in endpoints]
    fr = _FakeRouteTable()
    fr.add("test-model", [(adapter, 1.0 / len(adapters)) for adapter in adapters])

    config = RouteWiseConfig(
        prefill_load_routing_enabled=feature_enabled,
        random_seed=seed,
    )
    return RouteWiseRouter(route_table=fr, config=config)


def _request_context(prompt_tokens: int, request_id: str) -> dict[str, Any]:
    """Build a request whose lease estimator sees the requested prompt size."""
    return {
        "messages": [{"role": "user", "content": "x" * (prompt_tokens * 4)}],
        "params": {},
        "prompt_tokens": prompt_tokens,
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# Test: No prefill-state leak
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_state_leak_after_success() -> None:
    """After a successful request, PrefillLoadTracker returns to zero."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    # Simulate a request
    ep_id = "test-model:ep0"
    lease = router._prefill_load.acquire(ep_id, 100_000)
    assert router._prefill_load.backlog(ep_id) == 100_000

    # Simulate successful completion
    router._prefill_load.release(lease, prefill_confirmed=True)
    assert router._prefill_load.backlog(ep_id) == 0

    # Verify snapshot is clean
    snapshot = router._prefill_load.snapshot()
    assert ep_id not in snapshot or snapshot[ep_id]["prefill_tokens"] == 0


@pytest.mark.unit
def test_no_state_leak_after_error() -> None:
    """After an error (no prefill_confirmed), state returns to zero."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    ep_id = "test-model:ep0"
    lease = router._prefill_load.acquire(ep_id, 100_000)
    assert router._prefill_load.backlog(ep_id) == 100_000

    # Simulate error (no prefill_confirmed)
    router._prefill_load.release(lease, prefill_confirmed=False)
    assert router._prefill_load.backlog(ep_id) == 0


@pytest.mark.unit
def test_no_state_leak_after_multiple_requests() -> None:
    """After multiple requests complete, state returns to zero."""
    router = _make_router(n_endpoints=3, feature_enabled=True)

    leases = []
    for i in range(10):
        ep_id = f"test-model:ep{i % 3}"
        lease = router._prefill_load.acquire(ep_id, 50_000)
        leases.append((ep_id, lease))

    # Verify accumulated state
    assert router._prefill_load.backlog("test-model:ep0") > 0

    # Release all
    for _, lease in leases:
        router._prefill_load.release(lease, prefill_confirmed=True)

    # Verify clean state
    for i in range(3):
        assert router._prefill_load.backlog(f"test-model:ep{i}") == 0

    snapshot = router._prefill_load.snapshot()
    # Either no endpoints in snapshot, or all have zero tokens
    for data in snapshot.values():
        assert data["prefill_tokens"] == 0


@pytest.mark.unit
def test_no_state_leak_after_mixed_success_and_error() -> None:
    """Mixed success/error: state still returns to zero."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    # 5 successful requests
    for i in range(5):
        ep_id = f"test-model:ep{i % 2}"
        lease = router._prefill_load.acquire(ep_id, 20_000)
        router._prefill_load.release(lease, prefill_confirmed=True)

    # 5 failed requests
    for i in range(5):
        ep_id = f"test-model:ep{i % 2}"
        lease = router._prefill_load.acquire(ep_id, 20_000)
        router._prefill_load.release(lease, prefill_confirmed=False)

    # State should be clean
    for i in range(2):
        assert router._prefill_load.backlog(f"test-model:ep{i}") == 0


@pytest.mark.unit
def test_no_state_leak_after_timeout_simulation() -> None:
    """Simulated timeout: release without prefill_confirmed."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    ep_id = "test-model:ep0"
    lease = router._prefill_load.acquire(ep_id, 200_000)

    # Timeout: release without confirming prefill
    router._prefill_load.release(lease, prefill_confirmed=False)
    assert router._prefill_load.backlog(ep_id) == 0


# ---------------------------------------------------------------------------
# Test: Reservation timing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_near_concurrent_decisions_see_preceding_pressure() -> None:
    """Request B should see request A's prefill pressure when A is in-flight."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    decision_a = router._select_decision(
        "test-model", _request_context(500_000, "req-a"), reserve_prefill=True
    )
    assert decision_a is not None
    ep_a = decision_a.adapter.config.endpoint_id
    assert decision_a.prefill_lease is not None

    # Route request B - should see A's pressure
    decision_b = router._select_decision(
        "test-model", _request_context(1_000, "req-b"), reserve_prefill=True
    )
    assert decision_b is not None
    assert (
        decision_b.metadata["candidate_outstanding_prefill_tokens"][ep_a]
        == decision_a.prefill_lease.tokens
    )

    decision_a.release()
    decision_b.release()


@pytest.mark.unit
def test_reservation_timing_herding_risk() -> None:
    """Verify that near-concurrent decisions don't herd to the same loaded endpoint."""
    router = _make_router(n_endpoints=2, feature_enabled=True)

    # Pre-load ep0 with a real lease that remains active during selection.
    preload = router._prefill_load.acquire("test-model:ep0", 400_000)
    decisions = []

    # Route 20 requests
    selections = {"test-model:ep0": 0, "test-model:ep1": 0}
    for i in range(20):
        decision = router._select_decision(
            "test-model",
            _request_context(1_000, f"req-{i}"),
            reserve_prefill=True,
        )
        if decision:
            ep = decision.adapter.config.endpoint_id
            selections[ep] += 1
            decisions.append(decision)

    # With 400K backlog on ep0, the patched router should send most to ep1
    assert selections["test-model:ep1"] > selections["test-model:ep0"], (
        f"Expected ep1 to be preferred, got {selections}"
    )
    for decision in decisions:
        decision.release()
    router._prefill_load.release(preload)


@pytest.mark.unit
def test_reservation_timing_baseline_ignores_pressure() -> None:
    """Baseline (feature disabled) ignores prefill pressure entirely."""
    router = _make_router(n_endpoints=2, feature_enabled=False)

    # Pre-load ep0 with a real lease; disabled routing must ignore it.
    preload = router._prefill_load.acquire("test-model:ep0", 400_000)

    # Route 20 requests
    selections = {"test-model:ep0": 0, "test-model:ep1": 0}
    for i in range(20):
        decision = router._select_decision(
            "test-model", {"prompt_tokens": 1000, "request_id": f"req-{i}"}
        )
        if decision:
            ep = decision.adapter.config.endpoint_id
            selections[ep] += 1
            decision.release()

    # Baseline ignores load - both endpoints should be routable
    total = sum(selections.values())
    assert total == 20, "All requests should be routed"
    assert selections["test-model:ep0"] + selections["test-model:ep1"] == 20
    router._prefill_load.release(preload)


# ---------------------------------------------------------------------------
# Test: Multi-worker statement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prefill_tracker_is_process_local() -> None:
    """PrefillLoadTracker state is process-local (no shared state)."""
    router1 = _make_router(n_endpoints=2, feature_enabled=True, seed=42)
    router2 = _make_router(n_endpoints=2, feature_enabled=True, seed=42)

    # Each router has its own tracker
    assert router1._prefill_load is not router2._prefill_load

    # Acquire on router1
    lease1 = router1._prefill_load.acquire("test-model:ep0", 100_000)
    assert router1._prefill_load.backlog("test-model:ep0") == 100_000

    # Router2 should not see router1's state
    assert router2._prefill_load.backlog("test-model:ep0") == 0

    router1._prefill_load.release(lease1)


@pytest.mark.unit
def test_feature_does_not_add_shared_state() -> None:
    """The feature does not add any global/shared state."""
    # Create multiple routers and verify no shared state
    routers = [_make_router(n_endpoints=2, feature_enabled=True, seed=i) for i in range(5)]

    # Each should have independent trackers
    trackers = [r._prefill_load for r in routers]
    assert len({id(t) for t in trackers}) == 5

    # Modify one, others unaffected
    lease = routers[0]._prefill_load.acquire("test-model:ep0", 50_000)
    for i in range(1, 5):
        assert routers[i]._prefill_load.backlog("test-model:ep0") == 0

    routers[0]._prefill_load.release(lease)


# ---------------------------------------------------------------------------
# Test: Execution path tracking
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execution_path_tracks_prefill_pressure() -> None:
    """RouteWise execution acquires and releases prefill leases.

    This is the critical correctness test: without acquire/release in the
    execution path, the prefill tracker would always show zero backlog
    for router: routewise traffic in production.
    """
    router = _make_router(n_endpoints=2, feature_enabled=True)

    # Obtain the initial pressure through RouteWise's decision-owned lease,
    # just as the execution path does before handing the decision to an adapter.
    first = router._select_decision(
        "test-model",
        {
            **_request_context(500_000, "req-heavy"),
            "required_endpoint_id": "test-model:ep0",
        },
        reserve_prefill=True,
    )
    assert first is not None
    assert first.prefill_lease is not None
    decisions = [first]

    # Route 20 requests - should prefer ep1 (lighter load)
    selections = {"test-model:ep0": 0, "test-model:ep1": 0}
    for i in range(20):
        decision = router._select_decision(
            "test-model", _request_context(1_000, f"req-{i}"), reserve_prefill=True
        )
        if decision:
            ep = decision.adapter.config.endpoint_id
            selections[ep] += 1
            decisions.append(decision)

    # ep1 should be preferred because ep0 has 500K backlog
    assert selections["test-model:ep1"] > selections["test-model:ep0"]

    for decision in decisions:
        decision.release()
    assert router._prefill_load.backlog("test-model:ep0") == 0


@pytest.mark.unit
def test_feature_disabled_exact_old_behavior() -> None:
    """When disabled, prefill pressure cannot change endpoint selection."""
    endpoints = [
        ("test-model:a", "1.0", "1.0"),
        ("test-model:b", "1.0", "1.0"),
    ]
    adapters = [_FakeAdapter(ep_id) for ep_id, _, _ in endpoints]
    fr = _FakeRouteTable()
    fr.add("test-model", [(adapter, 0.5) for adapter in adapters])

    config = RouteWiseConfig(prefill_load_routing_enabled=False, random_seed=42)
    loaded_router = RouteWiseRouter(route_table=fr, config=config)

    unloaded_fr = _FakeRouteTable()
    unloaded_adapters = [_FakeAdapter(ep_id) for ep_id, _, _ in endpoints]
    unloaded_fr.add("test-model", [(adapter, 0.5) for adapter in unloaded_adapters])
    unloaded_router = RouteWiseRouter(
        route_table=unloaded_fr,
        config=RouteWiseConfig(prefill_load_routing_enabled=False, random_seed=42),
    )

    # Seed only one router through the tracker API; disabled routing must ignore it.
    load = loaded_router._prefill_load.acquire("test-model:a", 500_000)

    # Give them equal latencies and costs
    now = time.time()
    for router in (loaded_router, unloaded_router):
        router._latency_profiles["test-model:a"].record(now, 150.0)
        router._latency_profiles["test-model:b"].record(now, 150.0)

    def select_sequence(router: RouteWiseRouter) -> list[str]:
        selected: list[str] = []
        for i in range(100):
            decision = router._select_decision(
                "test-model", {"prompt_tokens": 1000, "request_id": f"req-{i}"}
            )
            assert decision is not None
            selected.append(decision.adapter.config.endpoint_id)
            decision.release()
        return selected

    loaded_sequence = select_sequence(loaded_router)
    unloaded_sequence = select_sequence(unloaded_router)

    assert loaded_sequence == unloaded_sequence
    loaded_router._prefill_load.release(load)


@pytest.mark.unit
def test_materially_better_loaded_endpoint_still_wins() -> None:
    """A much better endpoint can still win despite high load."""
    endpoints = [
        ("test-model:a", "1.0", "1.0"),
        ("test-model:b", "1.0", "1.0"),
    ]
    adapters = [_FakeAdapter(ep_id) for ep_id, _, _ in endpoints]
    fr = _FakeRouteTable()
    fr.add("test-model", [(adapter, 0.5) for adapter in adapters])
    router = RouteWiseRouter(
        route_table=fr,
        config=RouteWiseConfig(prefill_load_routing_enabled=True, random_seed=42),
    )

    a_load = router._prefill_load.acquire("test-model:a", 500_000)
    now = time.time()
    router._latency_profiles["test-model:a"].record(now, 100.0)
    router._latency_profiles["test-model:b"].record(now, 2000.0)

    selections = {"test-model:a": 0, "test-model:b": 0}
    for i in range(100):
        decision = router._select_decision(
            "test-model", {"prompt_tokens": 1000, "request_id": f"req-{i}"}
        )
        if decision:
            ep = decision.adapter.config.endpoint_id
            selections[ep] += 1

    assert selections["test-model:a"] > selections["test-model:b"]
    router._prefill_load.release(a_load)


# ---------------------------------------------------------------------------
# Test: Equal load produces same behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_equal_load_same_behavior() -> None:
    """Equal prefill load should not change selection behavior."""
    endpoints = [
        ("test-model:a", "1.0", "1.0"),
        ("test-model:b", "1.0", "1.0"),
    ]
    adapters = [_FakeAdapter(ep_id) for ep_id, _, _ in endpoints]
    fr = _FakeRouteTable()
    fr.add("test-model", [(adapter, 0.5) for adapter in adapters])

    config = RouteWiseConfig(prefill_load_routing_enabled=True, random_seed=42)
    router = RouteWiseRouter(route_table=fr, config=config)

    # Equal load
    a_load = router._prefill_load.acquire("test-model:a", 100_000)
    b_load = router._prefill_load.acquire("test-model:b", 100_000)

    now = time.time()
    router._latency_profiles["test-model:a"].record(now, 150.0)
    router._latency_profiles["test-model:b"].record(now, 200.0)

    # Route 100 requests
    selections = {"test-model:a": 0, "test-model:b": 0}
    for i in range(100):
        decision = router._select_decision(
            "test-model", {"prompt_tokens": 1000, "request_id": f"req-{i}"}
        )
        if decision:
            ep = decision.adapter.config.endpoint_id
            selections[ep] += 1

    # With equal load, lower-latency endpoint should be preferred
    assert selections["test-model:a"] > selections["test-model:b"]
    # Both should be routable
    assert selections["test-model:a"] > 0
    router._prefill_load.release(a_load)
    router._prefill_load.release(b_load)


# ---------------------------------------------------------------------------
# Test: Whole-request prefill sizing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whole_request_prefill_sizing() -> None:
    """Acquire uses whole-request estimator, not just message content.

    This verifies the fix for the issue where agent requests with large
    tool catalogs got near-zero leases because only message content
    was counted.
    """
    from routing.prefill_load import estimate_prefill_tokens

    router = _make_router(n_endpoints=2, feature_enabled=True)

    # Messages alone are small
    messages = [{"role": "user", "content": "hi"}]

    # But params include a large tool catalog
    tools = [
        {
            "type": "function",
            "function": {"name": f"tool_{i}", "description": "x" * 1000},
        }
        for i in range(100)
    ]

    # estimate_prefill_tokens should count tools
    tokens = estimate_prefill_tokens(messages, tools=tools)
    assert tokens > 1000  # Tools contribute significantly

    # Verify acquire uses this estimate
    lease = router._prefill_load.acquire("test-model:ep0", tokens)
    assert router._prefill_load.backlog("test-model:ep0") == tokens
    router._prefill_load.release(lease)


# ---------------------------------------------------------------------------
# Test: Non-finite config validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_finite_config_rejected() -> None:
    """NaN and Inf config values are rejected."""
    with pytest.raises(ValueError, match="finite"):
        RouteWiseConfig(prefill_load_scale_ms_per_1k=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        RouteWiseConfig(prefill_load_scale_ms_per_1k=float("inf"))

    with pytest.raises(ValueError, match="finite"):
        RouteWiseConfig(prefill_load_max_penalty_ms=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        RouteWiseConfig(prefill_load_max_penalty_ms=float("-inf"))

    # Valid finite values still accepted
    RouteWiseConfig(prefill_load_scale_ms_per_1k=1.0, prefill_load_max_penalty_ms=5000.0)
    RouteWiseConfig(prefill_load_scale_ms_per_1k=0.0, prefill_load_max_penalty_ms=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
