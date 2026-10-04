"""Regression: a transient startup stub must not poison prefix-locality memory.

Incident shape (2026-10-02, FreeInference /goal loop)
----------------------------------------------------
A provider answered HTTP 200 with a valid terminal SSE event whose only output
was a warmup notice::

    "The model is starting up - this takes about 120 seconds. Please wait..."

That string *contains text*, so ``not _is_empty_completion(...)`` was True and
the request was recorded as a success. Successes commit prefix locality, and
the commit path replaced the stored entry wholesale::

    self._entries[scope] = _Entry(blocks=new_blocks, ...)

A short transient response therefore overwrote the blocks describing a warm
~185k-token conversation. The next real request compared itself against the
stub's blocks, matched almost nothing, fell below ``min_match_tokens``, and
received no cache discount -- and because every later request rewrote the entry
from itself, the estimate stayed at zero for the whole ~87-request loop.

The defect is the conflation of *transport success* with *semantic progress*.
This file pins the required behavior:

* a non-progressing completion must not mutate positive locality evidence
* strong existing evidence must survive a weak or indeterminate observation
* the provider's own ``cached_input_tokens`` must reach the routing layer so
  "router expected warm / provider reported cold" is directly observable

The first test in ``TestTransientStubDoesNotPoisonLocality`` is a deliberate
control: it reproduces the pre-fix behavior and asserts the match collapses to
zero. If it ever passes differently, the fixture no longer represents the
incident and the assertions below it stop proving anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from routing.completion_outcome import CompletionOutcome
from routing.routers import RoutingObservation
from routing.routewise.prefix_cache import (
    Block,
    CacheScope,
    PrefixCacheCoordinator,
    SessionProviderPrefixMemory,
)
from routing.routewise.prefix_cache_pending import PendingPrefixCacheStore
from routing.routewise.router import RouteWiseRouter

SECRET = b"unit-test-secret"

WARMUP_STUB = "The model is starting up - this takes about 120 seconds. Please wait..."

_ENDPOINT = "p1:host:443"


def _chars(text: str) -> list[int]:
    return [ord(c) for c in text]


def _scope() -> CacheScope:
    return CacheScope(
        user_hash="u1",
        project_hash="proj1",
        session_hash="s1",
        provider_id="p1",
        endpoint_id=_ENDPOINT,
        model_profile="m1",
        key_slot_id="k1",
    )


def _blocks(*pairs: tuple[str, int]) -> tuple[Block, ...]:
    return tuple(Block(digest=digest, token_count=n) for digest, n in pairs)


def _coordinator(**kw) -> PrefixCacheCoordinator:
    min_match = kw.pop("min_match_tokens", 1)
    return PrefixCacheCoordinator(
        enabled=True,
        memory=SessionProviderPrefixMemory(min_match_tokens=min_match),
        block_size=8,
        secret=SECRET,
        tokenize=_chars,
        **kw,
    )


def _remember_with_observed_hit(
    coordinator: PrefixCacheCoordinator,
    scope: CacheScope,
    blocks: tuple[Block, ...],
    *,
    cached_tokens: int | None = None,
) -> None:
    """Establish remembered blocks backed by verified reuse evidence.

    Since #1417 a bare ``remember()`` only records *potential* warming; reaching
    ``VERIFIED_REUSABLE`` requires the provider to actually report cache reuse.
    The incident tests predate that model, so wherever they mean "strong
    existing evidence" they now establish it the way current dev requires.

    This keeps the invariant under test intact -- a transient stub must not
    destroy strong locality evidence -- without reverting #1417's rule that only
    an observed hit confirms reuse.
    """
    coordinator.remember(scope, blocks)
    # Default to the full block total so the established evidence bounds the
    # estimate exactly rather than one token short of it.
    total = cached_tokens if cached_tokens is not None else sum(b.token_count for b in blocks)
    coordinator.record_evidence(scope, total, blocks=blocks)


def _router_with_cache(**kw) -> SimpleNamespace:
    """Minimal object exposing exactly what the commit path touches."""
    router = SimpleNamespace(
        prefix_cache=_coordinator(**kw),
        pending_prefix_cache=PendingPrefixCacheStore(),
    )
    router.record_observation = lambda obs: RouteWiseRouter._commit_prefix_cache_observation(
        router, obs
    )
    # The commit path publishes telemetry through this helper; bind it the same
    # way so the harness exercises the real emission path.
    router._emit_prefix_evidence = lambda obs, scope, **kw: RouteWiseRouter._emit_prefix_evidence(
        router, obs, scope, **kw
    )
    router.warm_scope = lambda: router.prefix_cache.scope_for(
        session="s", provider_id="p", endpoint_id=_ENDPOINT, model_profile="m"
    )
    router.stash = lambda request_id, blocks: router.pending_prefix_cache.put(
        request_id, blocks, {_ENDPOINT: router.warm_scope()}
    )
    return router


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


class TestCompletionOutcome:
    def test_textual_warmup_stub_is_not_progress(self):
        outcome = CompletionOutcome.from_response(
            content=WARMUP_STUB, usage=None, http_status=200, terminal=True
        )
        assert outcome is CompletionOutcome.TRANSIENT_NO_PROGRESS

    def test_real_answer_is_progress(self):
        outcome = CompletionOutcome.from_response(
            content="I inspected the handler and applied the fix.",
            usage=None,
            http_status=200,
            terminal=True,
        )
        assert outcome is CompletionOutcome.PROGRESS

    def test_http_error_is_provider_error(self):
        outcome = CompletionOutcome.from_response(
            content=None, usage=None, http_status=503, terminal=True
        )
        assert outcome is CompletionOutcome.PROVIDER_ERROR

    def test_empty_body_is_empty(self):
        outcome = CompletionOutcome.from_response(
            content=None, usage=None, http_status=200, terminal=True
        )
        assert outcome is CompletionOutcome.EMPTY

    def test_non_terminal_is_aborted(self):
        outcome = CompletionOutcome.from_response(
            content=None, usage=None, http_status=200, terminal=False
        )
        assert outcome is CompletionOutcome.ABORTED

    def test_unknown_transient_wording_is_also_transient(self):
        """
        The architecture must not depend on one provider's exact wording.
        Registering an unknown-but-repeating notice works the same way.
        """
        from routing.completion_outcome import register_transient_marker

        register_transient_marker(r"synthetic queue notice \d+")
        outcome = CompletionOutcome.from_response(
            content="SYNTHETIC QUEUE NOTICE 42 - retry shortly",
            usage={"output_tokens": 12},
            http_status=200,
            terminal=True,
        )
        assert outcome is CompletionOutcome.TRANSIENT_NO_PROGRESS

    def test_outcome_admits_real_work(self):
        assert CompletionOutcome.PROGRESS.admits_real_work is True
        assert CompletionOutcome.COMPLETE.admits_real_work is True
        for bad in (
            CompletionOutcome.TRANSIENT_NO_PROGRESS,
            CompletionOutcome.EMPTY,
            CompletionOutcome.REPEATED_NOOP,
            CompletionOutcome.PROVIDER_ERROR,
            CompletionOutcome.ABORTED,
            CompletionOutcome.UNKNOWN,
        ):
            assert bad.admits_real_work is False, bad


# ---------------------------------------------------------------------------
# The core regression
# ---------------------------------------------------------------------------


class TestTransientStubDoesNotPoisonLocality:
    def test_unguarded_write_is_what_caused_the_incident(self):
        """
        Control: the pre-fix commit path called observe() unconditionally, and
        that alone collapsed the match to zero. If this stops holding, the
        fixture no longer reproduces the incident.
        """
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096), ("b", 4096))
        coordinator.remember(scope, conversation)
        coordinator.memory.observe(scope, _blocks(("stub", 8)))
        signal = coordinator.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        assert signal.matched_prefix_tokens == 0, "fixture no longer reproduces the incident"

    def test_stub_text_must_not_replace_strong_evidence(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096), ("b", 4096), ("c", 4096))
        _remember_with_observed_hit(coordinator, scope, conversation)

        outcome = CompletionOutcome.from_response(
            content=WARMUP_STUB, usage=None, http_status=200, terminal=True
        )
        assert outcome is CompletionOutcome.TRANSIENT_NO_PROGRESS

        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("stub", 8)),
            outcome=outcome,
            observed_cached_tokens=0,
        )

        after = coordinator.lookup_state(scope)
        assert after.blocks == conversation
        assert after.verified is True
        signal = coordinator.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        assert signal.matched_prefix_tokens == 3 * 4096

    def test_next_real_request_still_gets_full_discount(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096), ("b", 4096))
        _remember_with_observed_hit(coordinator, scope, conversation)
        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("stub", 8)),
            outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
            observed_cached_tokens=0,
        )
        signal = coordinator.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        assert signal.matched_prefix_tokens == 2 * 4096
        # The whole conversation must still be eligible. Expected tokens are
        # observed tokens scaled by decaying confidence, so the real invariant is
        # "essentially the full observed amount", not strict equality.
        assert signal.expected_cached_tokens == pytest.approx(2 * 4096, rel=1e-3)
        assert signal.cache_discount > 0.0

    def test_growing_conversation_after_stub_still_matches(self):
        """The /goal loop appends turns; locality must survive the stub."""
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096), ("b", 4096))
        coordinator.remember(scope, conversation)
        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("stub", 8)),
            outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
            observed_cached_tokens=0,
        )
        grown = _blocks(("a", 4096), ("b", 4096), ("c", 512))
        signal = coordinator.evaluate(scope, grown, cold_cost=1.0, price_delta=0.5)
        assert signal.matched_prefix_tokens == 2 * 4096


class TestEvidenceIsNotWholesaleReplaced:
    def test_progress_advances_evidence(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        first = _blocks(("a", 4096))
        coordinator.remember(scope, first)
        second = _blocks(("a", 4096), ("b", 4096))
        coordinator.remember(scope, second)
        assert coordinator.lookup_state(scope).blocks == second

    def test_unknown_outcome_preserves_existing_evidence(self):
        """An unrecognized outcome must not be treated as ground truth."""
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096))
        coordinator.remember(scope, conversation)
        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("other", 16)),
            outcome=CompletionOutcome.UNKNOWN,
            observed_cached_tokens=None,
        )
        assert coordinator.lookup_state(scope).blocks == conversation

    def test_non_progressing_outcome_refreshes_liveness(self):
        """
        A provider hiccup must not age strong evidence out via TTL. The entry
        stays, and stays verified -- only its timestamp moves.
        """
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096))
        _remember_with_observed_hit(coordinator, scope, conversation)
        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("stub", 8)),
            outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
            observed_cached_tokens=0,
        )
        state = coordinator.lookup_state(scope)
        assert state.present is True
        assert state.blocks == conversation
        assert state.verified is True

    def test_provider_error_preserves_existing_evidence(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096))
        coordinator.remember(scope, conversation)
        coordinator.record_outcome(
            scope=scope,
            blocks=_blocks(("err", 4)),
            outcome=CompletionOutcome.PROVIDER_ERROR,
            observed_cached_tokens=None,
        )
        assert coordinator.lookup_state(scope).blocks == conversation


# ---------------------------------------------------------------------------
# Provider cache telemetry
# ---------------------------------------------------------------------------


class TestProviderCacheTelemetry:
    def _obs(self, **kw) -> RoutingObservation:
        fields = {
            "model_id": "m",
            "endpoint_id": "e",
            "ttft_ms": None,
            "total_latency_ms": 1.0,
            "token_count": 10,
            "success": True,
        }
        fields.update(kw)
        return RoutingObservation(**fields)

    def test_observation_carries_observed_cached_tokens(self):
        obs = self._obs(observed_cached_tokens=185664)
        assert obs.observed_cached_tokens == 185664

    def test_default_is_none_not_zero(self):
        """Absent telemetry stays None; 0 means the provider said zero."""
        assert self._obs().observed_cached_tokens is None

    def test_observation_exposes_outcome(self):
        obs = self._obs(outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS)
        assert obs.outcome is CompletionOutcome.TRANSIENT_NO_PROGRESS

    def test_success_defaults_to_progress_for_back_compat(self):
        assert self._obs().outcome is CompletionOutcome.PROGRESS

    def test_failure_defaults_to_provider_error(self):
        assert self._obs(success=False).outcome is CompletionOutcome.PROVIDER_ERROR

    def test_expected_versus_observed_is_comparable(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096), ("b", 4096))
        _remember_with_observed_hit(coordinator, scope, conversation)
        signal = coordinator.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        assert signal.expected_cached_tokens > 0
        # A provider reporting zero against a positive expectation is the exact
        # ambiguity this telemetry exists to expose.
        record = coordinator.build_evidence_event(
            scope=scope,
            signal=signal,
            observed_cached_tokens=0,
            outcome=CompletionOutcome.PROGRESS,
            remember_written=True,
        )
        assert record["expected_cached_tokens"] > 0
        assert record["observed_cached_tokens"] == 0
        assert record["cache_prediction_mismatch"] is True

    def test_no_mismatch_when_provider_agrees(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        conversation = _blocks(("a", 4096))
        coordinator.remember(scope, conversation)
        signal = coordinator.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        record = coordinator.build_evidence_event(
            scope=scope,
            signal=signal,
            observed_cached_tokens=4096,
            outcome=CompletionOutcome.PROGRESS,
            remember_written=True,
        )
        assert record["cache_prediction_mismatch"] is False

    def test_evidence_event_carries_no_prompt_text(self):
        coordinator = _coordinator(min_match_tokens=1)
        scope = _scope()
        record = coordinator.build_evidence_event(
            scope=scope,
            signal=None,
            observed_cached_tokens=0,
            outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
            remember_written=False,
        )
        assert set(record) == {
            "scope_endpoint_id",
            "scope_model_profile",
            "scope_session_hash",
            "outcome",
            "admits_real_work",
            "matched_prefix_tokens",
            "expected_cached_tokens",
            "observed_cached_tokens",
            "cache_prediction_mismatch",
            "remember_written",
        }


# ---------------------------------------------------------------------------
# Router commit path
# ---------------------------------------------------------------------------


class TestRouterCommitPath:
    def _observation(self, **kw) -> RoutingObservation:
        fields = {
            "model_id": "m",
            "endpoint_id": _ENDPOINT,
            "ttft_ms": 10.0,
            "total_latency_ms": 20.0,
            "token_count": 10,
            "success": True,
            "request_id": "req-1",
        }
        fields.update(kw)
        return RoutingObservation(**fields)

    def test_transient_outcome_does_not_warm_locality(self):
        """
        The gate itself: a transport-successful but non-progressing observation
        must leave the stored entry byte-identical. Asserts committed evidence,
        not merely that the stash was consumed, so removing the gate fails this.
        """
        router = _router_with_cache(min_match_tokens=1)
        scope = router.warm_scope()
        conversation = _blocks(("a", 4096), ("b", 4096))
        router.prefix_cache.remember(scope, conversation)
        router.stash("req-1", _blocks(("stub", 8)))
        router.record_observation(
            self._observation(
                success=True,
                outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
            )
        )
        assert router.prefix_cache.lookup_state(scope).blocks == conversation
        signal = router.prefix_cache.evaluate(scope, conversation, cold_cost=1.0, price_delta=0.5)
        assert signal.matched_prefix_tokens == 8192

    def test_progress_outcome_does_advance_evidence(self):
        """Counter-test: real work must still be able to update the entry."""
        router = _router_with_cache(min_match_tokens=1)
        scope = router.warm_scope()
        newer = _blocks(("a", 4096), ("b", 4096), ("c", 4096))
        router.prefix_cache.remember(scope, _blocks(("a", 4096)))
        router.stash("req-ok", newer)
        router.record_observation(
            self._observation(
                request_id="req-ok",
                success=True,
                outcome=CompletionOutcome.PROGRESS,
            )
        )
        assert router.prefix_cache.lookup_state(scope).blocks == newer

    def test_terminal_failure_does_not_create_locality(self):
        """
        A terminal failure must leave the cache as empty as it found it -- it
        must never manufacture a warm entry out of a failed request.
        """
        router = _router_with_cache(min_match_tokens=1)
        scope = router.warm_scope()
        router.stash("req-fail", _blocks(("a", 4096)))
        router.record_observation(
            self._observation(
                request_id="req-fail",
                success=False,
                terminal=True,
                outcome=CompletionOutcome.PROVIDER_ERROR,
            )
        )
        assert router.prefix_cache.lookup_state(scope).present is False

    def test_non_terminal_failure_keeps_stash_for_fallback_winner(self):
        """A hedge/fallback leg still needs the stash to warm later."""
        router = _router_with_cache(min_match_tokens=1)
        router.stash("req-h", _blocks(("a", 4096)))
        router.record_observation(
            self._observation(
                request_id="req-h",
                success=False,
                terminal=False,
                outcome=CompletionOutcome.PROVIDER_ERROR,
            )
        )
        assert "req-h" in router.pending_prefix_cache

    def test_evidence_event_emitted_on_both_paths(self):
        router = _router_with_cache(min_match_tokens=1)
        events: list[dict] = []
        router.prefix_cache.set_evidence_sink(events.append)
        # Existing evidence, so the non-commit path has something to preserve
        # and therefore something to report.
        scope = router.warm_scope()
        router.prefix_cache.remember(scope, _blocks(("a", 4096)))

        router.stash("r1", _blocks(("a", 4096)))
        router.record_observation(
            self._observation(
                request_id="r1",
                success=True,
                outcome=CompletionOutcome.TRANSIENT_NO_PROGRESS,
                observed_cached_tokens=0,
            )
        )
        assert events, "a structured event must be emitted on the non-commit path"

        router.stash("r2", _blocks(("a", 4096), ("b", 4096)))
        router.record_observation(
            self._observation(
                request_id="r2",
                success=True,
                outcome=CompletionOutcome.PROGRESS,
                observed_cached_tokens=4096,
            )
        )
        assert len(events) >= 2, "events must be emitted on both paths"
        outcomes = [e["outcome"] for e in events]
        assert CompletionOutcome.TRANSIENT_NO_PROGRESS.value in outcomes
        assert CompletionOutcome.PROGRESS.value in outcomes
