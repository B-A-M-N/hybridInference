"""Cache evidence must not be fabricated by the Responses translation.

The public Responses shape requires a numeric ``cached_tokens``, so an unreported
measurement is serialised as 0 for SDK compatibility. That compatibility zero is
not evidence: reading it as a measured miss would fabricate a cache result for
every provider that simply does not report one. These tests pin the three-state
contract (unreported / reported_miss / reported_hit) and prove the wire shape is
unchanged in both response modes.
"""

from __future__ import annotations

from typing import Any, ClassVar

from serving.responses_translator import (
    CACHE_EVIDENCE_HIT,
    CACHE_EVIDENCE_MISS,
    CACHE_EVIDENCE_UNREPORTED,
    ResponsesStreamTranslator,
    _cache_token_count,
    cache_evidence_state,
    chat_response_to_responses,
    map_usage_with_evidence,
)
from serving.utils.token_utils import normalize_usage


def _usage(**overrides):
    base = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    base.update(overrides)
    return base


class TestCacheEvidenceStates:
    def test_no_cache_fields_is_unreported_not_a_miss(self):
        assert cache_evidence_state(_usage()) == CACHE_EVIDENCE_UNREPORTED
        assert _cache_token_count(_usage()) == (None, CACHE_EVIDENCE_UNREPORTED)

    def test_explicit_zero_is_a_reported_miss(self):
        assert cache_evidence_state(_usage(cache_read_tokens=0)) == CACHE_EVIDENCE_MISS
        assert _cache_token_count(_usage(cache_read_tokens=0)) == (
            0,
            CACHE_EVIDENCE_MISS,
        )

    def test_positive_count_is_a_reported_hit(self):
        assert cache_evidence_state(_usage(cache_read_tokens=512)) == CACHE_EVIDENCE_HIT
        assert _cache_token_count(_usage(cache_read_tokens=512)) == (
            512,
            CACHE_EVIDENCE_HIT,
        )

    def test_nested_openai_shape_is_read(self):
        assert (
            cache_evidence_state(_usage(prompt_tokens_details={"cached_tokens": 768}))
            == CACHE_EVIDENCE_HIT
        )

    def test_missing_usage_block_is_unreported(self):
        assert cache_evidence_state(None) == CACHE_EVIDENCE_UNREPORTED

    def test_malformed_counts_match_the_logging_layer(self):
        """Unreadable counts resolve the same way the log resolves them.

        The logging layer sanitises before extracting, so a numeric string is
        read as a count while a boolean, NaN, negative or unparseable value is
        billed as 0 with a warning. The translator must not classify these
        differently, or one request reads as a miss in ``api_logs`` and as
        unreported in the Responses translation.
        """
        for bad in ("768", "768.0", True, -1, float("nan"), "abc", None):
            usage = _usage(cache_read_tokens=bad)
            count, state = _cache_token_count(usage)
            logged = (normalize_usage(usage) or {}).get("cache_read_tokens")
            assert count == logged, f"{bad!r}: {count} != {logged}"
            expected = (
                CACHE_EVIDENCE_UNREPORTED
                if logged is None
                else (CACHE_EVIDENCE_HIT if logged > 0 else CACHE_EVIDENCE_MISS)
            )
            assert state == expected, f"{bad!r}: {state} != {expected}"


class TestEvidenceAgreesWithLoggingLayer:
    """The translator and the ``api_logs`` layer must not disagree.

    Both read the same raw usage block, so they must classify it identically. The
    translator used to re-parse that block with a private three-field list, which
    classified any provider whose cache evidence arrives in another shape as
    ``unreported`` -- while ``api_logs`` recorded a measured hit. That divergence
    is what made a provider's cache behaviour look unobservable.
    """

    #: Every cache-read shape the logging layer recognises.
    SHAPES = (
        {"cache_read_tokens": 768},
        {"cached_tokens": 768},
        {"prompt_tokens_details": {"cached_tokens": 768}},
        {"prompt_cache_hit_tokens": 768},
        {"cache_read_input_tokens": 768},
        {"input_tokens_details": {"cached_tokens": 768}},
    )

    def test_deepseek_native_shape_is_a_hit(self):
        """DeepSeek reports ``prompt_cache_hit_tokens``, not the OpenAI shape."""
        usage = _usage(prompt_cache_hit_tokens=768)
        assert cache_evidence_state(usage) == CACHE_EVIDENCE_HIT
        assert _cache_token_count(usage) == (768, CACHE_EVIDENCE_HIT)

    def test_every_logging_shape_is_classified(self):
        for shape in self.SHAPES:
            assert cache_evidence_state(_usage(**shape)) == CACHE_EVIDENCE_HIT, shape

    def test_translator_matches_the_logging_layer(self):
        for shape in self.SHAPES:
            usage = _usage(**shape)
            logged = (normalize_usage(usage) or {}).get("cache_read_tokens")
            count, state = _cache_token_count(usage)
            assert count == logged, f"{shape}: {count} != {logged}"
            expected = (
                CACHE_EVIDENCE_UNREPORTED
                if logged is None
                else (CACHE_EVIDENCE_HIT if logged > 0 else CACHE_EVIDENCE_MISS)
            )
            assert state == expected, f"{shape}: {state} != {expected}"

    def test_unreported_matches_an_absent_logged_value(self):
        usage = _usage()
        assert (normalize_usage(usage) or {}).get("cache_read_tokens") is None
        assert cache_evidence_state(usage) == CACHE_EVIDENCE_UNREPORTED

    def test_reported_miss_matches_a_zero_logged_value(self):
        usage = _usage(prompt_cache_hit_tokens=0)
        assert (normalize_usage(usage) or {}).get("cache_read_tokens") == 0
        assert cache_evidence_state(usage) == CACHE_EVIDENCE_MISS

    def test_no_input_diverges_from_the_log(self):
        """Exhaustive agreement, well-formed and malformed shapes alike.

        Divergence here is silent: the same request would read as a cache miss in
        ``api_logs`` and as unreported in the Responses translation, which is
        exactly how a provider's caching looked unobservable.
        """
        shapes = (
            {},
            {"cache_read_tokens": 0},
            {"cache_read_tokens": 768},
            {"cache_read_tokens": "768"},
            {"cache_read_tokens": "768.0"},
            {"cache_read_tokens": True},
            {"cache_read_tokens": -1},
            {"cache_read_tokens": float("nan")},
            {"cache_read_tokens": "abc"},
            {"cache_read_tokens": None},
            {"cached_tokens": 768},
            {"prompt_cache_hit_tokens": 768},
            {"prompt_cache_hit_tokens": 0},
            {"prompt_cache_hit_tokens": True},
            {"prompt_cache_hit_tokens": -5},
            {"cache_read_input_tokens": 768},
            {"input_tokens_details": {"cached_tokens": 768}},
            {"prompt_tokens_details": {"cached_tokens": 768}},
            {"prompt_tokens_details": {"cached_tokens": 0}},
            {"prompt_tokens_details": None},
        )
        for shape in shapes:
            usage = _usage(**shape)
            count, state = _cache_token_count(usage)
            logged = (normalize_usage(usage) or {}).get("cache_read_tokens")
            assert count == logged, f"{shape}: {count} != {logged}"
            expected = (
                CACHE_EVIDENCE_UNREPORTED
                if logged is None
                else (CACHE_EVIDENCE_HIT if logged > 0 else CACHE_EVIDENCE_MISS)
            )
            assert state == expected, f"{shape}: {state} != {expected}"


class TestWireShapeStaysSdkCompatible:
    """Both response modes must keep the exact public shape."""

    EXPECTED: ClassVar[dict[str, Any]] = {
        "input_tokens": 9,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 12,
    }

    def test_unreported_still_serialises_zero(self):
        usage, state = map_usage_with_evidence(_usage())
        assert usage == self.EXPECTED
        assert state == CACHE_EVIDENCE_UNREPORTED

    def test_reported_miss_serialises_zero_and_stays_distinguishable(self):
        """The wire value is identical, so only the state tells them apart."""
        usage, state = map_usage_with_evidence(_usage(cache_read_tokens=0))
        assert usage == self.EXPECTED
        assert state == CACHE_EVIDENCE_MISS

    def test_reported_hit_serialises_the_count(self):
        usage, state = map_usage_with_evidence(_usage(cache_read_tokens=512))
        assert usage["input_tokens_details"] == {"cached_tokens": 512}
        assert state == CACHE_EVIDENCE_HIT

    def test_nonstreaming_response_shape_unchanged(self):
        chat = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "glm-4.7",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        }
        response = chat_response_to_responses(
            chat,
            response_id="resp_1",
            created_at=123,
            model="glm-4.7",
            request_body={"instructions": "x"},
        )
        assert response["usage"] == self.EXPECTED

    def test_streaming_builder_records_the_evidence_state(self):
        """The streaming path must expose the state without changing the wire shape."""
        translator = ResponsesStreamTranslator(
            response_id="r", created_at=1, model="m", request_body={}
        )
        for chunk in (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}\n\n',
            "data: [DONE]\n\n",
        ):
            list(translator.feed(chunk))
        list(translator.finalize())
        assert translator._cache_evidence_state == CACHE_EVIDENCE_UNREPORTED
        assert translator.final_response["usage"]["input_tokens_details"] == {
            "cached_tokens": 0,
        }


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
