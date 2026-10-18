"""Completion outcome classification.

Why this exists
---------------
A transport-level success is not evidence that an agent made progress. A
provider can answer HTTP 200 with a perfectly well-formed terminal event whose
entire output is a warmup notice::

    "The model is starting up - this takes about 120 seconds. Please wait..."

That response contains *text*, so any "did we get content?" test scores it as a
success. When such a success is allowed to update prefix-cache locality state,
a transient provider condition can permanently damage the router's model of
where the warm prefix lives.

This module separates two questions that were previously conflated:

* **transport success** -- did the HTTP request complete? (``bool``)
* **semantic outcome**  -- did this response constitute real work?

``CompletionOutcome`` is the second. It is deliberately provider-neutral and
carries no routing policy; consumers decide what each outcome means for them.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

__all__ = ["CompletionOutcome", "classify_completion_outcome"]


class CompletionOutcome(str, Enum):
    """What a completed request actually represents."""

    #: Real assistant output was produced.
    PROGRESS = "progress"
    #: The objective was demonstrably finished.
    COMPLETE = "complete"
    #: Provider is not ready (warming up, cold replica, queued). Transport
    #: succeeded but no inference work happened.
    TRANSIENT_NO_PROGRESS = "transient_no_progress"
    #: Success, but no assistant-visible output.
    EMPTY = "empty"
    #: Same non-progressing response as a recent prior attempt.
    REPEATED_NOOP = "repeated_noop"
    #: Structured provider/transport failure.
    PROVIDER_ERROR = "provider_error"
    #: The request was abandoned or cancelled by the client.
    ABORTED = "aborted"
    #: Could not be determined. Must never be treated as positive evidence.
    UNKNOWN = "unknown"

    @property
    def admits_real_work(self) -> bool:
        """Whether this outcome may be used as evidence that real work happened.

        Only ``PROGRESS`` and ``COMPLETE`` qualify. Everything else -- including
        ``UNKNOWN`` -- must leave positive evidence untouched, because an
        indeterminate observation is not grounds for discarding what we already
        know.
        """
        return self in (CompletionOutcome.PROGRESS, CompletionOutcome.COMPLETE)

    @classmethod
    def from_response(
        cls,
        *,
        content: str | None,
        usage: dict[str, Any] | None,
        http_status: int | None,
        terminal: bool,
    ) -> CompletionOutcome:
        """Classify a single response.

        ``usage`` is the provider's own reported usage mapping, when available.
        ``http_status`` of None means no HTTP response was obtained.
        """
        return classify_completion_outcome(
            content=content,
            usage=usage,
            http_status=http_status,
            terminal=terminal,
        )


#: Provider warmup / placeholder notices. Patterns, not a hard dependency on any
#: one provider's wording; deployments can extend ``register_transient_markers``
#: without touching the classification logic.
_DEFAULT_TRANSIENT_MARKERS: tuple[str, ...] = (
    r"the model is starting up",
    r"model is starting up",
    r"the model is warming up",
    r"model is warming up",
    r"warming up the model",
    r"model is loading,? please wait",
    r"loading the model,? please wait",
    r"please wait while the model (loads|starts)",
    r"the model is not ready",
    r"model is initializing",
)

# Each marker is anchored so the pattern only matches when the transient notice
# is the whole message (possibly repeated with whitespace/ellipses), not when the
# phrase appears incidentally inside a genuine answer (e.g. a model quoting or
# discussing "the model is starting up"). A response is transient only if every
# non-empty line matches a marker.
# A response is transient only when a marker dominates the whole message. Each
# marker is paired with a whole-message-anchored variant that permits a bounded
# trailing tail (the provider's own "please wait"/duration wording); a genuine
# answer that merely mentions a marker mid-text does not match the anchored
# form, so it stays PROGRESS. The bare pattern keeps an exact marker match
# transient even when a trailing word falls outside the bounded tail.
# A response is transient only when a marker leads the message and everything
# after it is a known provider-notice suffix: punctuation, a short duration
# phrase ("this takes about 120 seconds"), a wait phrase ("please wait",
# "hold on", "try again soon"), or a plain trailing word such as "now". The
# suffix set is deliberately narrow -- free-form prose after the marker is a
# genuine answer that quotes a startup message, not a transient notice, so it
# stays PROGRESS.
_NOTICE_SUFFIX = (
    r"(?:"
    r"[\s.!\u2026,;-]*"  # punctuation / spacing
    r"(?:"
    r"this (?:takes|will take) about \d+ (?:seconds?|minutes?)"  # duration
    r"|[\w\s]*please wait"
    r"|[\w\s]*hold on"
    r"|[\w\s]*try again (?:soon|later|shortly)"
    r"|[\w\s]*please try again"
    r"|[\w\s]*shortly"
    r"|[\w\s]*yet"
    r"|now"
    r")?"
    r")*$"
)
_TRANSIENT_MARKERS: list[re.Pattern[str]] = [
    re.compile(pattern + _NOTICE_SUFFIX, re.IGNORECASE) for pattern in _DEFAULT_TRANSIENT_MARKERS
]

#: Statuses that indicate a retryable provider condition rather than a bad
#: request from this gateway.
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def register_transient_marker(pattern: str) -> None:
    """Register an additional provider-specific no-progress marker."""
    _TRANSIENT_MARKERS.append(re.compile(pattern + _NOTICE_SUFFIX, re.IGNORECASE))


def _looks_transient_text(text: str) -> bool:
    return any(marker.match(text) for marker in _TRANSIENT_MARKERS)


def _output_token_count(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in ("output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def classify_completion_outcome(
    *,
    content: str | None,
    usage: dict[str, Any] | None = None,
    http_status: int | None = None,
    terminal: bool = True,
) -> CompletionOutcome:
    """Decide what a response represents.

    Order matters: a structured provider signal outranks text, and text
    outranks the mere presence of output tokens. A response that says "the model
    is starting up" is transient *even though* it has text and reports output
    tokens -- that combination is precisely the incident signature.
    """
    if http_status is not None and http_status >= 400:
        if http_status in _TRANSIENT_STATUSES:
            return CompletionOutcome.PROVIDER_ERROR
        return CompletionOutcome.PROVIDER_ERROR
    if http_status is None:
        return CompletionOutcome.UNKNOWN
    if not terminal:
        return CompletionOutcome.ABORTED

    text = (content or "").strip()

    # 1. Structured provider signal first: transient text beats everything else.
    if text and _looks_transient_text(text):
        return CompletionOutcome.TRANSIENT_NO_PROGRESS

    # 2. Genuine content.
    if text:
        return CompletionOutcome.PROGRESS

    # 3. No content. If the provider reported output tokens, something was
    #    generated that we simply could not see (e.g. reasoning-only); do not
    #    call that empty.
    out_tokens = _output_token_count(usage)
    if out_tokens is not None and out_tokens > 0:
        return CompletionOutcome.PROGRESS

    return CompletionOutcome.EMPTY
