"""X-FI-* served-identity response headers (issue #1504).

The gateway resolves the served adapter, endpoint and provider before it
dispatches, but that identity never reached the caller: ``owned_by`` is only a
provider label, and the small-budget Anthropic reroute can serve a different
model than the one the client named. Four response headers carry the identity
that dispatch already knows:

``X-FI-Requested-Model``  the model string the client sent
``X-FI-Served-Model``     the model that actually served (post-alias/reroute)
``X-FI-Served-Endpoint``  the sanitized logical endpoint id
``X-FI-Served-Provider``  the provider label of the serving adapter

The endpoint id degrades to the provider label when it is not one of the
registry's sanitized shapes: a gateway-owned LAN server is stamped
``<first-octet>-api`` and an admin-supplied route_id is kept verbatim, so a
lexical pattern — not trust in the input — is what keeps private topology out
of a caller-visible header. The api_logs columns keep the full id either way.

Every served-* field is optional: auto-selection deliberately conceals which
model serves an auto request (the response names the auto model, and so do
error messages), and that concealment extends to these headers — callers
pass only the requested model there.
"""

from __future__ import annotations

import re

REQUESTED_MODEL_HEADER = "X-FI-Requested-Model"
SERVED_MODEL_HEADER = "X-FI-Served-Model"
SERVED_ENDPOINT_HEADER = "X-FI-Served-Endpoint"
SERVED_PROVIDER_HEADER = "X-FI-Served-Provider"

# Location shapes _make_provider_id mints that are safe to publish: a local
# port ("local-12003", or bare "local") and a single lowercase service label
# with the -api suffix ("zai-api", "openrouter-api"). Anything else — LAN
# octet stamps ("10-api"), hostnames ("gpu-7-api"), admin route_ids —
# degrades to the provider label.
_SAFE_LOCATION_RE = re.compile(r"^(?:local(?:-\d+)?|[a-z][a-z0-9]*-api)$")


def header_endpoint_id(endpoint_id: str | None, provider: str | None) -> str | None:
    """Sanitized endpoint identity for a public response header.

    The api_logs column may carry a LAN or cluster-DNS stamp the registry
    derived from base_url, or an admin-supplied route_id kept verbatim; a
    response header must not. The registry's sanitized shapes pass through;
    anything else degrades to the provider label, and both absent degrade to
    ``None`` so the header is omitted entirely.
    """
    if not endpoint_id:
        return provider
    _model, sep, location = endpoint_id.rpartition(":")
    candidate = location if sep else endpoint_id
    if _SAFE_LOCATION_RE.match(candidate):
        return endpoint_id
    return provider


def served_identity_headers(
    *,
    requested_model: str | None = None,
    served_model: str | None = None,
    endpoint_id: str | None = None,
    provider: str | None = None,
) -> dict[str, str]:
    """Build the X-FI-* header set; a field is omitted when its value is None.

    Auto-selection's concealment of the served model is expressed by passing
    only ``requested_model`` — the served endpoint and provider name the same
    model the concealment hides, so they are withheld with it.
    """
    headers: dict[str, str] = {}
    if requested_model:
        headers[REQUESTED_MODEL_HEADER] = requested_model
    if served_model:
        headers[SERVED_MODEL_HEADER] = served_model
    if provider:
        headers[SERVED_PROVIDER_HEADER] = provider
    endpoint = header_endpoint_id(endpoint_id, provider)
    if endpoint:
        headers[SERVED_ENDPOINT_HEADER] = endpoint
    return headers
