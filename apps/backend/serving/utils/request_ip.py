"""Helpers for extracting a stable client IP from proxied requests.

The single decision point for client identity. Everything downstream — request
logs, per-IP rate limits, the auth-failure blocklist, sticky routing affinity,
and the rows written to ``api_logs`` — reads its answer. See
``docs/developer/trusted-proxies-and-client-ips.md`` for the full trust model.

Security model
--------------

1. **The socket peer is the only initially trustworthy network fact.**
   Forwarding headers (``CF-Connecting-IP``, ``X-Forwarded-For``, etc.) are
   usable only when the immediate peer is in an explicitly configured
   trusted-proxy CIDR set.

2. **Explicit trust only.** Forwarding headers from untrusted peers are
   *never* used — they may be attacker-supplied. The trusted-proxy CIDR set
   defaults to empty, so forwarding headers are never trusted unless the
   operator explicitly opts in.

3. **First untrusted hop terminates provenance.** Within a trusted chain, walk
   right-to-left. Discard only hops explicitly in the trusted-proxy set. The
   first hop that is *not* a trusted proxy **is where provenance terminates**:
   * if it is routable, it is the real client;
   * if it is non-routable or malformed, provenance is **unresolved**.
   Never continue farther left — that would cross the trust boundary and
   consume attacker-controlled values.

4. **Separate Cloudflare trust.** A generic trusted reverse proxy does NOT
   make a client-supplied ``CF-Connecting-IP`` safe. ``CF-Connecting-IP`` is
   trusted only when the immediate peer is in an *additional* explicitly
   configured Cloudflare-authorized network. This network must be configured
   separately from generic trusted proxies.

5. **``"unknown"`` is a legitimate outcome.** If the socket peer is
   non-routable and there is no trustworthy forwarding provenance, the result
   is ``"unknown"`` rather than masquerading an internal address as a client.

6. **Unresolved identity is not recoverable.** When client provenance fails,
   there is no information to distinguish clients behind a shared proxy. The
   proxy IP is useful for diagnosing *where a connection came from*, but it is
   NOT a substitute for *who the client was*. Consumers must handle unresolved
   identity explicitly rather than silently falling back to the proxy address.

``trusted_proxies`` and ``trusted_cloudflare_networks`` are validated at
startup as comma-separated CIDR lists; invalid entries fail configuration.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from serving.config.settings import get_settings

if TYPE_CHECKING:
    from fastapi import Request

# IPv6 clients are routinely delegated a /64 (frequently a /56 or /48), and
# RFC 4941 privacy addresses rotate within it, so a single client can present
# effectively unlimited distinct addresses. Abuse-control and affinity buckets
# therefore collapse IPv6 to its /64 network. IPv4 keeps full-address buckets.
IPV6_BUCKET_PREFIXLEN = 64

# Cloudflare's Pseudo IPv4 synthetics live in the reserved Class E space. A real
# client address is never drawn from it, so its presence in CF-Connecting-IP is
# what distinguishes a genuine Pseudo IPv4 rewrite from a forged pairing.
PSEUDO_IPV4_NETWORK = ipaddress.ip_network("240.0.0.0/4")

# Non-routable ranges that can never identify a remote client. This is an
# explicit list rather than ``ipaddress.is_private`` / ``is_global`` on purpose:
# those reclassified special-use ranges across CPython releases, so relying on
# them would make IP resolution depend on the interpreter version. Keep the
# explicit deny set stable and include documentation, benchmarking, and
# reserved ranges that must never be treated as real client addresses.
_NON_ROUTABLE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("2001:db8::/32"),
)


def _header_value(value: object) -> str | None:
    """Return a stripped header value when the request provides a real string."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a string into an IP address, returning None on failure.

    IPv4-mapped IPv6 literals (``::ffff:192.0.2.1``) are unwrapped to their
    embedded IPv4 address for uniform handling — the caller sees one address
    type and the routability check below works for both families.
    """
    if not value:
        return None
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_reportable_ip(value: str | None) -> bool:
    """True when *value* could plausibly identify a real remote client.

    Rejects anything unparseable plus loopback, link-local, multicast,
    unspecified, RFC 1918 / CGNAT, IPv6 ULA, documentation, benchmarking,
    and reserved ranges. IPv4-mapped IPv6 literals are judged by their
    embedded IPv4 address so a mapped private peer is still rejected.
    """
    if not value:
        return False
    ip = _parse_ip(value)
    if ip is None:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return False
    return not any(ip in net for net in _NON_ROUTABLE_NETWORKS)


@dataclass(frozen=True)
class ClientIpInfo:
    """Client IP plus enough provenance to debug proxy hops.

    ``trusted_proxy_headers`` describes whether any forwarding identity header
    was actually trusted for this request. It is **True** only when the global
    ``TRUST_PROXY_HEADERS`` configuration flag is enabled and the immediate
    peer is authorized either by the generic trusted-proxy CIDRs or by the
    separately configured Cloudflare CIDRs with ``TRUST_CLOUDFLARE_HEADERS``.

    ``client_ip`` is the best-effort originating client address, or the literal
    ``"unknown"`` when no trustworthy routable address could be determined. It
    is never an RFC 1918 / CGNAT / ULA / loopback / link-local / unspecified
    address — if the resolution lands on one of those, the result is
    ``"unknown"`` instead.

    ``resolved`` is ``True`` when ``client_ip`` is a real routable address that
    can be attributed to a specific client. It is ``False`` when the result is
    ``"unknown"`` — meaning client provenance could not be established. This
    distinction is critical for downstream consumers: when ``resolved`` is
    ``False``, there is no information to distinguish clients behind a shared
    proxy, and the proxy IP must NOT be used as a substitute for client
    identity.
    """

    client_ip: str
    peer_ip: str
    source: str
    trusted_proxy_headers: bool
    resolved: bool
    x_forwarded_for: str | None = None
    x_real_ip: str | None = None
    cf_connecting_ip: str | None = None
    cf_connecting_ipv6: str | None = None


def _is_in_networks(
    peer_ip: str, networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
) -> bool:
    """True when *peer_ip* falls inside one of the given networks."""
    if not networks:
        return False
    ip = _parse_ip(peer_ip)
    if ip is None:
        return False
    return any(ip in net for net in networks)


def _pseudo_ipv4_origin(cf_connecting_ip: str | None, cf_connecting_ipv6: str | None) -> str | None:
    """Return the visitor's real IPv6 only for a genuine Pseudo IPv4 rewrite.

    Cloudflare overwrites ``CF-Connecting-IP`` on every request, but it emits
    ``CF-Connecting-IPv6`` *only* under Pseudo IPv4 "Overwrite headers" — when
    that setting is off the header is absent rather than cleared, so a caller
    can supply their own. Preferring it unconditionally would therefore hand an
    attacker the client identity on any ordinary Cloudflare request.

    Both halves of the pair must corroborate each other: the IPv6 header must
    parse as IPv6, and ``CF-Connecting-IP`` must hold the accompanying Class E
    synthetic. Cloudflare controls that second value and a real client address
    is never in ``240.0.0.0/4``, so the pairing cannot be forged from outside.
    """
    if not cf_connecting_ip or not cf_connecting_ipv6:
        return None
    try:
        synthetic = ipaddress.ip_address(cf_connecting_ip)
        original = ipaddress.ip_address(cf_connecting_ipv6)
    except ValueError:
        return None
    if original.version != 6 or synthetic.version != 4:
        return None
    if synthetic not in PSEUDO_IPV4_NETWORK:
        return None
    return str(original)


def _parse_forwarded_chain(raw: str) -> list[str]:
    """Split an X-Forwarded-For header value into ordered hops.

    The leftmost entry is the originally-reported client; each subsequent entry
    is a proxy that appended its address. We trim whitespace around each hop
    and preserve **all** entries including empty/malformed ones — the caller
    treats them as a provenance-terminating condition (returns "unknown").

    Silently dropping empty elements would be inconsistent with the fail-closed
    design: a malformed empty hop must terminate the chain, not disappear.
    A chain of ``""`` or pure whitespace yields a single empty string so the
    caller can reject it.
    """
    return [hop.strip() for hop in raw.split(",")] if raw else []


def normalize_ip_bucket(ip: str) -> str:
    """Return the grouping key used for per-client rate limits and affinity.

    IPv4 addresses (and anything unparseable, such as ``"unknown"`` or a
    scoped literal) bucket on the value itself. IPv6 addresses bucket on
    their ``/64`` network so a client cannot escape a limit by rotating
    through the prefix it was delegated.

    IPv4-mapped literals (``::ffff:192.0.2.1``, which a dual-stack listener
    reports for IPv4 peers) bucket on the embedded IPv4 address. Folding them
    by prefix would collapse every IPv4 client into a single ``::/64``.
    """
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if parsed.version == 4:
        return str(parsed)
    mapped_v4 = parsed.ipv4_mapped
    if mapped_v4 is not None:
        return str(mapped_v4)
    network = ipaddress.ip_network(f"{parsed}/{IPV6_BUCKET_PREFIXLEN}", strict=False)
    return str(network)


def derive_affinity_key(
    auth_key_hash: str | None,
    client_ip_info: ClientIpInfo,
    *,
    grant_id: str | None = None,
) -> str | None:
    """Compute the affinity key used for sticky multi-key routing.

    Falls through the caller identities in order of how precisely each names one
    caller:

    1. ``auth_key_hash`` — the hyi-xxx key presented. The ordinary case.
    2. ``grant_id`` — an inference-grant token carries no key hash, so without
       this a sandbox would key on its IP and every sandbox behind one NAT or
       relay address would collapse onto a single binding.
    3. The client IP bucket, for traffic with no credential at all **and**
       resolved client provenance.

    When client provenance is unresolved (``resolved=False``), there is no
    information to distinguish clients behind a shared proxy. Returns ``None``
    so the caller can use non-sticky routing rather than collapsing all
    unrelated clients onto a single backend.

    Anonymous IPv6 clients key on their ``/64`` so rotating privacy addresses
    within the delegated prefix keeps landing on the same backend.

    Lives here rather than on a router so every request surface that dispatches
    to a pooled adapter (``/v1/chat/completions``, ``/v1/messages``,
    ``/v1/embeddings``) derives the caller identity the same way.
    """
    if auth_key_hash:
        return auth_key_hash
    if grant_id:
        return f"grant:{grant_id}"
    if client_ip_info.resolved:
        return f"ip:{normalize_ip_bucket(client_ip_info.client_ip)}"
    return None


def _trusted_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return parsed trusted-proxy networks from settings, validated at startup."""
    return get_settings().trusted_proxies_parsed


def _trusted_cloudflare_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return parsed Cloudflare-authorized networks from settings."""
    return get_settings().trusted_cloudflare_parsed


def get_client_ip_info(request: Request) -> ClientIpInfo:
    """Return the originating client IP and the socket/proxy peer that supplied it.

    Resolution walks the trust boundary correctly:

    1. ``CF-Connecting-IP`` — only when ``TRUST_PROXY_HEADERS=1`` **and**
       ``TRUST_CLOUDFLARE_HEADERS=1`` **and** the socket peer is in
       ``trusted_cloudflare_networks``. The operator must explicitly
       configure which networks are Cloudflare-authorized; a generic reverse
       proxy does not make this header safe. A corroborated Pseudo IPv4 pair
       yields the real IPv6 address from ``CF-Connecting-IPv6`` instead.
    2. The **first untrusted hop** in ``X-Forwarded-For`` — only when the
       socket peer is a configured trusted proxy **and**
       ``TRUST_PROXY_HEADERS=1``. The chain is walked from the server side
       (rightmost) toward the origin (leftmost); hops that fall inside the
       trusted-proxy set are skipped. The first hop that is *not* a trusted
       proxy terminates provenance: if it is routable, it is the client; if it
       is non-routable or malformed, provenance is **unresolved**. We never
       continue leftward past the first untrusted hop.
    3. ``X-Real-IP`` — only when trusted and routable.
    4. The socket peer — a direct connection, or the last resort when no
       forwarded hop is usable. If the peer itself is non-routable, provenance
       is **unresolved**.

    When ``trusted_proxies`` is empty (the default), forwarding headers are
    never trusted, and the socket peer is the only network fact used. This is
    the fail-closed default — an operator must explicitly configure which CIDR
    ranges are allowed to assert forwarding provenance.
    """
    peer_ip = request.client.host if request.client else "unknown"
    global_trust_enabled = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
    cf_trust_enabled = os.getenv("TRUST_CLOUDFLARE_HEADERS", "0") == "1"
    x_forwarded_for = _header_value(request.headers.get("x-forwarded-for"))
    x_real_ip = _header_value(request.headers.get("x-real-ip"))
    cf_connecting_ip = _header_value(request.headers.get("cf-connecting-ip"))
    cf_connecting_ipv6 = _header_value(request.headers.get("cf-connecting-ipv6"))

    # Determine whether the immediate peer is a configured trusted proxy.
    # This is the gate for ALL forwarding-header trust. Without this, any
    # caller can spoof its identity via X-Forwarded-For / CF-Connecting-IP.
    proxy_networks = _trusted_networks()
    cf_networks = _trusted_cloudflare_networks()
    peer_is_trusted_proxy = global_trust_enabled and _is_in_networks(peer_ip, proxy_networks)
    # Cloudflare trust requires BOTH the global proxy flag AND the Cloudflare flag.
    # This preserves the master kill-switch semantics: TRUST_PROXY_HEADERS=0 means
    # no forwarded headers can influence identity, regardless of other flags.
    peer_is_cloudflare = (
        global_trust_enabled and cf_trust_enabled and _is_in_networks(peer_ip, cf_networks)
    )

    # Request-level trust: headers are trusted only if the peer is authorized
    # by either forwarding trust model. CF-only requests must not be recorded
    # as though their authoritative header was untrusted.
    headers_trusted = peer_is_trusted_proxy or peer_is_cloudflare

    def _info(client_ip: str, source: str, resolved: bool) -> ClientIpInfo:
        return ClientIpInfo(
            client_ip=client_ip,
            peer_ip=peer_ip,
            source=source,
            trusted_proxy_headers=headers_trusted,
            resolved=resolved,
            x_forwarded_for=x_forwarded_for,
            x_real_ip=x_real_ip,
            cf_connecting_ip=cf_connecting_ip,
            cf_connecting_ipv6=cf_connecting_ipv6,
        )

    if peer_is_cloudflare and cf_connecting_ip:
        # CF-Connecting-IP is authoritative only from a Cloudflare-authorized hop.
        # Validate the derived client address before returning it — never return
        # a private/loopback/ULA/multicast/unspecified address as the client.
        pseudo_origin = _pseudo_ipv4_origin(cf_connecting_ip, cf_connecting_ipv6)
        if pseudo_origin is not None:
            # Genuine Pseudo IPv4 pair: use the corroborated IPv6 address.
            if _is_reportable_ip(pseudo_origin):
                return _info(pseudo_origin, "cf-connecting-ipv6", True)
            return _info("unknown", "cf-connecting-ipv6", False)
        # No valid Pseudo IPv4 pair. Parse the CF-Connecting-IP directly.
        parsed_cf = _parse_ip(cf_connecting_ip)
        if parsed_cf is not None and parsed_cf in PSEUDO_IPV4_NETWORK:
            # A bare Class-E value without a valid Pseudo IPv4 pair is not a
            # real client address (Cloudflare puts the synthetic in
            # CF-Connecting-IP and the real IPv6 in CF-Connecting-IPv6).
            return _info("unknown", "cf-connecting-ip", False)
        if _is_reportable_ip(cf_connecting_ip):
            return _info(cf_connecting_ip, "cf-connecting-ip", True)
        return _info("unknown", "cf-connecting-ip", False)

    if peer_is_trusted_proxy:
        # Walk the XFF chain right-to-left, skipping explicitly trusted hops.
        # The rightmost hop is the one our trusted proxy appended (the address
        # *it* saw as the incoming peer). Each hop further left was added by an
        # earlier proxy in the chain. We trust only hops inside our configured
        # proxy set — the first hop that is *not* a trusted proxy is the trust
        # boundary, and provenance terminates there.
        #
        # Example:  XFF = "1.2.3.4, 10.0.0.1, 172.16.0.5"
        #   172.16.0.5 is trusted (our peer proxy) → skip
        #   10.0.0.1 is also trusted → skip
        #   1.2.3.4 is NOT trusted → provenance terminates here
        #     If routable: it is the client
        #     If non-routable / malformed: unresolved
        #     NEVER continue farther left
        if x_forwarded_for:
            hops = _parse_forwarded_chain(x_forwarded_for)
            for hop in reversed(hops):
                if _is_in_networks(hop, proxy_networks):
                    continue
                # First untrusted hop: provenance terminates here.
                if _is_reportable_ip(hop):
                    return _info(hop, "x-forwarded-for", True)
                # Non-routable or malformed untrusted hop: unresolved.
                # Do NOT continue leftward — that would cross the trust
                # boundary and consume attacker-controlled values.
                return _info("unknown", "x-forwarded-for", False)

        if _is_reportable_ip(x_real_ip):
            return _info(x_real_ip, "x-real-ip", True)  # type: ignore[arg-type]

    # No trustworthy forwarded hop: fall back to the socket peer. A direct
    # public connection is a real client; an internal peer (docker bridge,
    # etc.) is unresolved — it cannot represent a real remote client, and
    # logging it as one is the pollution this fix eliminates.
    if _is_reportable_ip(peer_ip):
        return _info(peer_ip, "socket", True)
    return _info("unknown", "unknown", False)


def get_client_ip(request: Request) -> str:
    """Return the best-effort originating client IP for a request.

    This is the **provenance identity** — it may be ``"unknown"`` when no
    trustworthy client address could be determined. Use this for logging,
    audit, and display.

    .. note::
        For rate-limiting, auth-blocking, and affinity, do **not** use this
        function directly. Use :func:`get_client_ip_info` and inspect
        ``resolved`` to determine the appropriate identity for the operation.
        When ``resolved`` is ``False``, there is no information to distinguish
        clients behind a shared proxy.
    """
    return get_client_ip_info(request).client_ip


def get_client_ip_bucket(request: Request) -> str:
    """Return the rate-limit/affinity bucket key for a request's client IP.

    .. deprecated::
        This function preserves the old behavior for backward compatibility but
        will collapse all ``"unknown"`` callers onto one bucket. Use
        :func:`get_client_ip_info` and inspect ``resolved`` for new code.
    """
    return normalize_ip_bucket(get_client_ip(request))


def client_ip_metadata(ip_info: ClientIpInfo) -> dict[str, Any]:
    """Build the metadata dict for persisting client IP provenance.

    Used by all logging surfaces that record ``api_logs.metadata`` so that
    the stored IP can be audited: which rung fired, what the socket peer
    was, and what the forwarding chain said. All fields are present even
    when their values are None so the schema is stable.

    Keys:
        ip: resolved client IP or "unknown"
        peer_ip: socket peer that supplied the connection
        source: resolution rung that produced the IP
        resolved: whether client provenance was established
        trusted_proxy_headers: whether forwarding headers were trusted
        x_forwarded_for: raw header value (None if absent)
        x_real_ip: raw header value (None if absent)
        cf_connecting_ip: raw header value (None if absent)
        cf_connecting_ipv6: raw header value (None if absent)
    """
    return {
        "ip": ip_info.client_ip,
        "peer_ip": ip_info.peer_ip,
        "ip_source": ip_info.source,
        "resolved": ip_info.resolved,
        "trusted_proxy_headers": ip_info.trusted_proxy_headers,
        "x_forwarded_for": ip_info.x_forwarded_for,
        "x_real_ip": ip_info.x_real_ip,
        "cf_connecting_ip": ip_info.cf_connecting_ip,
        "cf_connecting_ipv6": ip_info.cf_connecting_ipv6,
    }
