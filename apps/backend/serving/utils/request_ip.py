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
   * if it is non-routable or malformed, return ``"unknown"``.
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

``trusted_proxies`` and ``trusted_cloudflare_networks`` are validated at
startup as comma-separated CIDR lists; invalid entries fail configuration.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
# those reclassified the documentation and benchmark ranges across CPython
# 3.12.4 / 3.13, so relying on them would make IP resolution depend on the
# interpreter version. RFC 1918, CGNAT (RFC 6598) and IPv6 ULA are stable.
_NON_ROUTABLE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fc00::/7"),
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
    unspecified, RFC 1918 / CGNAT and IPv6 ULA addresses. IPv4-mapped IPv6
    literals are judged by their embedded IPv4 address so a mapped private peer
    is still rejected.
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

    ``trusted_proxy_headers`` describes whether forwarding headers were
    actually trusted for this request — that is, whether the socket peer
    was an authorized proxy for forwarding provenance. It is **True** only
    when both the global ``TRUST_PROXY_HEADERS`` configuration flag is enabled
    **and** the immediate peer falls inside a configured trusted-proxy CIDR.

    ``client_ip`` is the best-effort originating client address, or the literal
    ``"unknown"`` when no trustworthy routable address could be determined. It
    is never an RFC 1918 / CGNAT / ULA / loopback / link-local / unspecified
    address — if the resolution lands on one of those, the result is
    ``"unknown"`` instead.
    """

    client_ip: str
    peer_ip: str
    source: str
    trusted_proxy_headers: bool
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
    client_ip: str,
    *,
    grant_id: str | None = None,
) -> str:
    """Compute the affinity key used for sticky multi-key routing.

    Falls through the caller identities in order of how precisely each names one
    caller:

    1. ``auth_key_hash`` — the hyi-xxx key presented. The ordinary case.
    2. ``grant_id`` — an inference-grant token carries no key hash, so without
       this a sandbox would key on its IP and every sandbox behind one NAT or
       relay address would collapse onto a single binding.
    3. The client IP bucket, for traffic with no credential at all.

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
    return f"ip:{normalize_ip_bucket(client_ip)}"


def _trusted_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return parsed trusted-proxy networks from settings, validated at startup."""
    return get_settings().trusted_proxies_parsed


def _trusted_cloudflare_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return parsed Cloudflare-authorized networks from settings."""
    return get_settings().trusted_cloudflare_parsed


def get_client_ip_info(request: Request) -> ClientIpInfo:
    """Return the originating client IP and the socket/proxy peer that supplied it.

    Resolution walks the trust boundary correctly:

    1. ``CF-Connecting-IP`` — only when the socket peer is a configured
       Cloudflare-authorized proxy (separate from generic trusted proxies)
       **and** ``TRUST_CLOUDFLARE_HEADERS=1``. The operator must explicitly
       configure which networks are Cloudflare-authorized; a generic reverse
       proxy does not make this header safe. A corroborated Pseudo IPv4 pair
       yields the real IPv6 address from ``CF-Connecting-IPv6`` instead.
    2. The **first untrusted hop** in ``X-Forwarded-For`` — only when the
       socket peer is a configured trusted proxy **and**
       ``TRUST_PROXY_HEADERS=1``. The chain is walked from the server side
       (rightmost) toward the origin (leftmost); hops that fall inside the
       trusted-proxy set are skipped. The first hop that is *not* a trusted
       proxy terminates provenance: if it is routable, it is the client; if it
       is non-routable or malformed, the result is ``"unknown"``. We never
       continue leftward past the first untrusted hop.
    3. ``X-Real-IP`` — only when trusted and routable.
    4. The socket peer — a direct connection, or the last resort when no
       forwarded hop is usable. If the peer itself is non-routable, the result
       is ``"unknown"``.

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
    peer_is_cloudflare = cf_trust_enabled and _is_in_networks(peer_ip, cf_networks)

    # Request-level trust: headers are trusted only if the peer is authorized.
    headers_trusted = peer_is_trusted_proxy

    def _info(client_ip: str, source: str) -> ClientIpInfo:
        return ClientIpInfo(
            client_ip=client_ip,
            peer_ip=peer_ip,
            source=source,
            trusted_proxy_headers=headers_trusted,
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
                return _info(pseudo_origin, "cf-connecting-ipv6")
            return _info("unknown", "cf-connecting-ipv6")
        # No valid Pseudo IPv4 pair. Parse the CF-Connecting-IP directly.
        parsed_cf = _parse_ip(cf_connecting_ip)
        if parsed_cf is not None and parsed_cf in PSEUDO_IPV4_NETWORK:
            # A bare Class-E value without a valid Pseudo IPv4 pair is not a
            # real client address (Cloudflare puts the synthetic in
            # CF-Connecting-IP and the real IPv6 in CF-Connecting-IPv6).
            return _info("unknown", "cf-connecting-ip")
        if _is_reportable_ip(cf_connecting_ip):
            return _info(cf_connecting_ip, "cf-connecting-ip")
        return _info("unknown", "cf-connecting-ip")

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
        #     If non-routable / malformed: return "unknown"
        #     NEVER continue farther left
        if x_forwarded_for:
            hops = _parse_forwarded_chain(x_forwarded_for)
            for hop in reversed(hops):
                if _is_in_networks(hop, proxy_networks):
                    continue
                # First untrusted hop: provenance terminates here.
                if _is_reportable_ip(hop):
                    return _info(hop, "x-forwarded-for")
                # Non-routable or malformed untrusted hop: return unknown.
                # Do NOT continue leftward — that would cross the trust
                # boundary and consume attacker-controlled values.
                return _info("unknown", "x-forwarded-for")

        if _is_reportable_ip(x_real_ip):
            return _info(x_real_ip, "x-real-ip")  # type: ignore[arg-type]

    # No trustworthy forwarded hop: fall back to the socket peer. A direct
    # public connection is a real client; an internal peer (docker bridge,
    # etc.) is reported as "unknown" — it cannot represent a real remote
    # client, and logging it as one is the pollution this fix eliminates.
    if _is_reportable_ip(peer_ip):
        return _info(peer_ip, "socket")
    return _info("unknown", "unknown")


def get_client_ip(request: Request) -> str:
    """Return the best-effort originating client IP for a request.

    This is the **provenance identity** — it may be ``"unknown"`` when no
    trustworthy client address could be determined. Use this for logging,
    audit, and display. For rate-limiting, auth-blocking, and affinity, use
    :func:`get_client_enforcement_id` instead, which never returns a shared
    ``"unknown"`` value that could collapse unrelated callers onto one key.
    """
    return get_client_ip_info(request).client_ip


def get_client_bucket(request: Request) -> str:
    """Return the client bucket key for rate-limiting/blocking.

    Unlike :func:`get_client_ip`, this **never returns a shared sentinel** like
    ``"unknown"``. When client provenance fails, it falls back to the socket
    peer's bucket, so unrelated callers don't collapse onto one key.

    Returns the raw bucket (e.g. ``"172.19.0.1"`` or ``"2001:db8::/64"``)
    without the ``"ip:"`` prefix.
    """
    info = get_client_ip_info(request)
    if info.client_ip != "unknown":
        return normalize_ip_bucket(info.client_ip)
    return normalize_ip_bucket(info.peer_ip)


def get_client_enforcement_id(request: Request) -> str:
    """Return the enforcement identity for a request.

    Unlike :func:`get_client_ip`, this **never returns a shared sentinel** like
    ``"unknown"``. When client provenance fails, it falls back to the socket
    peer's bucket, so that:

    * Rate limits key on the actual network-level source, not a shared
      ``"unknown"`` bucket that every untrusted caller collapses onto.
    * Auth-failure blocks target the real peer, not a global ``"unknown"``
      bucket that would block all direct-connection callers at once.
    * Affinity keys on the peer when the client is unresolvable, preserving
      per-source routing rather than merging everyone onto one backend.

    Returns ``"ip:<bucket>"`` (e.g. ``"ip:172.19.0.1"``).
    """
    return f"ip:{get_client_bucket(request)}"


def get_client_ip_bucket(request: Request) -> str:
    """Return the rate-limit/affinity bucket key for a request's client IP.

    .. deprecated::
        Use :func:`get_client_enforcement_id` for new code. This function
        preserves the old behavior for backward compatibility but will
        collapse all ``"unknown"`` callers onto one bucket.
    """
    return normalize_ip_bucket(get_client_ip(request))
