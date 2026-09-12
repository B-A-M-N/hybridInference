"""Helpers for extracting a stable client IP from proxied requests.

The single decision point for client identity. Everything downstream — request
logs, per-IP rate limits, the auth-failure blocklist, sticky routing affinity,
and the rows written to api_logs — reads its answer. See
docs/developer/trusted-proxies-and-client-ips.md for the full trust model.

The previous implementation trusted ``X-Forwarded-For`` headers based on two
boolean flags (``TRUST_PROXY_HEADERS``, ``TRUST_CLOUDFLARE_HEADERS``) that
asserted only that *some* proxy exists. That let a caller-spoofed leftmost
``X-Forwarded-For`` (or ``CF-Forwarding``) override the real client IP, and
logged internal bridge / ULA addresses as clients when no forwarding header
was present (issue #1036).

The rewrite fixes this with one rule:

    * The **socket peer** is the only initially trustworthy network fact.
    * Forwarding headers (``CF-Connecting-IP``, ``X-Forwarded-For``, ...) are
      consulted **only when the immediate peer is in an explicitly configured
      trusted-proxy CIDR set**.
    * Within a trusted chain, walk right-to-left from the server side toward
      the origin. Discard only hops we explicitly trust as infrastructure.
      The first untrusted hop is the real client.
    * If the resulting client address is non-routable (RFC 1918, CGNAT, IPv6
      ULA, loopback, link-local, unspecified, multicast), return ``"unknown"``
      rather than masquerading an internal address as a client.

``trusted_proxies`` in the Settings class is validated at startup as a list of
CIDR ranges; an invalid entry fails configuration. The default is empty, so
forwarding headers are never trusted unless the operator explicitly opts in.
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


def _is_trusted_proxy(
    peer_ip: str, trusted_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
) -> bool:
    """True when *peer_ip* falls inside one of the configured trusted-proxy networks."""
    if not trusted_networks:
        return False
    ip = _parse_ip(peer_ip)
    if ip is None:
        return False
    return any(ip in net for net in trusted_networks)


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
    """Split and sanitize an X-Forwarded-For header value into ordered hops.

    The leftmost entry is the originally-reported client; each subsequent entry
    is a proxy that appended its own address. We trim whitespace around each
    hop and drop empty entries (a malformed ``"1.2.3.4,, 5.6.7.8"`` chain still
    yields two addresses).

    Entries that do not parse as IP addresses are preserved in the returned
    list as-is; the caller decides whether to skip or reject them. A chain of
    ``""`` or pure whitespace yields an empty list.
    """
    return [hop.strip() for hop in raw.split(",") if hop.strip()]


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
    """Return parsed trusted-proxy networks from settings, validated at startup.

    The Settings class already rejects invalid CIDRs at configuration time,
    so this will only raise if the cache is in an inconsistent state.
    """
    return get_settings().trusted_proxies_parsed


def get_client_ip_info(request: Request) -> ClientIpInfo:
    """Return the originating client IP and the socket/proxy peer that supplied it.

    Resolution walks the trust boundary correctly:

    1. ``CF-Connecting-IP`` — only when the socket peer is a configured
       trusted proxy **and** ``TRUST_CLOUDFLARE_HEADERS=1``. Cloudflare
       overwrites this header on every request, so it cannot be forged by the
       client. A corroborated Pseudo IPv4 pair yields the real IPv6 address
       from ``CF-Connecting-IPv6`` instead.
    2. The rightmost **untrusted** ``X-Forwarded-For`` hop — only when the
       socket peer is a configured trusted proxy **and**
       ``TRUST_PROXY_HEADERS=1``. The chain is walked from the server side
       (rightmost) toward the origin (leftmost); any hop that falls inside the
       trusted-proxy set is skipped. The first hop that is *not* a trusted
       proxy is the real client. This prevents a caller from prepending
       arbitrary public addresses to spoof its identity.
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
    trusted = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
    trust_cloudflare = trusted and os.getenv("TRUST_CLOUDFLARE_HEADERS", "0") == "1"
    x_forwarded_for = _header_value(request.headers.get("x-forwarded-for"))
    x_real_ip = _header_value(request.headers.get("x-real-ip"))
    cf_connecting_ip = _header_value(request.headers.get("cf-connecting-ip"))
    cf_connecting_ipv6 = _header_value(request.headers.get("cf-connecting-ipv6"))

    def _info(client_ip: str, source: str) -> ClientIpInfo:
        return ClientIpInfo(
            client_ip=client_ip,
            peer_ip=peer_ip,
            source=source,
            trusted_proxy_headers=trusted,
            x_forwarded_for=x_forwarded_for,
            x_real_ip=x_real_ip,
            cf_connecting_ip=cf_connecting_ip,
            cf_connecting_ipv6=cf_connecting_ipv6,
        )

    # Determine whether the immediate peer is a configured trusted proxy.
    # This is the gate for ALL forwarding-header trust. Without this, any
    # caller can spoof its identity via X-Forwarded-For / CF-Connecting-IP.
    networks = _trusted_networks()
    peer_is_trusted = _is_trusted_proxy(peer_ip, networks)

    if peer_is_trusted:
        # Cloudflare's edge-set header is authoritative and un-spoofable behind CF.
        if trust_cloudflare and cf_connecting_ip:
            pseudo_origin = _pseudo_ipv4_origin(cf_connecting_ip, cf_connecting_ipv6)
            return _info(
                pseudo_origin or cf_connecting_ip,
                "cf-connecting-ipv6" if pseudo_origin else "cf-connecting-ip",
            )

        # Walk the XFF chain right-to-left, skipping trusted hops and any
        # non-routable untrusted hop. The rightmost hop is the one our trusted
        # proxy appended (the address *it* saw as the incoming peer). Each hop
        # further left was added by an earlier proxy in the chain. We trust
        # only hops inside our configured proxy set — anything else is either
        # the real client or an untrusted intermediary. The first untrusted
        # routable hop we find is the client; non-routable untrusted hops
        # (private, ULA, etc.) are skipped because they cannot represent a real
        # remote client — reporting one is what leaked internal-overlay
        # addresses as clients.
        #
        # Example:  XFF = "1.2.3.4, fdbd:dc02::153, 10.0.0.1, 172.16.0.5"
        #   172.16.0.5 is trusted (our peer proxy) → skip
        #   10.0.0.1 is also trusted → skip
        #   fdbd:dc02::153 is untrusted but ULA → skip (not reportable)
        #   1.2.3.4 is untrusted and routable → this is the real client
        if x_forwarded_for and trusted:
            hops = _parse_forwarded_chain(x_forwarded_for)
            for hop in reversed(hops):
                if _is_trusted_proxy(hop, networks):
                    continue
                if _is_reportable_ip(hop):
                    return _info(hop, "x-forwarded-for")
                # Non-routable untrusted hop (private/ULA/etc.) — skip and
                # keep looking toward the origin.
                continue

        if trusted and _is_reportable_ip(x_real_ip):
            return _info(x_real_ip, "x-real-ip")  # type: ignore[arg-type]

    # No trustworthy forwarded hop: fall back to the socket peer. A direct
    # public connection is a real client; an internal peer (docker bridge,
    # etc.) is reported as "unknown" — it cannot represent a real remote
    # client, and logging it as one is the pollution this fix eliminates.
    if _is_reportable_ip(peer_ip):
        return _info(peer_ip, "socket")
    return _info("unknown", "unknown")


def get_client_ip(request: Request) -> str:
    """Return the best-effort originating client IP for a request."""
    return get_client_ip_info(request).client_ip


def get_client_ip_bucket(request: Request) -> str:
    """Return the rate-limit/affinity bucket key for a request's client IP."""
    return normalize_ip_bucket(get_client_ip(request))
