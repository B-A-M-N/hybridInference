"""Tests for proxied client IP extraction with trusted-proxy trust model.

Covers the security fix for issue #1036: client IP resolution now requires
the socket peer to be in an explicitly configured trusted-proxy CIDR set
before any forwarding header is consulted. Non-routable peers resolve to
"unknown" instead of masquerading as clients.
"""

from __future__ import annotations

import ipaddress
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from serving.utils.request_ip import (
    _is_in_networks,
    _is_reportable_ip,
    _parse_forwarded_chain,
    _parse_ip,
    derive_affinity_key,
    get_client_bucket,
    get_client_enforcement_id,
    get_client_ip_bucket,
    get_client_ip_info,
    normalize_ip_bucket,
)


def _request(headers: dict[str, str], peer_ip: str = "10.0.0.2"):
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer_ip))


def _networks(*cidrs: str) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Create a tuple of parsed networks from CIDR strings."""
    return tuple(ipaddress.ip_network(c) for c in cidrs)


@contextmanager
def _patch_networks(proxy_nets, cf_nets):
    """Return a context manager that patches both trusted_networks and
    trusted_cloudflare_networks."""
    with patch("serving.utils.request_ip._trusted_networks", return_value=proxy_nets), patch(
        "serving.utils.request_ip._trusted_cloudflare_networks",
        return_value=cf_nets,
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_ip_accepts_ipv4_ipv6_mapped():
    assert str(_parse_ip("1.2.3.4")) == "1.2.3.4"
    assert str(_parse_ip("2001:db8::1")) == "2001:db8::1"
    assert str(_parse_ip("::ffff:192.0.2.1")) == "192.0.2.1"


def test_parse_ip_rejects_garbage():
    assert _parse_ip("not-an-ip") is None
    assert _parse_ip("") is None
    assert _parse_ip(None) is None


def test_parse_forwarded_chain_splits_and_trims():
    assert _parse_forwarded_chain("1.2.3.4, 5.6.7.8") == ["1.2.3.4", "5.6.7.8"]


def test_parse_forwarded_chain_preserves_empty():
    """Empty/malformed entries are preserved — they terminate provenance."""
    assert _parse_forwarded_chain("8.8.8.8, , 172.19.0.1") == ["8.8.8.8", "", "172.19.0.1"]
    assert _parse_forwarded_chain("  1.2.3.4  ,  , 5.6.7.8  ") == ["1.2.3.4", "", "5.6.7.8"]


def test_parse_forwarded_chain_empty_input():
    """A chain of empty/whitespace yields a list with empty strings preserved."""
    assert _parse_forwarded_chain(" , ") == ["", ""]


def test_is_reportable_ip_rejects_non_routable():
    assert _is_reportable_ip("172.19.0.1") is False
    assert _is_reportable_ip("10.0.0.1") is False
    assert _is_reportable_ip("fc00::1") is False
    assert _is_reportable_ip("127.0.0.1") is False
    assert _is_reportable_ip("::1") is False
    assert _is_reportable_ip("fe80::1") is False
    assert _is_reportable_ip("224.0.0.1") is False
    assert _is_reportable_ip("0.0.0.0") is False


def test_is_reportable_ip_accepts_public():
    assert _is_reportable_ip("8.8.8.8") is True
    assert _is_reportable_ip("203.0.113.9") is True
    assert _is_reportable_ip("2001:db8::1") is True


def test_is_in_networks_matches_cidr():
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    assert _is_in_networks("172.19.0.1", nets) is True
    assert _is_in_networks("10.0.0.1", nets) is True
    assert _is_in_networks("8.8.8.8", nets) is False
    assert _is_in_networks("not-an-ip", nets) is False


def test_is_in_networks_empty_means_no_trust():
    assert _is_in_networks("172.19.0.1", ()) is False


# ---------------------------------------------------------------------------
# Default behavior: no trusted proxies configured
# ---------------------------------------------------------------------------


def test_non_routable_peer_returns_unknown_no_trusted_proxies(monkeypatch):
    """Docker bridge peer without trustworthy forwarding provenance → "unknown"."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        request = _request({}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_public_peer_used_directly_no_trusted_proxies(monkeypatch):
    """A direct connection from a routable peer is a legitimate client IP."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        info = get_client_ip_info(_request({}, peer_ip="8.8.8.8"))
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_forged_xff_ignored_without_trusted_peer(monkeypatch):
    """Attacker cannot spoof XFF when peer is not a configured trusted proxy."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    with _patch_networks((), ()):
        request = _request({"x-forwarded-for": "203.0.113.9"}, peer_ip="8.8.8.8")
        info = get_client_ip_info(request)
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_forged_cf_connecting_ip_ignored_without_trusted_peer(monkeypatch):
    """Attacker cannot spoof CF-Connecting-IP without a trusted peer."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    with _patch_networks((), ()):
        request = _request({"cf-connecting-ip": "203.0.113.9"}, peer_ip="8.8.8.8")
        info = get_client_ip_info(request)
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_no_client_means_unknown(monkeypatch):
    """A request with no client at all resolves to "unknown"."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    request = SimpleNamespace(headers={}, client=None)
    with _patch_networks((), ()):
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


# ---------------------------------------------------------------------------
# Trusted proxy behavior: peer in configured CIDR
# ---------------------------------------------------------------------------


def test_trusted_proxy_first_untrusted_hop_is_client(monkeypatch):
    """XFF chain walked right-to-left; first untrusted hop is the client."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "1.2.3.4, 10.0.0.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "1.2.3.4"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_all_trusted_hops_returns_unknown(monkeypatch):
    """When every XFF hop is a trusted proxy, no client can be determined."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("10.0.0.0/8", "172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # All hops trusted → fall through. Peer is non-routable → "unknown".
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_trusted_proxy_first_untrusted_private_hop_returns_unknown(monkeypatch):
    """First untrusted hop that is non-routable → "unknown"."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "8.8.8.8, 10.50.0.8, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # 172.19.0.1 is trusted → skip. 10.50.0.8 is first untrusted hop but
        # non-routable → provenance terminates, return "unknown".
        assert info.client_ip == "unknown"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_first_untrusted_ula_hop_returns_unknown(monkeypatch):
    """First untrusted hop that is ULA → "unknown"."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "8.8.8.8, fdbd:dc02:19:383::153, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # fdbd:dc02::153 is first untrusted hop but ULA → "unknown"
        assert info.client_ip == "unknown"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_first_untrusted_malformed_hop_returns_unknown(monkeypatch):
    """First untrusted hop that is malformed → "unknown"."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "8.8.8.8, garbage, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # "garbage" is first untrusted hop and malformed → "unknown"
        assert info.client_ip == "unknown"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_empty_hop_terminates_provenance(monkeypatch):
    """Empty hop between trusted and client terminates provenance with unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        # 8.8.8.8, [empty], 172.19.0.1
        # Walking right-to-left: 172.19.0.1 trusted, "" first untrusted → unknown
        request = _request(
            {"x-forwarded-for": "8.8.8.8, , 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_attacker_prepends_fake_addresses(monkeypatch):
    """Attacker prepending public addresses to XFF does not spoof identity."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "1.2.3.4, 5.6.7.8, 203.0.113.9, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_x_real_ip_fallback(monkeypatch):
    """X-Real-IP used when XFF has no usable hop (all trusted)."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    # Both 172.19.0.1 and 10.0.0.1 are trusted, so XFF has no untrusted hop.
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 172.19.0.1", "x-real-ip": "203.0.113.9"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # XFF: 172.19.0.1 trusted, 10.0.0.1 trusted → fall through to X-Real-IP
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-real-ip"


# ---------------------------------------------------------------------------
# Cloudflare header handling
# ---------------------------------------------------------------------------


def test_cf_connecting_ip_when_cloudflare_authorized(monkeypatch):
    """CF-Connecting-IP is authoritative when peer is in Cloudflare-authorized networks."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request(
            {
                "x-forwarded-for": "1.2.3.4, 203.0.113.9",
                "cf-connecting-ip": "203.0.113.9",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "cf-connecting-ip"


def test_cf_connecting_ip_ignored_with_generic_trust_only(monkeypatch):
    """Generic proxy trust does not authorize CF-Connecting-IP."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    proxy_nets = _networks("172.16.0.0/12")
    # Generic trusted proxies are configured, but NOT Cloudflare-authorized.
    with _patch_networks(proxy_nets, ()):
        request = _request(
            {
                "x-forwarded-for": "203.0.113.9",
                "cf-connecting-ip": "1.2.3.4",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # CF-Connecting-IP is NOT trusted because the peer is not in
        # Cloudflare-authorized networks. Falls through to XFF.
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_cf_connecting_ip_ignored_without_cloudflare_trust_flag(monkeypatch):
    """Even with Cloudflare networks configured, the flag must be enabled."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.delenv("TRUST_CLOUDFLARE_HEADERS", raising=False)
    proxy_nets = _networks("172.16.0.0/12")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks(proxy_nets, cf_nets):
        request = _request(
            {
                "x-forwarded-for": "203.0.113.9",
                "cf-connecting-ip": "1.2.3.4",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # TRUST_CLOUDFLARE_HEADERS is off → CF not trusted, falls through to XFF
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_cf_connecting_ipv6_pseudo_ipv4(monkeypatch):
    """Pseudo IPv4 "Overwrite headers" yields the real IPv6 address."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request(
            {
                "cf-connecting-ip": "240.1.2.3",
                "cf-connecting-ipv6": "2001:db8:abcd:1234::5",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "2001:db8:abcd:1234::5"
        assert info.source == "cf-connecting-ipv6"


def test_forged_cf_connecting_ipv6_rejected(monkeypatch):
    """A caller-supplied CF-Connecting-IPv6 must not displace the real IP."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request(
            {
                "cf-connecting-ip": "203.0.113.9",
                "cf-connecting-ipv6": "2001:db8:dead:beef::1",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "cf-connecting-ip"


# ---------------------------------------------------------------------------
# Cloudflare adversarial tests (P0: validate CF-derived client)
# ---------------------------------------------------------------------------


def test_cf_malformed_ip_returns_unknown(monkeypatch):
    """Malformed CF-Connecting-IP returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request({"cf-connecting-ip": "garbage"}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_rfc1918_returns_unknown(monkeypatch):
    """RFC1918 CF-Connecting-IP returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request({"cf-connecting-ip": "10.0.0.1"}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_loopback_returns_unknown(monkeypatch):
    """Loopback CF-Connecting-IP returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request({"cf-connecting-ip": "127.0.0.1"}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_ula_returns_unknown(monkeypatch):
    """ULA CF-Connecting-IP returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request({"cf-connecting-ip": "fc00::1"}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_bare_class_e_without_pair_returns_unknown(monkeypatch):
    """Bare Class-E address without valid Pseudo IPv4 pair returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        # CF-Connecting-IP is 240.1.2.3 but no CF-Connecting-IPv6 to corroborate
        request = _request({"cf-connecting-ip": "240.1.2.3"}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_pseudo_ipv4_with_invalid_ipv6_returns_unknown(monkeypatch):
    """Pseudo IPv4 pair with invalid/non-reportable IPv6 returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        # CF-Connecting-IP is Class-E, CF-Connecting-IPv6 is loopback (not IPv6)
        request = _request(
            {
                "cf-connecting-ip": "240.1.2.3",
                "cf-connecting-ipv6": "127.0.0.1",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # 127.0.0.1 is not IPv6, so _pseudo_ipv4_origin returns None.
        # Falls through to direct CF-Connecting-IP path which sees bare Class-E.
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ip"


def test_cf_pseudo_ipv4_with_ula_ipv6_returns_unknown(monkeypatch):
    """Pseudo IPv4 pair with ULA IPv6 returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        # CF-Connecting-IP is Class-E, CF-Connecting-IPv6 is ULA
        request = _request(
            {
                "cf-connecting-ip": "240.1.2.3",
                "cf-connecting-ipv6": "fc00::1",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "cf-connecting-ipv6"


# ---------------------------------------------------------------------------
# Edge cases and adversarial inputs
# ---------------------------------------------------------------------------


def test_mixed_ipv4_ipv6_chain(monkeypatch):
    """Mixed IPv4/IPv6 hops are handled correctly."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12", "fd00::/8")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, fd00::1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"


def test_ipv4_mapped_peer_trust(monkeypatch):
    """An IPv4-mapped IPv6 peer address matches an IPv4 trusted-proxy CIDR."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, ::ffff:172.19.0.1"},
            peer_ip="::ffff:172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"


def test_all_trusted_chain_returns_unknown(monkeypatch):
    """A chain of only trusted addresses resolves to "unknown"."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"


def test_trust_flags_without_trusted_cidr_ignored(monkeypatch):
    """TRUST_PROXY_HEADERS=1 without configured CIDRs does not trust headers."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    with _patch_networks((), ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9"},
            peer_ip="8.8.8.8",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


# ---------------------------------------------------------------------------
# Production pollution regression tests (issue #1036)
# ---------------------------------------------------------------------------


def test_regression_docker_bridge_pollution(monkeypatch):
    """~43K rows recorded 172.19.0.1 as client IP. Now resolves to "unknown"."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        request = _request(
            {"user-agent": "OpenAI-SDK/1.0"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_regression_operator_ula_pollution(monkeypatch):
    """~2.4K rows recorded fdbd:dc0x:: ULA as client IP. Now returns unknown."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("2605:340::/48")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "fdbd:dc02:19:383::153, 2605:340::1"},
            peer_ip="2605:340::1",
        )
        info = get_client_ip_info(request)
        # fdbd:dc02::153 is first untrusted hop but ULA → "unknown"
        assert info.client_ip == "unknown"
        assert info.source == "x-forwarded-for"


# ---------------------------------------------------------------------------
# Provenance is always recorded
# ---------------------------------------------------------------------------


def test_provenance_always_includes_raw_headers(monkeypatch):
    """Raw forwarding headers are always captured for auditability."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        request = _request(
            {
                "x-forwarded-for": "1.2.3.4, 5.6.7.8",
                "x-real-ip": "5.6.7.8",
                "cf-connecting-ip": "1.2.3.4",
            },
            peer_ip="8.8.8.8",
        )
        info = get_client_ip_info(request)
        assert info.x_forwarded_for == "1.2.3.4, 5.6.7.8"
        assert info.x_real_ip == "5.6.7.8"
        assert info.cf_connecting_ip == "1.2.3.4"


# ---------------------------------------------------------------------------
# trusted_proxy_headers provenance semantics
# ---------------------------------------------------------------------------


def test_trusted_proxy_headers_true_when_trusted(monkeypatch):
    """trusted_proxy_headers is True when global trust enabled AND peer is trusted."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.trusted_proxy_headers is True


def test_trusted_proxy_headers_false_when_peer_untrusted(monkeypatch):
    """trusted_proxy_headers is False when peer is not in trusted CIDRs."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    with _patch_networks((), ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9"},
            peer_ip="8.8.8.8",
        )
        info = get_client_ip_info(request)
        assert info.trusted_proxy_headers is False


def test_trusted_proxy_headers_false_when_global_disabled(monkeypatch):
    """trusted_proxy_headers is False when TRUST_PROXY_HEADERS=0."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.trusted_proxy_headers is False


# ---------------------------------------------------------------------------
# Adversarial tests: shared-unknown poisoning
# ---------------------------------------------------------------------------


def test_enforcement_id_never_shared_unknown(monkeypatch):
    """Enforcement ID never returns a shared 'unknown' value."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        # Direct connection from docker bridge (non-routable)
        request = _request({}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        # Provenance identity is "unknown"
        assert info.client_ip == "unknown"
        # But enforcement ID falls back to peer bucket
        enforcement_id = get_client_bucket(request)
        assert enforcement_id != "unknown"
        assert enforcement_id == "172.19.0.1"


def test_enforcement_id_different_peers_different_keys(monkeypatch):
    """Different non-routable peers get different enforcement IDs."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        request1 = _request({}, peer_ip="172.19.0.1")
        request2 = _request({}, peer_ip="10.0.0.1")
        id1 = get_client_bucket(request1)
        id2 = get_client_bucket(request2)
        assert id1 != id2


def test_enforcement_id_ipv6_peer_folds_to_64(monkeypatch):
    """Enforcement ID for IPv6 folds to /64."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        # Public IPv6 peer
        request = _request({}, peer_ip="2001:db8:abcd:1234::1")
        info = get_client_ip_info(request)
        assert info.client_ip == "2001:db8:abcd:1234::1"
        enforcement_id = get_client_bucket(request)
        # IPv6 folds to /64 for bucketing
        assert enforcement_id == "2001:db8:abcd:1234::/64"


def test_enforcement_id_xff_unknown_falls_to_peer(monkeypatch):
    """When XFF provenance fails, enforcement ID falls back to peer."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with _patch_networks(nets, ()):
        # Trusted proxy, but XFF first untrusted hop is private → unknown
        request = _request(
            {"x-forwarded-for": "10.50.0.8, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        # Enforcement ID falls back to peer bucket
        enforcement_id = get_client_bucket(request)
        assert enforcement_id == "172.19.0.1"


# ---------------------------------------------------------------------------
# Adversarial tests: direct-origin / intermediary CF header forgery
# ---------------------------------------------------------------------------


def test_cf_forged_by_direct_origin(monkeypatch):
    """Direct-origin request with forged CF-Connecting-IP is ignored."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    with _patch_networks((), ()):
        # Attacker sends CF-Connecting-IP directly (no trusted peer)
        request = _request(
            {"cf-connecting-ip": "203.0.113.9"},
            peer_ip="1.2.3.4",
        )
        info = get_client_ip_info(request)
        # CF header is ignored because peer is not Cloudflare-authorized
        assert info.client_ip == "1.2.3.4"
        assert info.source == "socket"


def test_cf_forged_by_generic_proxy(monkeypatch):
    """Generic trusted proxy cannot authorize attacker-supplied CF-Connecting-IP.

    When a generic trusted proxy is not Cloudflare-authorized, the CF-Connecting-IP
    header is ignored. The routable generic proxy peer becomes the client.
    """
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    proxy_nets = _networks("203.0.113.1/32")
    # Generic trusted proxies configured, but NOT Cloudflare-authorized
    with _patch_networks(proxy_nets, ()):
        # Attacker behind generic proxy sends forged CF-Connecting-IP
        request = _request(
            {"cf-connecting-ip": "1.2.3.4"},
            peer_ip="203.0.113.1",
        )
        info = get_client_ip_info(request)
        # CF header is NOT trusted because peer is generic, not Cloudflare-authorized.
        # Peer is routable, so it becomes the client.
        assert info.client_ip == "203.0.113.1"
        assert info.source == "socket"


def test_cf_generic_proxy_no_xff(monkeypatch):
    """Generic trusted proxy without XFF falls back to peer."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    proxy_nets = _networks("203.0.113.1/32")
    with _patch_networks(proxy_nets, ()):
        # No XFF, no CF-Connecting-IP
        request = _request({}, peer_ip="203.0.113.1")
        info = get_client_ip_info(request)
        # Peer is routable, becomes client
        assert info.client_ip == "203.0.113.1"
        assert info.source == "socket"


def test_cf_requires_explicit_authorization(monkeypatch):
    """CF-Connecting-IP only trusted when peer is explicitly Cloudflare-authorized."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    # Only Cloudflare-authorized networks, no generic trusted proxies
    cf_nets = _networks("10.0.0.1/32")
    with _patch_networks((), cf_nets):
        request = _request(
            {"cf-connecting-ip": "203.0.113.9"},
            peer_ip="10.0.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "cf-connecting-ip"


# ---------------------------------------------------------------------------
# Bucketing and affinity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("203.0.113.9", "203.0.113.9"),
        ("2001:db8:abcd:1234::5", "2001:db8:abcd:1234::/64"),
        ("2001:db8:abcd:1234:ffff:ffff:ffff:ffff", "2001:db8:abcd:1234::/64"),
        ("2001:db8:abcd:9999::1", "2001:db8:abcd:9999::/64"),
        ("::ffff:192.0.2.1", "192.0.2.1"),
        ("::ffff:203.0.113.9", "203.0.113.9"),
        ("unknown", "unknown"),
        ("", ""),
    ],
)
def test_normalize_ip_bucket(ip, expected):
    """IPv6 buckets on /64; IPv4 and non-addresses bucket on themselves."""
    assert normalize_ip_bucket(ip) == expected


def test_rotating_ipv6_privacy_addresses_share_a_bucket():
    rotated = [
        "2001:db8:abcd:1234::1",
        "2001:db8:abcd:1234:9c2b:1f4e:aa01:7d3f",
        "2001:db8:abcd:1234:4411:beef:0:2",
    ]
    assert len({normalize_ip_bucket(ip) for ip in rotated}) == 1


def test_ipv4_mapped_clients_keep_distinct_buckets():
    buckets = {
        normalize_ip_bucket(ip)
        for ip in ("::ffff:192.0.2.1", "::ffff:203.0.113.9", "::ffff:8.8.8.8")
    }
    assert len(buckets) == 3


def test_derive_affinity_key_falls_through():
    assert derive_affinity_key(None, "8.8.8.8") == "ip:8.8.8.8"
    assert derive_affinity_key(None, "2001:db8::1") == "ip:2001:db8::/64"
    assert derive_affinity_key(None, "unknown") == "ip:unknown"
    assert derive_affinity_key("keyhash", "8.8.8.8") == "keyhash"
    assert derive_affinity_key(None, "8.8.8.8", grant_id="g1") == "grant:g1"


def test_get_client_ip_bucket_normalizes(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    cf_nets = _networks("172.16.0.0/12")
    with _patch_networks((), cf_nets):
        request = _request(
            {"cf-connecting-ip": "2001:db8:abcd:1234::5"},
            peer_ip="172.19.0.1",
        )
        assert get_client_ip_bucket(request) == "2001:db8:abcd:1234::/64"


def test_get_client_enforcement_id_never_shared_unknown(monkeypatch):
    """get_client_enforcement_id never returns a shared 'unknown' value."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with _patch_networks((), ()):
        # Direct connection from docker bridge (non-routable)
        request = _request({}, peer_ip="172.19.0.1")
        enforcement_id = get_client_enforcement_id(request)
        assert enforcement_id != "unknown"
        assert enforcement_id != "ip:unknown"
        assert enforcement_id == "ip:172.19.0.1"


# ---------------------------------------------------------------------------
# Sabotage test: prove adversarial tests catch the vulnerability
# ---------------------------------------------------------------------------


def test_sabotage_pre_fix_spoofing_model():
    """Prove adversarial tests catch the pre-fix spoofable model.

    BEFORE: caller could prepend XFF to spoof any client IP.
    AFTER: spoofed XFF is ignored when peer is not a trusted proxy.
    """
    os.environ["TRUST_PROXY_HEADERS"] = "1"
    with _patch_networks((), ()):
        # Attacker sends XFF claiming to be a victim
        request = _request(
            {"x-forwarded-for": "198.51.100.42"},
            peer_ip="1.2.3.4",  # attacker's real IP, not trusted
        )
        info = get_client_ip_info(request)
        # POST-FIX: spoofed XFF ignored, peer is public so socket peer used
        assert info.client_ip == "1.2.3.4"
        assert info.source == "socket"
    os.environ.pop("TRUST_PROXY_HEADERS", None)
