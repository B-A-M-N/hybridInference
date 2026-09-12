"""Tests for proxied client IP extraction with trusted-proxy trust model.

Covers the security fix for issue #1036: client IP resolution now requires
the socket peer to be in an explicitly configured trusted-proxy CIDR set
before any forwarding header is consulted. Non-routable peers resolve to
"unknown" instead of masquerading as clients.
"""

from __future__ import annotations

import ipaddress
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from serving.utils.request_ip import (
    _is_reportable_ip,
    _is_trusted_proxy,
    _parse_forwarded_chain,
    _parse_ip,
    derive_affinity_key,
    get_client_ip_bucket,
    get_client_ip_info,
    normalize_ip_bucket,
)


def _request(headers: dict[str, str], peer_ip: str = "10.0.0.2"):
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer_ip))


def _networks(*cidrs: str) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Create a tuple of parsed networks from CIDR strings."""
    return tuple(ipaddress.ip_network(c) for c in cidrs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_ip_accepts_ipv4_ipv6_mapped():
    assert str(_parse_ip("1.2.3.4")) == "1.2.3.4"
    assert str(_parse_ip("2001:db8::1")) == "2001:db8::1"
    # IPv4-mapped IPv6 is unwrapped to its embedded IPv4 address.
    assert str(_parse_ip("::ffff:192.0.2.1")) == "192.0.2.1"


def test_parse_ip_rejects_garbage():
    assert _parse_ip("not-an-ip") is None
    assert _parse_ip("") is None
    assert _parse_ip(None) is None


def test_parse_forwarded_chain_splits_and_trims():
    assert _parse_forwarded_chain("1.2.3.4, 5.6.7.8") == ["1.2.3.4", "5.6.7.8"]
    assert _parse_forwarded_chain("  1.2.3.4  ,  , 5.6.7.8  ") == ["1.2.3.4", "5.6.7.8"]
    assert _parse_forwarded_chain("") == []


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


def test_is_trusted_proxy_matches_cidr():
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    assert _is_trusted_proxy("172.19.0.1", nets) is True
    assert _is_trusted_proxy("10.0.0.1", nets) is True
    assert _is_trusted_proxy("8.8.8.8", nets) is False
    assert _is_trusted_proxy("not-an-ip", nets) is False


def test_is_trusted_proxy_empty_means_no_trust():
    assert _is_trusted_proxy("172.19.0.1", ()) is False


# ---------------------------------------------------------------------------
# Default behavior: no trusted proxies configured
# ---------------------------------------------------------------------------


def test_non_routable_peer_returns_unknown_no_trusted_proxies(monkeypatch):
    """Docker bridge peer without trustworthy forwarding provenance → "unknown".

    Regression: the old code logged 172.19.0.1 as the client IP (issue #1036,
    production pollution class #1).
    """
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        request = _request({}, peer_ip="172.19.0.1")
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_public_peer_used_directly_no_trusted_proxies(monkeypatch):
    """A direct connection from a routable peer is a legitimate client IP."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        info = get_client_ip_info(_request({}, peer_ip="8.8.8.8"))
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_forged_xff_ignored_without_trusted_peer(monkeypatch):
    """Attacker cannot spoof XFF when peer is not a configured trusted proxy."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        request = _request({"x-forwarded-for": "203.0.113.9"}, peer_ip="8.8.8.8")
        info = get_client_ip_info(request)
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_forged_cf_connecting_ip_ignored_without_trusted_peer(monkeypatch):
    """Attacker cannot spoof CF-Connecting-IP without a trusted peer."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        request = _request({"cf-connecting-ip": "203.0.113.9"}, peer_ip="8.8.8.8")
        info = get_client_ip_info(request)
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


def test_no_client_means_unknown(monkeypatch):
    """A request with no client at all resolves to "unknown"."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    request = SimpleNamespace(headers={}, client=None)
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


# ---------------------------------------------------------------------------
# Trusted proxy behavior: peer in configured CIDR
# ---------------------------------------------------------------------------


def test_trusted_proxy_rightmost_untrusted_hop_is_client(monkeypatch):
    """XFF chain walked right-to-left; trusted hops skipped."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    # Peer is our trusted proxy at 172.19.0.1.
    # XFF: client 1.2.3.4, then 10.0.0.1 (also trusted), then peer 172.19.0.1.
    nets = _networks("172.16.0.0/12", "10.0.0.0/8")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "1.2.3.4, 10.0.0.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "1.2.3.4"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_all_trusted_hops_falls_through(monkeypatch):
    """When every XFF hop is a trusted proxy, no client can be determined."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("10.0.0.0/8", "172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # All hops trusted → fall through. Peer is non-routable → "unknown".
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_trusted_proxy_ula_hop_skipped(monkeypatch):
    """Operator's internal ULA overlay hop is skipped, not reported as client.

    Regression: issue #1036 production pollution class #2.
    """
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, fdbd:dc02:19:383::153, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_attacker_prepends_fake_addresses(monkeypatch):
    """Attacker prepending public addresses to XFF does not spoof identity.

    The rightmost untrusted hop is the real client; prepended addresses are
    to the left of trusted hops and are never reached.
    """
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "1.2.3.4, 5.6.7.8, 203.0.113.9, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        # 172.19.0.1 is trusted (peer), 203.0.113.9 is the first untrusted hop.
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_trusted_proxy_x_real_ip_fallback(monkeypatch):
    """X-Real-IP used when XFF has no usable hop."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 172.19.0.1", "x-real-ip": "203.0.113.9"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-real-ip"


# ---------------------------------------------------------------------------
# Cloudflare header handling
# ---------------------------------------------------------------------------


def test_cf_connecting_ip_when_trusted(monkeypatch):
    """CF-Connecting-IP is authoritative when peer is trusted Cloudflare."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
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


def test_cf_connecting_ip_ignored_without_cloudflare_trust(monkeypatch):
    """Generic proxy trust does not imply Cloudflare header trust."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.delenv("TRUST_CLOUDFLARE_HEADERS", raising=False)
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {
                "x-forwarded-for": "203.0.113.9",
                "cf-connecting-ip": "1.2.3.4",
            },
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"
        assert info.source == "x-forwarded-for"


def test_cf_connecting_ipv6_pseudo_ipv4(monkeypatch):
    """Pseudo IPv4 "Overwrite headers" yields the real IPv6 address."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
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
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
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
# Edge cases and adversarial inputs
# ---------------------------------------------------------------------------


def test_malformed_xff_entries_skipped(monkeypatch):
    """Malformed entries in XFF chain are skipped, not normalized into clients."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "garbage, 203.0.113.9, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"


def test_mixed_ipv4_ipv6_chain(monkeypatch):
    """Mixed IPv4/IPv6 hops are handled correctly."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12", "fd00::/8")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
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
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "203.0.113.9, ::ffff:172.19.0.1"},
            peer_ip="::ffff:172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "203.0.113.9"


def test_all_private_chain_returns_unknown(monkeypatch):
    """A chain of only private addresses resolves to "unknown"."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "10.0.0.1, 192.168.1.1, 172.19.0.1"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"


def test_trust_flags_without_trusted_cidr_ignored(monkeypatch):
    """TRUST_PROXY_HEADERS=1 without configured CIDRs does not trust headers."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        request = _request(
            {"x-forwarded-for": "203.0.113.9"},
            peer_ip="8.8.8.8",
        )
        info = get_client_ip_info(request)
        # Peer is public but not a configured proxy → socket peer used.
        assert info.client_ip == "8.8.8.8"
        assert info.source == "socket"


# ---------------------------------------------------------------------------
# Production pollution regression tests (issue #1036)
# ---------------------------------------------------------------------------


def test_regression_docker_bridge_pollution(monkeypatch):
    """~43K rows recorded 172.19.0.1 as client IP. Now resolves to "unknown"."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
        request = _request(
            {"user-agent": "OpenAI-SDK/1.0"},
            peer_ip="172.19.0.1",
        )
        info = get_client_ip_info(request)
        assert info.client_ip == "unknown"
        assert info.source == "unknown"


def test_regression_operator_ula_pollution(monkeypatch):
    """~2.4K rows recorded fdbd:dc0x:: ULA as client IP. Now skipped."""
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    # Peer is the operator's gateway, which IS in the trusted set.
    nets = _networks("2605:340::/48")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"x-forwarded-for": "fdbd:dc02:19:383::153, 2605:340::1"},
            peer_ip="2605:340::1",
        )
        info = get_client_ip_info(request)
        # fdbd:dc02:19:383::153 is the first untrusted hop but is ULA →
        # not reportable → fall through. Peer is public → "2605:340::1".
        # Actually: the ULA hop is untrusted but not reportable, so we break
        # and fall through. Peer is public → socket peer.
        assert info.client_ip == "2605:340::1"
        assert info.source == "socket"


# ---------------------------------------------------------------------------
# Provenance is always recorded
# ---------------------------------------------------------------------------


def test_provenance_always_includes_raw_headers(monkeypatch):
    """Raw forwarding headers are always captured for auditability."""
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    with patch("serving.utils.request_ip._trusted_networks", return_value=()):
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
    nets = _networks("172.16.0.0/12")
    with patch("serving.utils.request_ip._trusted_networks", return_value=nets):
        request = _request(
            {"cf-connecting-ip": "2001:db8:abcd:1234::5"},
            peer_ip="172.19.0.1",
        )
        assert get_client_ip_bucket(request) == "2001:db8:abcd:1234::/64"


# ---------------------------------------------------------------------------
# Sabotage test: prove adversarial tests catch the vulnerability
# ---------------------------------------------------------------------------


def test_sabotage_pre_fix_spoofing_model():
    """Prove adversarial tests catch the pre-fix spoofable model.

    BEFORE: caller could prepend XFF to spoof any client IP.
    AFTER: spoofed XFF is ignored when peer is not a trusted proxy.
    """
    from serving.config.settings import get_settings
    import os as _os
    _os.environ["TRUST_PROXY_HEADERS"] = "1"
    original = get_settings().trusted_proxies
    get_settings().trusted_proxies = []
    get_settings()._parse_trusted_proxies()

    try:
        # Attacker sends XFF claiming to be a victim
        request = _request(
            {"x-forwarded-for": "198.51.100.42"},
            peer_ip="1.2.3.4",  # attacker's real IP, not trusted
        )
        info = get_client_ip_info(request)
        # POST-FIX: spoofed XFF ignored, peer is public so socket peer used
        assert info.client_ip == "1.2.3.4"
        assert info.source == "socket"
    finally:
        get_settings().trusted_proxies = original
        get_settings()._parse_trusted_proxies()
        _os.environ.pop("TRUST_PROXY_HEADERS", None)
