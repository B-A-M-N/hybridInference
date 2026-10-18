"""Adversarial checks for admitted-login and refusal accounting."""

from __future__ import annotations

import asyncio

import pytest

from serving.config.settings import settings
from serving.utils import login_rate_limit as limiter
from serving.utils.request_ip import ClientIpInfo

ATTACKER = "203.0.113.7"
OWNER = "198.51.100.9"


def _unresolved_provenance() -> ClientIpInfo:
    return ClientIpInfo(
        client_ip="unknown",
        peer_ip="10.0.0.2",
        source="unknown",
        trusted_proxy_headers=False,
        resolved=False,
    )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(settings, "login_rate_limit_per_15min", 2)
    monkeypatch.setattr(settings, "login_account_rate_limit_per_15min", 50)
    monkeypatch.setattr(settings, "login_rate_limit_per_hour_per_ip", 10)
    monkeypatch.setattr(limiter, "_MAX_ACTIVE_BUCKETS", 20_000)
    limiter.reset_login_rate_limit_state()
    yield
    limiter.reset_login_rate_limit_state()


@pytest.mark.asyncio
async def test_sustained_address_flood_does_not_grow_admission_buckets(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(limiter, "_now", lambda: now[0])

    for _ in range(2):
        assert await limiter.check_and_record_login("admin", ATTACKER) == (True, None)

    lengths = (
        len(limiter._account_address_attempts[("admin", ATTACKER)]),
        len(limiter._account_attempts["admin"]),
        len(limiter._ip_attempts[ATTACKER]),
    )
    first_refusal = await limiter.check_and_record_login("admin", ATTACKER)
    assert first_refusal == (False, "email")
    pressure_started_at = limiter._ip_refusal_pressure[ATTACKER].started_at

    for _ in range(500):
        allowed, _ = await limiter.check_and_record_login("admin", ATTACKER)
        assert not allowed

    assert lengths == (
        len(limiter._account_address_attempts[("admin", ATTACKER)]),
        len(limiter._account_attempts["admin"]),
        len(limiter._ip_attempts[ATTACKER]),
    )
    # The separate abuse signal is a bounded scalar and has a fixed expiry.
    pressure = limiter._ip_refusal_pressure[ATTACKER]
    assert pressure.count == settings.login_rate_limit_per_hour_per_ip - 2
    assert pressure.started_at == pressure_started_at == now[0]


@pytest.mark.asyncio
async def test_distributed_guessing_trips_account_limit_without_poisoning_shared_ip(
    monkeypatch,
):
    now = [1_500.0]
    monkeypatch.setattr(limiter, "_now", lambda: now[0])
    settings.login_account_rate_limit_per_15min = 6
    settings.login_rate_limit_per_hour_per_ip = 20

    for suffix in range(3):
        ip = f"203.0.113.{suffix + 1}"
        for _ in range(2):
            assert await limiter.check_and_record_login("admin", ip) == (True, None)

    new_source = "198.51.100.44"
    for _ in range(500):
        assert await limiter.check_and_record_login("admin", new_source) == (
            False,
            "account",
        )
    assert len(limiter._account_attempts["admin"]) == 6

    # Account-wide refusals do not enter the shared IP admission window or
    # refusal-pressure map, so another account behind that address can proceed.
    assert new_source not in limiter._ip_attempts
    assert new_source not in limiter._ip_refusal_pressure
    assert await limiter.check_and_record_login("other@example.com", new_source) == (
        True,
        None,
    )

    now[0] += 15 * 60 + 1
    assert await limiter.check_and_record_login("admin", new_source) == (True, None)


@pytest.mark.asyncio
async def test_address_refusal_escalates_to_ip_and_exposes_shared_ip_collateral():
    settings.login_rate_limit_per_15min = 1
    settings.login_rate_limit_per_hour_per_ip = 3

    assert await limiter.check_and_record_login("admin", ATTACKER) == (True, None)
    assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "email")
    # The second refusal reaches the IP threshold but is still reported at the
    # account/address layer; the next request sees the IP-wide refusal.
    assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "email")
    assert await limiter.check_and_record_login("other@example.com", ATTACKER) == (
        False,
        "ip",
    )
    assert await limiter.check_and_record_login("admin", OWNER) == (True, None)

    # A recognized browser retains the existing independent recovery path.
    assert await limiter.check_and_record_login("admin", ATTACKER, device="trusted-device") == (
        True,
        None,
    )


@pytest.mark.asyncio
async def test_refusal_pressure_expires_from_first_refusal_and_login_recovers(monkeypatch):
    now = [2_000.0]
    monkeypatch.setattr(limiter, "_now", lambda: now[0])
    settings.login_rate_limit_per_15min = 2
    settings.login_rate_limit_per_hour_per_ip = 4

    for _ in range(2):
        assert await limiter.check_and_record_login("admin", ATTACKER) == (True, None)
    assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "email")
    assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "email")
    assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "ip")
    started_at = limiter._ip_refusal_pressure[ATTACKER].started_at

    now[0] += 3_599
    for _ in range(100):
        assert await limiter.check_and_record_login("admin", ATTACKER) == (False, "ip")
    assert limiter._ip_refusal_pressure[ATTACKER].started_at == started_at

    now[0] += 2
    assert await limiter.check_and_record_login("admin", ATTACKER) == (True, None)
    assert ATTACKER not in limiter._ip_refusal_pressure


@pytest.mark.asyncio
async def test_login_bucket_key_cardinality_is_bounded_and_expired_keys_recover(
    monkeypatch,
):
    now = [3_000.0]
    monkeypatch.setattr(limiter, "_now", lambda: now[0])
    monkeypatch.setattr(limiter, "_MAX_ACTIVE_BUCKETS", 2)

    assert await limiter.check_and_record_login("first", "198.51.100.1") == (True, None)
    assert await limiter.check_and_record_login("second", "198.51.100.2") == (True, None)
    assert await limiter.check_and_record_login("third", "198.51.100.3") == (
        False,
        "email",
    )
    assert len(limiter._account_address_attempts) == 2
    assert len(limiter._account_attempts) == 2
    assert len(limiter._ip_attempts) == 2

    now[0] += 3_601
    assert await limiter.check_and_record_login("third", "198.51.100.3") == (True, None)
    assert len(limiter._account_address_attempts) <= 2
    assert len(limiter._account_attempts) <= 2
    assert len(limiter._ip_attempts) <= 2


@pytest.mark.asyncio
async def test_refusal_pressure_map_is_bounded(monkeypatch):
    settings.login_rate_limit_per_15min = 1
    settings.login_rate_limit_per_hour_per_ip = 10

    ips = ["198.51.100.1", "198.51.100.2", "198.51.100.3"]
    for index, ip in enumerate(ips):
        assert await limiter.check_and_record_login(f"user{index}", ip) == (True, None)

    monkeypatch.setattr(limiter, "_MAX_ACTIVE_BUCKETS", 2)

    # These attempts are against existing account/address keys, but begin a
    # separate refusal record for each source IP.
    for index, ip in enumerate(ips):
        if index < 2:
            assert await limiter.check_and_record_login(f"user{index}", ip) == (
                False,
                "email",
            )
        else:
            assert await limiter.check_and_record_login(f"user{index}", ip) == (
                False,
                "ip",
            )
    assert len(limiter._ip_refusal_pressure) == 2


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_exceed_admission_limit():
    settings.login_rate_limit_per_15min = 5
    settings.login_account_rate_limit_per_15min = 100
    settings.login_rate_limit_per_hour_per_ip = 1_000

    results = await asyncio.gather(
        *(limiter.check_and_record_login("admin", ATTACKER) for _ in range(100))
    )

    assert sum(allowed for allowed, _ in results) == 5
    assert sum(reason == "email" for _, reason in results) == 95
    assert len(limiter._account_address_attempts[("admin", ATTACKER)]) == 5
    assert len(limiter._account_attempts["admin"]) == 5
    assert len(limiter._ip_attempts[ATTACKER]) == 5
    assert limiter._ip_refusal_pressure[ATTACKER].count == 95


@pytest.mark.asyncio
async def test_unresolved_login_budget_is_global_bounded_and_recovers(monkeypatch):
    now = [4_000.0]
    monkeypatch.setattr(limiter, "_now", lambda: now[0])
    monkeypatch.setattr(settings, "unresolved_login_rate_limit_per_hour", 2)

    assert await limiter.check_and_record_login("first", _unresolved_provenance()) == (
        True,
        None,
    )
    assert await limiter.check_and_record_login("second", _unresolved_provenance()) == (
        True,
        None,
    )
    for _ in range(100):
        assert await limiter.check_and_record_login("third", _unresolved_provenance()) == (
            False,
            "unresolved",
        )

    assert len(limiter._unresolved_attempts) == 2
    assert not limiter._ip_attempts
    assert not limiter._account_address_attempts

    # A resolved caller has its own address and can still use the resolved
    # protections; unresolved traffic is not assigned the proxy's address.
    assert await limiter.check_and_record_login("third", OWNER) == (True, None)

    now[0] += 3_601
    assert await limiter.check_and_record_login("third", _unresolved_provenance()) == (
        True,
        None,
    )
    assert len(limiter._unresolved_attempts) == 1


@pytest.mark.asyncio
async def test_unresolved_account_and_recognized_device_protections_remain():
    settings.login_account_rate_limit_per_15min = 1
    settings.unresolved_login_rate_limit_per_hour = 10
    unknown = _unresolved_provenance()

    assert await limiter.check_and_record_login("owner", unknown) == (True, None)
    assert await limiter.check_and_record_login("owner", unknown) == (False, "account")

    # A recognized-device budget remains independent of unresolved traffic and
    # account-wide admission history, as in the upstream limiter contract.
    assert await limiter.check_and_record_login("owner", unknown, device="browser") == (
        True,
        None,
    )
