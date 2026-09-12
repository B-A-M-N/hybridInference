"""IPv6 clients must not escape per-IP limits by rotating within their /64."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from serving.config.settings import settings
from serving.utils.login_rate_limit import (
    check_and_record_login,
    reset_login_rate_limit_state,
)
from serving.utils.signup_rate_limit import (
    check_and_record_signup,
    reset_signup_rate_limit_state,
)

# Distinct /128 addresses inside one delegated /64 — what an attacker rotates
# through, and what RFC 4941 privacy addressing produces on its own.
_SAME_PREFIX = [f"2001:db8:abcd:1234::{n:x}" for n in range(1, 64)]
_OTHER_PREFIX = "2001:db8:abcd:9999::1"


def _request(ip: str) -> SimpleNamespace:
    return SimpleNamespace(headers={}, client=SimpleNamespace(host=ip))


@pytest.fixture(autouse=True)
def _clean_limiter_state():
    reset_signup_rate_limit_state()
    reset_login_rate_limit_state()
    yield
    reset_signup_rate_limit_state()
    reset_login_rate_limit_state()


@pytest.mark.asyncio
async def test_signup_limit_survives_ipv6_rotation():
    """Rotating the low 64 bits shares one signup bucket.

    Every attempt lands inside the same hour, so the hourly window trips
    first even though the daily cap is higher.
    """
    per_hour = settings.signup_rate_limit_per_hour
    addresses = _SAME_PREFIX[: per_hour + 1]
    assert len(addresses) == per_hour + 1

    results = [await check_and_record_signup(_request(ip)) for ip in addresses]

    assert all(allowed for allowed, _ in results[:per_hour])
    assert results[-1] == (False, "hour")


@pytest.mark.asyncio
async def test_signup_limit_isolates_distinct_ipv6_prefixes():
    """A different /64 is a separate bucket — no collateral throttling."""
    # Fill the hourly window from the first prefix.
    for _ in range(settings.signup_rate_limit_per_hour + 1):
        await check_and_record_signup(_request(_SAME_PREFIX[0]))
    assert (await check_and_record_signup(_request(_SAME_PREFIX[0])))[0] is False

    # A client from a separate /64 is unaffected.
    assert (await check_and_record_signup(_request(_OTHER_PREFIX)))[0] is True


@pytest.mark.asyncio
async def test_login_per_ip_limit_survives_ipv6_rotation():
    """Per-IP login limit folds to /64 so rotation cannot dodge it."""
    per_ip = settings.login_rate_limit_per_hour_per_ip

    # Use different emails so we're testing the per-IP limit, not per-email
    for i, ip in enumerate(_SAME_PREFIX[: per_ip + 1]):
        email = f"user{i}@example.com"
        result = await check_and_record_login(email, _request(ip))
        if not result[0]:
            assert result == (False, "ip")
            break
    else:
        pytest.fail("per_ip limit did not trip across a full /64")


@pytest.mark.asyncio
async def test_login_ipv4_buckets_remain_per_address():
    """IPv4 clients keep per-address buckets — /64 folding does not apply."""
    email = "stable@example.com"

    first = await check_and_record_login(email, _request("203.0.113.20"))
    second = await check_and_record_login(email, _request("203.0.113.21"))
    # Each address is its own bucket, so both succeed independently.
    assert first[0] is True
    assert second[0] is True
