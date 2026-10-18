"""In-memory sliding-window rate limits for login and first-run setup.

Login distinguishes admitted attempts from refusals. An admitted login is one
allowed through to password verification; only admitted attempts enter the
account/address, account-wide, and IP-wide windows. A refusal caused by the
account/address window increments a separate, saturating per-IP counter. The
IP check combines that pressure with admitted IP attempts, preserving abuse
escalation without growing admission queues or extending account-wide windows.

Account-wide refusals do not add IP pressure, which limits collateral effects
for other accounts behind a shared address. Once address refusals reach the
IP-wide threshold, the existing IP-wide policy applies to every account at
that address. Recognized devices retain their independent window and do not
consume or get blocked by the other login windows.

Login bucket maps are bounded. Attacker-populated maps (account/address,
account, IP) evict their oldest *expired* key to make room, and refuse a new
key only while every resident key is still live; they never evict a live
entry, so a flooding attacker cannot lock out identities that are actively
signing in. The recognized-device map is different: it is the owner's escape
hatch from every other window, so a full device map evicts the oldest live
entry rather than refusing a recognized browser. Device keys are additionally
scoped per account, so one account cannot exhaust the device map and disable
recognized-browser sign-in for everyone else.

Unresolved provenance (no routable client address can be attributed, e.g.
every browser behind the console proxy in a default Compose deployment) does
not receive a synthetic per-client address. It keeps the per-account/address
window, keyed on the shared ``"unresolved"`` address bucket together with the
account-wide and the coarse hourly admission budgets. Setup-code accounting
is unchanged. State is process-local and lost on restart, as with
:mod:`serving.utils.signup_rate_limit`.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from serving.config.settings import settings
from serving.utils.request_ip import ClientIpInfo, get_client_ip_info, normalize_ip_bucket

if TYPE_CHECKING:
    from fastapi import Request

_FIFTEEN_MIN_SECONDS = 15 * 60
_HOUR_SECONDS = 3600
_SWEEP_EVERY = 1024
_CAPACITY_SWEEP_INTERVAL_SECONDS = 60
# Bound attacker-controlled login identities in each process-local map. When a
# map is full, expired keys are swept and then the oldest expired key's slot is
# reused; a new key is refused only while every resident key is still live.
# Live entries are never evicted from the attacker-populated maps, so active
# sign-ins keep their recorded state.
_MAX_ACTIVE_BUCKETS = 20_000

# Wrong first-run setup codes an IP may send in any 15-minute window before
# further wrong ones are answered 429.
SETUP_ATTEMPTS_PER_15MIN = 10

# Login admissions, keyed by device, account/address, account, and address.
_device_attempts: dict[tuple[str, str], deque[float]] = {}
_account_address_attempts: dict[tuple[str, str], deque[float]] = {}
_account_attempts: dict[str, deque[float]] = {}
_ip_attempts: dict[str, deque[float]] = {}
_unresolved_attempts: deque[float] = deque()

# Shared address bucket for clients whose provenance could not be resolved.
# There is no information to distinguish them, so they share one bucket per
# account rather than one bucket per synthetic client address.
_UNRESOLVED_ADDRESS_BUCKET = "unresolved"


@dataclass(slots=True)
class _RefusalPressure:
    """Constant-size refusal accounting with an expiry fixed at first refusal."""

    started_at: float
    count: int = 0


# Refusal pressure is separate from every admission queue. Its count is capped
# at the configured IP limit, and repeated refusals never move started_at.
_ip_refusal_pressure: dict[str, _RefusalPressure] = {}
_setup_attempts: dict[str, deque[float]] = {}
_lock = asyncio.Lock()
_sweep_counter = 0
_setup_sweep_counter = 0
_last_capacity_sweep: dict[str, float] = {}

_Key = TypeVar("_Key")


def _now() -> float:
    return time.time()


def _sweep_inactive(buckets: dict[_Key, deque[float]], cutoff: float) -> None:
    for key in list(buckets):
        bucket = buckets[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if not bucket:
            del buckets[key]


def _prune_bucket(
    buckets: dict[_Key, deque[float]], key: _Key, cutoff: float
) -> deque[float] | None:
    """Expire one bucket and remove its key when no admissions remain."""
    bucket = buckets.get(key)
    if bucket is None:
        return None
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if not bucket:
        del buckets[key]
        return None
    return bucket


def _evict_oldest(buckets: dict[_Key, deque[float]]) -> None:
    """Drop the entry whose oldest admission is earliest (a live-entry victim)."""
    oldest_key = min(buckets, key=lambda k: buckets[k][0])
    del buckets[oldest_key]


def _can_add_bucket_key(
    buckets: dict[_Key, deque[float]],
    key: _Key,
    cutoff: float,
    *,
    now: float,
    scope: str,
) -> bool:
    """Check map capacity, sweeping expired keys at most once per interval.

    A full map first sweeps expired keys. If the map is still at capacity it
    evicts the entry whose oldest admission is earliest rather than refusing
    every new key: map saturation must never become a global denial of
    service, and it must never disable a live identity's recorded state.
    """
    if key in buckets or len(buckets) < _MAX_ACTIVE_BUCKETS:
        return True
    if now - _last_capacity_sweep.get(scope, float("-inf")) >= _CAPACITY_SWEEP_INTERVAL_SECONDS:
        _sweep_inactive(buckets, cutoff)
        _last_capacity_sweep[scope] = now
    if len(buckets) < _MAX_ACTIVE_BUCKETS:
        return True
    _evict_oldest(buckets)
    return True


def _sweep_expired_refusals(now: float) -> None:
    for ip_key, pressure in list(_ip_refusal_pressure.items()):
        if now - pressure.started_at >= _HOUR_SECONDS:
            del _ip_refusal_pressure[ip_key]


def _active_refusal_count(ip_key: str, now: float) -> int:
    pressure = _ip_refusal_pressure.get(ip_key)
    if pressure is None:
        return 0
    if now - pressure.started_at >= _HOUR_SECONDS:
        del _ip_refusal_pressure[ip_key]
        return 0
    return pressure.count


def _record_address_refusal(ip_key: str, now: float, per_ip: int) -> bool:
    """Record one address-limit refusal; return False if its map is full."""
    pressure = _ip_refusal_pressure.get(ip_key)
    if pressure is None:
        if len(_ip_refusal_pressure) >= _MAX_ACTIVE_BUCKETS:
            last_sweep = _last_capacity_sweep.get("refusal", float("-inf"))
            if now - last_sweep >= _CAPACITY_SWEEP_INTERVAL_SECONDS:
                _sweep_expired_refusals(now)
                _last_capacity_sweep["refusal"] = now
        if len(_ip_refusal_pressure) >= _MAX_ACTIVE_BUCKETS:
            return False
        pressure = _RefusalPressure(started_at=now)
        _ip_refusal_pressure[ip_key] = pressure

    pressure.count = min(pressure.count + 1, per_ip)
    return True


async def check_and_record_login(
    email: str, provenance: str | ClientIpInfo | Request, *, device: str | None = None
) -> tuple[bool, str | None]:
    """Return whether a login may proceed, recording only admitted attempts.

    *email* is the normalized login identifier (email or login name).
    *provenance* is a resolved address, :class:`ClientIpInfo`, or request.
    A valid recognized *device* uses only its own 15-minute window. Otherwise
    the limiter checks account/address (15 minutes), account-wide (15
    minutes), and address-wide (one hour) limits.

    If provenance is unresolved, no address identity is inferred; the attempt
    is keyed on the shared ``"unresolved"`` address bucket for the
    account/address window, uses the account-wide limit, and also draws on a
    separate coarse hourly admission budget shared by every unresolved
    client. A result reason is ``"device"``, ``"email"`` (account/address),
    ``"account"``, ``"ip"``, or ``"unresolved"``. Only resolved-address
    refusals caused by the account/address window add to fixed-expiry IP
    pressure. IP-refused and account-wide-refused requests do not change
    stored counts. The check, pressure update, and admissions are atomic
    under one lock.
    """
    global _sweep_counter

    now = _now()
    account_cutoff = now - _FIFTEEN_MIN_SECONDS
    ip_cutoff = now - _HOUR_SECONDS
    per_account_address = settings.login_rate_limit_per_15min
    per_account = settings.login_account_rate_limit_per_15min
    per_ip = settings.login_rate_limit_per_hour_per_ip
    account_key = email.strip().lower()
    if isinstance(provenance, ClientIpInfo):
        ip = provenance.client_ip
        resolved = provenance.resolved
    elif isinstance(provenance, str):
        ip = provenance
        # Plain-address callers are resolved by contract. Callers that need to
        # distinguish an unresolved client must pass ClientIpInfo.
        resolved = True
    else:
        info = get_client_ip_info(provenance)
        ip = info.client_ip
        resolved = info.resolved
    ip_key = normalize_ip_bucket(ip) if resolved else _UNRESOLVED_ADDRESS_BUCKET

    async with _lock:
        _sweep_counter += 1
        if _sweep_counter >= _SWEEP_EVERY:
            _sweep_counter = 0
            _sweep_inactive(_device_attempts, account_cutoff)
            _sweep_inactive(_account_address_attempts, account_cutoff)
            _sweep_inactive(_account_attempts, account_cutoff)
            _sweep_inactive(_ip_attempts, ip_cutoff)
            while _unresolved_attempts and _unresolved_attempts[0] < ip_cutoff:
                _unresolved_attempts.popleft()
            _sweep_expired_refusals(now)
            _last_capacity_sweep.update(
                device=now,
                account_address=now,
                account=now,
                ip=now,
                refusal=now,
            )

        if device is not None:
            # The device window is the owner's escape hatch from every other
            # window. Device keys are scoped per account so one account cannot
            # exhaust the map, and a full map evicts its oldest entry rather
            # than refusing a recognized browser.
            device_key = (account_key, device)
            device_bucket = _prune_bucket(_device_attempts, device_key, account_cutoff)
            if device_bucket is not None and len(device_bucket) >= per_account_address:
                return False, "device"
            _can_add_bucket_key(
                _device_attempts,
                device_key,
                account_cutoff,
                now=now,
                scope="device",
            )
            _device_attempts.setdefault(device_key, deque()).append(now)
            return True, None

        address_key = (account_key, ip_key)
        address_bucket = _prune_bucket(_account_address_attempts, address_key, account_cutoff)
        account_bucket = _prune_bucket(_account_attempts, account_key, account_cutoff)
        resolved_ip_bucket = _prune_bucket(_ip_attempts, ip_key, ip_cutoff) if resolved else None
        address_count = len(address_bucket) if address_bucket is not None else 0
        account_count = len(account_bucket) if account_bucket is not None else 0
        ip_count = len(resolved_ip_bucket) if resolved_ip_bucket is not None else 0
        refusal_count = _active_refusal_count(ip_key, now) if resolved else 0

        # Abuse escalation combines admitted IP traffic and bounded address
        # refusal pressure. Requests already refused by this policy add no
        # more pressure and cannot extend either window. Unresolved provenance
        # has no per-client address, so it is not subject to the per-IP window.
        if resolved and ip_count + refusal_count >= per_ip:
            return False, "ip"

        if address_count >= per_account_address:
            if resolved and not _record_address_refusal(ip_key, now, per_ip):
                return False, "ip"
            return False, "email"

        if not resolved:
            while _unresolved_attempts and _unresolved_attempts[0] < ip_cutoff:
                _unresolved_attempts.popleft()
            if len(_unresolved_attempts) >= settings.unresolved_login_rate_limit_per_hour:
                return False, "unresolved"

        # Account-wide checks apply after the address-specific limit. A
        # refusal here does not contribute to the IP-wide abuse counter.
        if account_count >= per_account:
            return False, "account"

        # Check every map before writing to any of them, so capacity handling
        # cannot leave a partial admission in one of the other windows. A full
        # map evicts its oldest entry; it never refuses solely for capacity.
        _can_add_bucket_key(
            _account_address_attempts,
            address_key,
            account_cutoff,
            now=now,
            scope="account_address",
        )
        _can_add_bucket_key(
            _account_attempts,
            account_key,
            account_cutoff,
            now=now,
            scope="account",
        )
        if resolved:
            _can_add_bucket_key(
                _ip_attempts,
                ip_key,
                ip_cutoff,
                now=now,
                scope="ip",
            )

        _account_address_attempts.setdefault(address_key, deque()).append(now)
        _account_attempts.setdefault(account_key, deque()).append(now)
        if resolved:
            _ip_attempts.setdefault(ip_key, deque()).append(now)
        else:
            _unresolved_attempts.append(now)

    return True, None


async def record_failed_setup_attempt(ip: str) -> bool:
    """Record a wrong first-run setup code and report whether the IP is under limit.

    One sliding 15-minute window per client IP (IPv6 bucketed to its /64).
    Returns False once the IP had already sent ``SETUP_ATTEMPTS_PER_15MIN``
    wrong codes in the window; the caller answers 429 instead of 403. Only
    failures are recorded: a correct code never reaches this function.
    """
    global _setup_sweep_counter

    now = _now()
    cutoff = now - _FIFTEEN_MIN_SECONDS
    ip_key = normalize_ip_bucket(ip)

    async with _lock:
        _setup_sweep_counter += 1
        if _setup_sweep_counter >= _SWEEP_EVERY:
            _setup_sweep_counter = 0
            _sweep_inactive(_setup_attempts, cutoff)

        bucket = _setup_attempts.setdefault(ip_key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        count = len(bucket)
        bucket.append(now)

    return count < SETUP_ATTEMPTS_PER_15MIN


def reset_login_rate_limit_state() -> None:
    """Wipe all recorded attempts (login and setup). Test-only helper."""
    global _sweep_counter, _setup_sweep_counter
    _device_attempts.clear()
    _account_address_attempts.clear()
    _account_attempts.clear()
    _ip_attempts.clear()
    _unresolved_attempts.clear()
    _ip_refusal_pressure.clear()
    _setup_attempts.clear()
    _last_capacity_sweep.clear()
    _sweep_counter = 0
    _setup_sweep_counter = 0
