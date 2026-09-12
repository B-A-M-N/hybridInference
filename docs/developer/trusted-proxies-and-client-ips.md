# Trusted Proxies and Client IPs

Almost every deployment puts something in front of the gateway — a CDN, a
reverse proxy, a tunnel, a load balancer. Once that is true, the socket peer
the gateway sees is the proxy, not the caller, and the caller's address only
survives in a request header that anyone can also set by hand.

`apps/backend/serving/utils/request_ip.py` is the single place that decides
which address to believe. Everything downstream — request logs, per-IP rate
limits, the auth-failure blocklist, sticky routing affinity, and the rows
written to `api_logs` — reads its answer. This page describes what that module
does, how to configure it, and what ends up stored as a result.

## The server underneath

For the module to be the single decision point, the server it runs inside must
not make the same decision first. uvicorn carries its own proxy-header
handling, and it defaults **on**: unless told otherwise, it rewrites
`request.client` — the socket peer as the application sees it — and the URL
scheme from `X-Forwarded-For` / `X-Forwarded-Proto` whenever the TCP peer is
in `--forwarded-allow-ips` (default `127.0.0.1`, also settable through the
`FORWARDED_ALLOW_IPS` environment variable). That rewrite happens before any
application code runs, upstream of everything this page describes, so left
enabled it hands `request_ip.py` an already-forged "socket peer" while
`TRUST_PROXY_HEADERS=0` promises that no header influences the result. The
Docker image shipped for a while with the widest form of this —
`--proxy-headers --forwarded-allow-ips "*"` — which made the leftmost
`X-Forwarded-For` entry, i.e. whatever the caller wrote, the socket peer on
every request (HarvardMadSys/freeInference#72).

Every launch configuration in this repository therefore passes
`--no-proxy-headers` explicitly — `deploy/docker/Dockerfile.backend`
and both systemd units —
and `tests/unit/deploy/test_uvicorn_proxy_headers.py` fails if one stops doing
so. If you run the gateway under your own process manager, carry the flag
over: deleting the two flags is not enough, because the default is on.

Two consequences of the server never interpreting forwarded headers:

- `request.url.scheme` is always `http` behind a TLS-terminating proxy. Set
  `BASE_URL` (see `.env.example`) so absolute URLs — signup verification and
  password-reset email links — do not fall back to the request scheme.
- `peer_ip` below is the genuine TCP peer again, which is what makes it usable
  as the un-forgeable anchor the rest of this page treats it as.

## Trust configuration: the `trusted_proxies` list

Forwarding headers (`X-Forwarded-For`, `CF-Connecting-IP`, etc.) are
**attacker-controlled on every request** that reaches the origin without passing
through the proxy you trust. The gateway therefore requires **two independent
facts** to be established before any header influences the result:

1. The `TRUST_PROXY_HEADERS` environment variable is set to `1`.
2. The immediate socket peer falls inside a CIDR range listed in the
   `trusted_proxies` setting.

Both must be true. Setting `TRUST_PROXY_HEADERS=1` alone does nothing if
`trusted_proxies` is empty — and it defaults to empty, so forwarding headers
are **never trusted** unless the operator explicitly opts in.

```python
# settings.py (or via env var)
trusted_proxies = ["172.17.0.0/16", "10.0.0.0/8"]  # Docker bridge, internal
```

Invalid CIDRs fail configuration at startup rather than silently weakening
identity resolution. Parsed networks are cached in `trusted_proxies_parsed`
and validated once at startup.

`TRUST_CLOUDFLARE_HEADERS` additionally gates `CF-Connecting-IP`: it is honored
only when `TRUST_PROXY_HEADERS=1` **and** the peer is in `trusted_proxies` **and**
`TRUST_CLOUDFLARE_HEADERS=1`. A non-Cloudflare proxy may rewrite
`X-Forwarded-For` correctly while forwarding a client-supplied
`CF-Connecting-IP` untouched, so the two facts are gated apart.

### Why this matters

Without this gate, a client can send `X-Forwarded-For: <anything>` and the
gateway logs and rate-limits on the attacker-chosen address. The old boolean
flags (issue #1036) asserted only that *some* proxy exists; they did not
restrict which peer may assert forwarding provenance. The new model makes
trust explicit and fail-closed.

## Resolution order

`get_client_ip_info()` returns a frozen `ClientIpInfo` with the resolved
`client_ip`, the `peer_ip` it was resolved against, and a `source` label naming
the rung that won. Resolution walks the trust boundary correctly:

1. **Peer is not in `trusted_proxies`** — forwarding headers are ignored
   entirely. The socket peer is used if it is routable; otherwise
   `"unknown"` is returned.
2. **`CF-Connecting-IP`** — only when the peer is trusted **and**
   `TRUST_CLOUDFLARE_HEADERS=1`. Cloudflare overwrites this header on every
   request, so it cannot be forged by the client. A corroborated Pseudo IPv4
   pair yields the real IPv6 address from `CF-Connecting-IPv6`.
3. **`X-Forwarded-For`** — walked **right-to-left**. The rightmost entry is the
   address the immediate proxy saw; each entry to the left was added by an
   earlier hop. Trusted-proxy hops are skipped; the first untrusted **routable**
   hop is the real client. Non-routable untrusted hops (RFC 1918, ULA, etc.)
   are skipped because they cannot represent a remote client — this is what
   prevented the internal-overlay leak (issue #1036).
4. **`X-Real-IP`** — when it is routable.
5. **The socket peer** — a direct connection, or the last resort when no
   forwarded hop is usable. If the peer itself is non-routable, the result is
   `"unknown"`.

When `trusted_proxies` is empty (the default), forwarding headers are never
trusted, and the socket peer is the only network fact used. This is the
fail-closed default.

### Example: multi-hop chain

```
XFF: "1.2.3.4, fdbd:dc02::153, 10.0.0.1, 172.16.0.5"
Peer (our proxy): 172.16.0.5  ← in trusted_proxies
```

Walking right-to-left:

| Hop | Trusted? | Routable? | Action |
|-----|----------|-----------|--------|
| 172.16.0.5 | yes (our peer) | no | skip (trusted) |
| 10.0.0.1 | yes | no | skip (trusted) |
| fdbd:dc02::153 | no | no (ULA) | skip (unroutable) |
| 1.2.3.4 | no | yes | **return as client** |

### Example: attacker prepends fake addresses

```
XFF: "8.8.8.8, 1.2.3.4, 203.0.113.9, 172.16.0.5"
Peer (our proxy): 172.16.0.5  ← in trusted_proxies
```

Walking right-to-left:

| Hop | Trusted? | Routable? | Action |
|-----|----------|-----------|--------|
| 172.16.0.5 | yes | no | skip (trusted) |
| 203.0.113.9 | no | yes | **return as client** |

The attacker's prepended `8.8.8.8` and `1.2.3.4` sit to the left of the first
untrusted routable hop and are never reached. The old leftmost-trust model
would have returned `8.8.8.8`.

### What counts as routable

`_is_reportable_ip()` rejects anything unparseable, plus loopback, link-local,
multicast and unspecified addresses, and these networks:

```text
10.0.0.0/8        172.16.0.0/12     192.168.0.0/16    (RFC 1918)
100.64.0.0/10     (CGNAT, RFC 6598)
fc00::/7          (IPv6 unique local)
```

An IPv4-mapped IPv6 literal is judged by its embedded IPv4 address, so a mapped
private peer is still rejected.

The list is written out explicitly rather than delegating to
`ipaddress.is_private` / `is_global`, because those reclassified the
documentation (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) and
benchmark (`198.18.0.0/15`) ranges between CPython 3.12.4 and 3.13; hard-coding
the stable RFC ranges keeps IP resolution from depending on the interpreter
version.

## The "unknown" outcome

When the gateway cannot determine a trustworthy routable client address, it
returns `"unknown"` rather than masquerading an internal address as a client.
This is correct: if the socket peer is a Docker bridge and there is no
trustworthy forwarding provenance, `"unknown"` is more useful than
`172.19.0.1`.

Downstream consumers that key on `client_ip` (rate limits, auth-failure
blocklist, routing affinity) must handle `"unknown"` safely — it is a
legitimate outcome, not an error. `normalize_ip_bucket("unknown")` returns
`"unknown"` unchanged, so such callers collapse onto one bucket rather than
creating a per-internal-address bucket.

## Cloudflare Pseudo IPv4

Within rung 2, `CF-Connecting-IPv6` wins over `CF-Connecting-IP` — it is not a
rung of its own — but **only when the two corroborate each other**.

Cloudflare emits `CF-Connecting-IPv6` solely when
[Pseudo IPv4](https://developers.cloudflare.com/network/pseudo-ipv4/) is set to
"Overwrite headers" — in that mode `CF-Connecting-IP` holds a synthetic Class E
(`240.0.0.0/4`) address derived from the visitor rather than the visitor's real
one. Preferring the synthetic would push an IPv6 client down the IPv4 bucketing
path, handing every rotated privacy address its own rate-limit bucket and
defeating the `/64` grouping below.

The corroboration matters because with Pseudo IPv4 off the header is *absent*
rather than cleared, so any caller can supply one. `_pseudo_ipv4_origin()`
therefore honours it only when the IPv6 header parses as IPv6 *and*
`CF-Connecting-IP` parses as IPv4 *and* that IPv4 falls inside `240.0.0.0/4`.
Cloudflare controls that second value and a real client address is never drawn
from the reserved Class E range, so the pairing cannot be forged from outside.
Otherwise `CF-Connecting-IP` stays authoritative. Both raw headers are carried
on `ClientIpInfo` and logged, so a synthetic — or a forgery attempt — stays
visible after the fact.

## Bucketing: why IPv6 folds to a /64

A single IPv6 client is typically delegated an entire prefix (a `/64` at
minimum, often a `/56` or `/48`), and RFC 4941 privacy addresses rotate within
it. A full IPv6 address is therefore a poor identity key: a client can present
effectively unlimited distinct ones.

`normalize_ip_bucket()` is the grouping key used wherever an address has to
stand in for a caller:

- IPv6 → the `/64` network it sits in (`IPV6_BUCKET_PREFIXLEN = 64`).
- IPv4 → the address itself.
- IPv4-mapped literals (`::ffff:192.0.2.1`, which a dual-stack listener reports
  for IPv4 peers) → the embedded IPv4 address. Folding these by prefix would
  collapse every IPv4 client into a single `::/64`.
- Anything unparseable (including the `"unknown"` fallback and scoped literals)
  → returned unchanged.

Logs and analytics keep the full address; only the buckets fold. Callers of
`normalize_ip_bucket()` at this revision are signup rate limiting, login rate
limiting, the repeated-auth-failure blocklist, and routing affinity.

`derive_affinity_key()` is the sticky-routing variant. It falls through caller
identities in order of how precisely each names one caller: the presented API
key's hash, then an inference-grant id (`grant:<id>`), then
`ip:<bucket>` for traffic with no credential at all. It lives beside the IP
helpers rather than on a router so that every surface dispatching to a pooled
adapter derives the caller identity the same way.

### An IPv6 client on an IPv4-only origin is normal

Seeing IPv6 addresses in the logs does not mean the origin gained IPv6. A CDN
that publishes an AAAA record accepts the client over IPv6 and then opens a
separate IPv4 connection to the origin, carrying the original address in the
forwarding header. The client's address family is decoupled from the origin's,
so for an IPv4-only origin it is an IPv6 `peer_ip` — not an IPv6 `remote_ip` —
that would be the genuine surprise.

## What is logged

`apps/backend/serving/servers/middleware/request_log.py` emits one structured
`http_request` line per request carrying the resolved address *and* its
provenance: `remote_ip`, `peer_ip`, `ip_source`, `x_forwarded_for`, `x_real_ip`,
`cf_connecting_ip`, `cf_connecting_ipv6`, alongside `user_agent`, `host`,
`origin`, `referer`, `request_id` and `session_id`.

Keeping the raw headers next to the verdict is what makes a wrong address
diagnosable: you can see which rung fired and what the alternatives said.

At `DEBUG` the middleware additionally emits an `http_request_headers` line with
every request header, truncated to 256 characters each and with `authorization`
and `x-api-key` replaced by `***`.

## What is persisted

The output of this module does not stay in the log file. It is written to the
database.

**`api_logs.metadata`** (a JSONB column; see
`apps/backend/serving/storage/log_schema.py` and the insert in
`apps/backend/serving/storage/postgres_log.py`) receives, per request:

| Surface | Handler | IP-related keys stored |
|---|---|---|
| `/v1/chat/completions` | `apps/backend/serving/servers/routers/completions.py` | `ip`, `user_agent`, `referer` |
| `/v1/messages`, `/anthropic/v1/messages` | `apps/backend/serving/servers/routers/anthropic_messages.py` | `ip`, `peer_ip`, `ip_source`, `x_forwarded_for`, `x_real_ip`, `user_agent`, `referer` |
| Rejected requests (when rejection logging is enabled) | `apps/backend/serving/observability/rejection_log.py` | `ip` |

The Anthropic surface stores the full provenance, not just the verdict — so a
disputed address can be re-derived from the row. The two `CF-Connecting-*`
headers are logged but not persisted on any surface.

**`login_events`** (`apps/backend/serving/storage/postgres_operational.py`)
stores `ip` and `user_agent` per login attempt alongside the outcome.

### Retention is yours to set

Nothing in this repository expires either table on a timer. The only deletions
that exist are operator-initiated:

- `DELETE /admin/login-events?older_than_days=N` — purge `login_events` by age.
- `DELETE /admin/login-events?user_id=...` — purge one user's login events.
- `hard_delete_user_data(user_id)` — wipes that user's `api_logs` rows.
- `delete_recent_error_requests(hours=N)` — drops recent error rows.

If your deployment is subject to a data-protection regime, or you simply do not
want to hold client addresses indefinitely, you must decide on and implement a
retention policy yourself. This project does not ship one and does not pick a
default on your behalf.

Two knobs that reduce what there is to retain in the first place:

- Leaving `trusted_proxies` empty where no proxy is in front means only the
  socket peer is ever recorded.
- Prompt and response content is governed separately by
  `DB_STORE_FULL_CONTENT` (default `false`, which hashes content rather than
  storing it verbatim; see `apps/backend/serving/config/settings.py`).

## Testing your setup

`tests/unit/utils/test_request_ip.py` covers the resolution table, the
Pseudo IPv4 corroboration, the bucketing rules, and adversarial cases (spoofed
headers, multi-hop chains, malformed inputs, all-private chains, the two
production pollution classes from issue #1036);
`tests/unit/config/test_trusted_proxies.py` covers CIDR validation;
`tests/unit/middleware/test_request_log.py` covers the log fields. Run them
with:

```bash
uv run pytest tests/unit/utils/test_request_ip.py tests/unit/config/test_trusted_proxies.py tests/unit/middleware/test_request_log.py
```

To check a live gateway, send a request with a deliberately absurd forwarded
header and look at the `ip_source` in the resulting log line — it tells you
which rung the gateway actually believed.
