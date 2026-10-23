# Staging cache-locality and routing diagnosis — 2026-10-22

Record of a measurement-driven investigation into whether normal FreeInference
staging routing reliably reuses warmed prefixes for `glm-5.3-flash`, and where
reuse actually stops. It complements
[`2026-10-22-staging-cache-diagnosis.md`](2026-10-22-staging-cache-diagnosis.md),
which proved the telemetry-side defect (a missing usage-request capability) from
client-side evidence. This record is about **physics**: whether the KV/prefix
cache is reused at all, on which endpoint, and why not.

## Authorized observability used

No admin session was available. Admin HTTP routes are unreachable through the
local bridge (`/staging/admin/*` 404s) and no `ADMIN_TOKEN` exists in the
environment; `/admin/*` requires a JWT or `ADMIN_TOKEN`
(`servers/deps.py::verify_admin_access`). Three authorized paths were used
instead:

1. **Local FI gateway request log** (`FI-gateway-release/fi-gateway-log.jsonl`)
   — per-request `request_id`, `physical_attempt_count`, `cached_input_tokens`,
   `first_event_latency_ms`, `duration_ms`. Its `served_provider` /
   `served_endpoint_id` are **product-level only** (`freeinference`, `staging`),
   never a worker, so it cannot identify an upstream endpoint.
2. **Staging inference API** through the local bridge
   (`http://127.0.0.1:18769/staging/v1`), with the staging provider key.
3. **`X-Route-Pin`**, which the gateway honours only for admin accounts
   (`servers/routers/completions.py:772`). The staging key resolves to an admin
   user, so pins are honoured — this is the authorized endpoint-pinned path, and
   the pre-flight `400 Pinned provider ... not found` response makes the header a
   provider-name oracle.

A second, unpinned fingerprint is available for free: the two staging endpoints
tokenize the same text differently, so `usage.prompt_tokens` identifies the
endpoint that served a request.

Measurement scripts are scratch (`.tmp-probe/`, not committed).

## Measurements

### Staging `glm-5.3-flash` has two provider endpoints

Pin-oracle results (`X-Route-Pin` value → accepted/rejected):

| Pin | Result |
|---|---|
| `ollama` | accepted |
| `vllm` | accepted |
| `glm-5.3-flash:ollama-api` | accepted |
| `zai`, `zai-api`, `zai_pro`, `fp_ollama`, `fp_zai`, `staging`, `freeinference`, `local`, `sglang`, … (≈40 more) | `Pinned provider ... not found` |

So `glm-5.3-flash` is a **two-provider route**: `vllm` and `ollama`. The
`owned_by: zai` in `/v1/models` is catalog metadata, not a provider label.
`fp_ollama` — the name in the investigation brief — is **not** in this route's
pin namespace; `ollama` is. The alternation it describes reproduces exactly.

### The two endpoints are distinguishable, and behave differently

Identical-prefix repeats, non-streaming, pinned:

| Endpoint | `prompt_tokens` on the same text | cache fields | cached | TTFT | reply |
|---|---|---|---|---|---|
| `vllm` | 2595 / 2659 / 8829 | present (`cached_tokens`, `cache_read_tokens`, `prompt_tokens_details`) | 0 or 4608 (see below) | 420–1300 ms | `ok` |
| `ollama` | 2536 / 2656 / 8815 | **absent entirely** | n/a | 870–2000 ms | empty |

`ollama` is the shape-B endpoint from the earlier diagnosis: a different
tokenizer (≈60 fewer tokens), ~2× slower, no usage/cache telemetry at all, and
an empty completion.

### Prefix reuse on `vllm` is size-gated; on production it is not

Pinned to `vllm`, second request of an identical pair:

| prefix tokens | cached (2nd request) | prompt − cached |
|---|---|---|
| 2595 | 0 | 2595 |
| 2659 | 0 | 2659 |
| 4514 | 0 | 4514 |
| 6274 | 2304 | 3970 |
| 8819 | 4608 | 4211 |
| 13382 | 9216 | 4166 |

`cached ≈ max(0, prompt − ~4100)`: the endpoint reuses tokens **beyond a fixed
~4.1k head** and nothing at all below ~4.2k tokens. This is stable across runs,
fresh salts and repeated prefixes, so it is not eviction: the first request of a
pair is cold, every following one is warm, and the boundary is the same.

Production, same model, same client, same 2595-token prefix, **unpinned**:

| request | cached | TTFT |
|---|---|---|
| 0 | 0 | 2209 ms |
| 1 | 2560 | 5236 ms |
| 2–5 | 2560 | 1500–2626 ms |

~98.7 % of a 2.6k-token prefix is reused on the first warm request, reliably,
with automatic routing. The same bytes under the same gateway code do not reuse
on staging. The gateway is not rewriting the prompt: `prompt_tokens` is stable,
and production reuses it.

### Normal (unpinned) staging routing only sometimes reaches the warm endpoint

Identical prefix, unpinned, one client key:

| prefix | requests | reached `vllm` (telemetry) | reached `ollama` (no telemetry) | reuse |
|---|---|---|---|---|
| 2659 tok | 12 | 11 | 1 | 0 / 11 on `vllm` (ineligible size) |
| 8815 tok | 8 | 4 | 4 | 4 / 4 on `vllm` at `cached=4608` |
| 8815 tok | 16 | 7 | 9 | 7 / 7 on `vllm` at `cached=4608` |

Two facts:

- For an **eligible** prefix, every request that reaches `vllm` reuses it
  (11 of 11 across the two runs). Reuse is deterministic once the request lands
  on the cache-capable endpoint.
- But only **44–50 %** of eligible requests reached it. The other half went to
  `ollama`, which cannot reuse anything and is ~2× slower.

The endpoint mix depends on prompt size: `ollama` served 1/12 (8 %) of 2.7k
prefixes but 9/16 (56 %) of 8.8k prefixes. Selection is therefore influenced by
the request itself, not a fixed weight.

### Streaming behaves the same, and `ollama`'s telemetry gap is real

- `vllm`, streaming with `stream_options.include_usage` → a usage chunk is
  delivered (`cached=4608`). The staging route accepts the flag.
- `ollama`, streaming with the same flag → **no usage chunk at all**, and no
  cache fields. This is the missing-telemetry case the capability fix governs:
  an endpoint that does not accept `stream_options` silently drops it, and
  downstream cannot then tell "no cache" from "no telemetry".
- One streaming `vllm` request had TTFT 30.3 s and 49.5 s total with `cached=0`,
  while a cold prefill of the same size measured 1.2–1.9 s elsewhere in the same
  session. That gap is **queueing/backpressure**, not prefill: the slow
  responses are not all cold prefills.

## Findings

- **F1 — The reported alternation is real and structural.** `glm-5.3-flash` on
  staging is a two-provider route (`vllm`, `ollama`); the brief's `fp_ollama`
  maps to the `ollama` endpoint, which is not reachable by that literal pin
  name.
- **F2 — Endpoint behaviour differs by an order of magnitude in usefulness.**
  `ollama`: different tokenizer, no usage/cache telemetry, ~2× slower, empty
  completion. `vllm`: cache-aware, fast, correct content.
- **F3 — The dominant reuse failure for ordinary prompts is endpoint-side.**
  Even pinned to `vllm` with locality guaranteed, identical prefixes below
  ~4.2k tokens never reuse (`cached=0`, 12/12 and 8/8 and 10/10 across runs).
  Above that the endpoint reuses only `prompt − ~4.1k` tokens. Production reuses
  98.7 % of the same 2.6k-token prompt. This is a staging worker/provider
  behaviour, not a gateway behaviour.
- **F4 — For eligible prefixes, reuse is deterministic once the request lands
  on `vllm`** (11/11). The residual loss is selection, not the cache.
- **F5 — Normal staging routing reuses eligible warmed prefixes only ~44–50 %
  of the time**, because it steers the other half to the cache-blind endpoint.
  The steering is prompt-size dependent, so it is a routing-policy/priority
  effect, not a fixed weight.
- **F6 — The slow responses are heterogeneous.** They include genuine cold
  prefills (first request of a pair), queueing (30 s TTFT at an otherwise 1–2 s
  prefill size), endpoint differences (`ollama` ~2× `vllm`), and — for the
  `ollama` endpoint only — plain missing telemetry. These are distinguishable by
  `cached` + TTFT + the `prompt_tokens` fingerprint.
- **F7 — The gateway already contains the locality machinery.** `FixedRouter`
  keeps a per-`(affinity_key, model_id)` endpoint pin
  (`routers.py::_Affinity`, default on via `ROUTING_AFFINITY_ENABLED`, 300 s
  sliding TTL) and discounts warm continuations in prefill accounting
  (`PrefillLoadTracker.uncached_estimate`). The pin is honoured unless the
  endpoint is absent from `allowed` (health/circuit filter, or an offload-route
  reduction), or its foreign in-flight backlog exceeds
  `ROUTING_PREFILL_AFFINITY_CEILING` (150k tokens, `should_keep_affinity`), in
  which case it is re-drawn and re-pinned. A pin is also dropped on failure
  (`_drop_affinity`). Production's behaviour (5/5 warm hits under automatic
  routing) is what this machinery looks like when it holds.

## What is external, and what is missing

- **External, precisely identified:** staging `vllm` does not reuse prefixes
  below ~4.2k tokens and caps reuse at `prompt − ~4.1k` above that; staging
  `ollama` reuses nothing and reports nothing. Production's endpoint reuses
  98.7 % of the same 2.6k prefix. No gateway code path can make a worker reuse a
  prefix it declines to reuse — pinning the request to that worker is the
  strongest possible locality guarantee the gateway can give, and the prefix
  still is not reused.
- **Missing evidence (admin-only):** staging's routing configuration decides
  whether the ~50 % steering in F5 is (a) `ROUTING_AFFINITY_ENABLED=0`, (b) the
  route's weights/`router` for `glm-5.3-flash`, or (c) pins dropped by the
  health filter / backlog ceiling / failover. The per-request routing decision,
  the route's adapter list and weights, and `api_logs` rows for the probes all
  live behind `/admin/*`, which needs a JWT or `ADMIN_TOKEN` this session does
  not have. `X-Route-Pin` proved the account is admin; it does not expose the
  decision record.

## Consequence for the objective

"Normal staging routing reliably reuses eligible warmed prefixes" is **not**
satisfied on the evidence: the measured eligible-prefix reuse rate under
automatic routing is 44–50 %. The shortfall has two independent components, and
they must not be conflated:

1. **Endpoint eligibility (dominant for ordinary prompts, external).** Below
   ~4.2k tokens staging's endpoints reuse nothing; production's reuses almost
   all of it.
2. **Endpoint selection (dominant for eligible prompts, gateway-config).** Half
   of eligible requests are steered to a cache-blind endpoint even though the
   gateway has, and production appears to use, affinity that would prevent it.

Component 1 cannot be fixed in this repository. Component 2 plausibly can —
but only after the admin-only configuration is read, because the observable
behaviour is equally consistent with "affinity disabled by configuration" and
with "affinity dropped by the health/backlog/failover paths", and those call
for opposite changes. Changing selection semantics now would be a
hypothesis-driven edit, which the objective explicitly forbids.

## Reproduce

```bash
# staging, automatic routing, 2.7k-token identical prefix
python3 .tmp-probe/probe.py --mode chat --count 12 --salt U1
# staging, pinned to the cache-capable endpoint
python3 .tmp-probe/probe.py --mode chat --count 12 --pin vllm --salt U1
# staging, eligible size (compare cached on vllm-pinned vs the tail after ~4.1k)
python3 .tmp-probe/probe.py --mode chat --count 4 --pin vllm --prefix-repeats 200 --salt B2
# production reference (unpinned)
python3 .tmp-probe/probe.py --provider production --mode chat --count 6
# the pin oracle
python3 .tmp-probe/enumerate_pins.py glm-5.3-flash staging
```

## Recommendation

1. Read the staging `ROUTING_AFFINITY_ENABLED` value and the `glm-5.3-flash`
   route weights/`router` from Admin → Configuration, and the `api_logs` rows
   for the probe requests. That is the one missing fact that decides which of
   the two selection causes in F5 is real.
2. Treat the staging `vllm` ~4.1k-token no-reuse head as a provider defect to
   raise with that endpoint's operator; it makes every ordinary prompt
   cold-start a prefill that production reuses.
3. Consider whether a cache-blind, `include_usage`-incompatible endpoint should
   serve `glm-5.3-flash` traffic at all, or at a low weight behind the
   cache-capable one. This is a routing-configuration decision, not a code fix.
