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

### Staging `glm-5.3-flash` exposes two *pinnable* endpoints; the route has three candidates

Pin-oracle results (`X-Route-Pin` value → accepted/rejected):

| Pin | Result |
|---|---|
| `ollama` | accepted |
| `vllm` | accepted |
| `glm-5.3-flash:ollama-api` | accepted |
| `zai`, `zai-api`, `zai_pro`, `fp_ollama`, `fp_zai`, `staging`, `freeinference`, `local`, `sglang`, … (≈40 more) | `Pinned provider ... not found` |

The pin namespace is therefore `vllm` and `ollama`. `fp_ollama` — the name in the
investigation brief — is **not** in this route's pin namespace; `ollama` is. The
alternation it describes reproduces exactly.

The admin routing panel (read later, see "The routing policy, read from the
console") shows **three** candidates: `zai`, `vllm` and `ollama`. So the
two-endpoint view above is a *pin-namespace* artifact, not the route's full
membership: a candidate the pin oracle does not expose can still take ordinary
selection. That distinction matters for attributing the ~50 % split in F5 and is
the reason the weight values remain the deciding missing input.

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

The *sequence* of choices (lowercase `o` = `ollama`, `v` = `vllm` miss,
`V` = `vllm` hit) shows runs, not independent draws:

| prefix | sequence |
|---|---|
| 2.7k, unpinned | `vvovvvvvvvvv` |
| 8.8k, unpinned | `ooVVVVooVoooooVV` |
| 8.8k, pinned to `vllm` | `VVVVVVVVVVVV` |

When the eligible run leaves `vllm` it stays away for up to five consecutive
requests and then returns. That is the shape of a pin being lost and re-drawn —
and, on re-draw, the cache-blind endpoint winning and holding the caller for its
TTL — rather than a per-request coin flip. Each excursion costs a full cold
prefill on a prefix that was already warm in the sibling endpoint.

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
  staging is a three-candidate route, read directly from
  `/dashboard/admin/routing` (see "The routing policy, read from the console"):
  ordinary candidates `zai` (`https://api.z.ai/api/coding/paas/v4/`) and `vllm`
  (`http://host.docker.internal:8004/v1`), plus `ollama` (`https://ollama.com/v1`)
  designated as the model's **offload route**. The brief's `fp_ollama` maps to the
  `ollama` endpoint, which is not reachable by that literal pin name.
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
  of the time.** Two effects are now separable and were previously conflated:
  the **offload route** (F8) removes requests from the cache-capable endpoint by
  wait, and the residual split among the ordinary candidates is a selection
  effect. The offload share is prompt-size dependent (8 % at 2.7k tokens, 56 % at
  8.8k), which is exactly what a first-token/queue deadline predicts: a longer
  prompt takes longer to produce a first token, so it crosses the threshold more
  often. Whether the residual ordinary split is the configured weights or
  affinity failing to hold could not be settled — the console renders the weight
  values as number inputs whose values are not exposed to a read-only DOM
  snapshot, and all three rows read "Config default". Any claim that it is "not a
  fixed weight" is therefore unproven; it is left open rather than restated.
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
- **Read from the console, no longer missing:** the model's route table and
  offload policy (see "The routing policy, read from the console"). Still not
  captured: the **numeric weight values** for the three candidates. The panel
  renders them as number inputs; a read-only DOM snapshot exposes neither `value`
  nor `title` for them, and no screenshot is serialized by the bridge. Everything
  else that decides this behaviour — the offload route's identity, its trigger
  and its parameters, and each candidate's status — was read directly.

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

Component 1 cannot be fixed in this repository. Component 2 splits in two once
the console has been read: the **offload** half is gateway-side and was
corrected here (see "Correction implemented"), because the deadline was
cancelling cold prefills that would have completed. The **ordinary-candidate**
half had a structural gap that did not depend on the missing weight values, and
that gap is now closed too (see "Selection now accounts for locality"):
`PrefillLoadTracker` already knew the per-endpoint un-cached estimate and used it
only to discount load and gate elephants, while `select_index` broke ties by
backlog — so `FixedRouter` selection had no locality term at all, although
`RouteWise` has one (`routing.routewise.prefix_cache`). What still is not settled
is how much of the measured residual split the configured weights explain; that
needs the numeric weights, and is a tuning question rather than a mechanism gap.

## Reproduce

```bash
# The one-off readers here were scratch and are renamed ``*.py.scratch`` so the
# repository lint gate does not format them; the durable evidence is this file.
# staging, automatic routing, 2.7k-token identical prefix
python3 .tmp-probe/probe.py.scratch --mode chat --count 12 --salt U1
# staging, pinned to the cache-capable endpoint
python3 .tmp-probe/probe.py.scratch --mode chat --count 12 --pin vllm --salt U1
# staging, eligible size (compare cached on vllm-pinned vs the tail after ~4.1k)
python3 .tmp-probe/probe.py.scratch --mode chat --count 4 --pin vllm --prefix-repeats 200 --salt B2
# production reference (unpinned)
python3 .tmp-probe/probe.py.scratch --provider production --mode chat --count 6
# the pin oracle
python3 .tmp-probe/enumerate_pins.py.scratch glm-5.3-flash staging
```

## Recommendation

1. Capture the three numeric route weights (and `ROUTING_AFFINITY_ENABLED`) for
   staging `glm-5.3-flash`, plus `api_logs` for the probe requests. That is the
   remaining input for the *ordinary-candidate* half of F5; the offload half is
   no longer open (see "Correction implemented").
2. Treat the staging `vllm` ~4.1k-token no-reuse head as a provider defect to
   raise with that endpoint's operator; it makes every ordinary prompt
   cold-start a prefill that production reuses.
3. Consider whether a cache-blind, `include_usage`-incompatible endpoint should
   serve `glm-5.3-flash` traffic at all, or at a low weight behind the
   cache-capable one. This is a routing-configuration decision, not a code fix.
   The `ollama` candidate is *already* held back from ordinary traffic by its
   offload-route designation, so this now concerns `zai` only.

## The routing policy, read from the console

Read-only, from `/dashboard/admin/routing` with `glm-5.3-flash` selected. API-key
cells are redacted here; the page shows only "Configured default `<x>` key" text
in them.

| Candidate | `on_demand` | Target | Weight control | Status |
|---|---|---|---|---|
| `zai` | yes | `https://api.z.ai/api/coding/paas/v4/` | `YAML · Effective` + Save + **Reset** | Config default |
| `vllm` | yes | `http://host.docker.internal:8004/v1` | `YAML · Effective` + Save | Config default |
| `ollama` | yes | **`https://ollama.com/v1`** | `YAML · Effective` + Save; *"Offload route: gets no ordinary traffic"* | **Offload** |

The `ollama` row's `Status` carries, verbatim:

> title: "Takes requests that waited **0.8s** for a slot or a first token, and
> serves as the **last resort**"

Three consequences:

- The `ollama` candidate is **`ollama.com`, a remote cloud service**, not a local
  server. That explains its tokenizer offset, its absent usage/cache telemetry,
  its ~2× latency and its occasional empty content (F2).
- It is the model's **offload route**, so it receives *no ordinary traffic*: the
  `ollama` observations under automatic routing were **offload**, not a weighted
  draw. The 8 % → 56 % prompt-size dependence is the deadline's signature, not a
  weight's.
- `zai` is the only row offering `Reset`, i.e. the only candidate with a runtime
  weight override beyond the YAML default. The numeric weights are still not
  captured (see above); no console control was operated.

## The offload lifecycle, traced in source

`routing.offload` arms **three** distinct triggers, all with the same
`wait_seconds`:

1. **Gateway queue wait.** `routers.FixedRouter._queue_deadline` returns
   `time.monotonic() + policy.wait_seconds`, pushed as
   `req_ctx.UPSTREAM_QUEUE_DEADLINE` around an attempt. It is enforced in
   `upstream_limiter._queue_wait` (a remote provider's slot queue) and
   `_hold_wait` (a local engine's `EngineHold`). Expiry raises
   `UpstreamQueueWaitExpired` → `OFFLOAD_QUEUE_WAIT`. The request was never sent.
2. **Engine first-token wait.** `FirstTokenWatch` (`routing.engine_wait`),
   enforced by `asyncio.timeout` inside `FixedRouter._await_first_token`. Expiry
   raises `EngineWaitExpired` → `OFFLOAD_ENGINE_WAIT`.
3. **Last resort.** `FallbackOrder` hands the offload route over when nothing
   else is admissible, or after everything else has failed.

Precise answers to the questions the objective asks:

- **When the timer begins.** The *queue* clock is absolute, armed when the router
  prepares the attempt — before dispatch, and one value is reused for that
  request's later attempts. The *engine* clock is different: `FirstTokenWatch.arm`
  starts it, but `on_sent` (`dispatch_watch.report_sent`, called by
  `upstream_limiter.acquire` when the request takes a slot or is sent to a local
  engine) *restarts* it, so the engine wait begins when the request leaves the
  gateway for the engine.
- **What pauses it.** `on_queued` (`dispatch_watch.report_queued`) pauses the
  engine clock while the request waits for an outbound slot or inside a local
  `EngineHold`. Gateway queue time therefore never counts against the engine —
  proven by `test_time_in_the_gateway_queue_does_not_count_against_the_engine`.
- **What the first token is.** The first frame the adapter reads from the
  upstream that carries generated output — text, reasoning, or any part of a tool
  call — reported *before* a stream processor can hold it back
  (`dispatch_watch.report_first_token`). Output an adapter buffers whole (a MiniMax
  `think` block, a tool call's JSON) therefore does *not* count against the
  deadline (`test_output_a_processor_holds_back_is_not_an_engine_wait`).
- **Does the gateway cancel an attempt that was about to finish prefill? Yes.**
  There is no prefill-progress signal at all: the only evidence of progress is
  the first token, and it arrives *after* prefill. On expiry
  `_await_first_token` unwinds the attempt's stream, which closes the connection
  and aborts the request inside the engine. The prefill already performed is
  discarded, and the prompt is prefilled **from scratch, cold, on the offload
  route** — duplicated work and a guaranteed cache miss on a service that shares
  no KV with the engine that was answering. Closing the stream is what aborts the
  request at the engine (`_Engine.cancelled` in
  `tests/unit/routing/test_engine_wait_routing.py`).
- **The engine is not blamed.** `EngineWaitExpired` is exempt from the circuit
  breaker (observed in a test run as `engine_wait_skip_breaker`, and asserted in
  `test_a_stream_with_no_first_token_is_cancelled_and_offloaded`). The policy's
  intent is right; only its trigger is blind.

## Correction implemented — a prefill-aware, bounded engine wait

The defect is that one constant bounds a quantity — prefill + time-to-first-token
— whose expected value grows with the prompt's *un-cached* length, so the
deadline fires hardest exactly when the engine is doing legitimate work. The
correction keeps the policy's shape and bounds and makes the wait a function of
the work it bounds:

- `routing/offload.py`: `OffloadPolicy.engine_wait_seconds(uncached_tokens)`,
  with `OFFLOAD_PREFILL_CREDIT_TOKENS = 4096` (tokens credited inside the base
  wait) and `OFFLOAD_MAX_WAIT_SECONDS = 8.0` (a fixed ceiling, so a wedged engine
  still fails over). The result is never below `wait_seconds`, never above the
  ceiling, and monotone in the un-cached estimate.
- `routing/routers.py`: `_watch_for` takes the endpoint's estimated un-cached
  prefill and builds the watch from `engine_wait_seconds`; **both** call sites
  pass `PrefillLoadTracker.uncached_estimate(endpoint_id, prefill_tokens,
  affinity_key)` — the primary and the fallback stream.

Unchanged, deliberately: the gateway queue-wait trigger and its absolute
deadline; `engine_queue_limit` and `EngineHold`; the last-resort path; the
`FallbackOrder` retry of a route cut short; circuit-breaker and cancellation
semantics; `max_input_tokens`. Nothing here disabled offloading or pinned an
endpoint permanently.

Effect at staging's configured `wait_seconds = 0.8 s`:

| un-cached prefill | engine wait |
|---|---|
| ≤ 4096 (a warm continuation, or a short prompt) | 0.8 s — unchanged |
| 8192 | 1.6 s |
| 20480 | 4.0 s |
| ≥ 40960 | 8.0 s (ceiling) |

Regression coverage:

- `tests/unit/routing/test_offload_routing.py` — the value: base floor for a warm
  or short prompt, proportionality above the credit, the ceiling, and that a
  configured wait above the ceiling is never shortened.
- `tests/unit/routing/test_engine_wait_routing.py` —
  `test_a_cold_prefill_longer_than_the_base_wait_is_not_cancelled_mid_prefill`
  (served locally, `cancelled == 0`),
  `test_a_short_prompt_still_offloads_at_the_base_wait` (no regression for the
  case the feature exists for), and
  `test_a_fallback_stream_is_watched_under_its_own_budget` (the second call site).

Mutation check: with `engine_wait_seconds` reverted to a flat base, the two
cold-prefill tests fail and the request is offloaded — the captured log shows
`engine_wait_skip_breaker ... within 0.1s` followed by
`route_offload reason="engine_wait"`, i.e. precisely the staging symptom. The
short-prompt test still passes, confirming the fix is targeted.

Status: local commit only. Nothing was pushed, deployed, or configured on
staging, and no live routing policy was changed.

## Selection now accounts for locality

The requirement is that route selection account for expected un-cached prefill
cost, actual endpoint capacity and cache locality, and that it distinguish a
healthy cache-warm endpoint from a short recoverable queue, an overloaded one,
and one with no known prefix. Two of those three inputs already existed and were
not used for selection:

- `PrefillLoadTracker.uncached_estimate(endpoint, tokens, affinity_key)` — how
  much of *this caller's* prompt that endpoint must still prefill. Used to
  discount backlog and to gate elephants; never to choose.
- `self._backlog[endpoint]` — the live, capacity-relevant load. Used only as the
  power-of-two tie-break, and only above `INTERVENE_TOKENS`.

So the draw could give a conversation with a resident prefix to a cold replica
while the tracker knew the warm one would cost thousands fewer tokens. That is
the Fixed-versus-RouteWise asymmetry: `routing.routewise.prefix_cache` tracks
reuse evidence per endpoint, `FixedRouter` had no locality term.

`PrefillLoadTracker.select_index` now ends with `_prefer_cached`, which reuses the
un-cached estimates it already computes:

- prefer the candidate with the materially smaller un-cached estimate;
- **only** when it saves at least `LOCALITY_MIN_SAVING_TOKENS` (4096) — below
  that it is jitter or a partly-shared prefix, and the configured weight (cost,
  contractual preference) keeps deciding;
- **only** while that candidate's foreign backlog is within
  `AFFINITY_BACKLOG_CEILING` — the same judgement `should_keep_affinity` already
  makes for a sticky pin, so a *short ok queue* is preferred but an overloaded
  endpoint is not;
- never resurrecting a candidate dropped by `avoid` or by the elephant
  saturation filter.

The mapping to the four states the requirement names: a warm endpoint with a
short queue wins on the un-cached saving; the same endpoint with a foreign
backlog past the ceiling loses to capacity; an overloaded or unavailable endpoint
is already excluded by health/elephant filters; an endpoint with no known prefix
has no discount, so it competes only on weights and backlog as before.

Not a hard pin and not a cache-hit maximizer: weights still decide below the
saving threshold, and the backlog ceiling keeps a warm-but-queued endpoint from
absorbing traffic. `ROUTING_PREFILL_AWARE_ENABLED=0` still disables the whole
prefill-aware path.

Regression coverage in `tests/unit/routing/test_prefill_load.py`: a material
saving redirecting the draw; an immaterial saving leaving it alone; a backlogged
warm endpoint losing; no caller identity leaving the draw alone; `avoid` not
being overridden. Mutation check: disabling `_prefer_cached` fails only the
material-saving test, so the other four are genuine no-change assertions rather
than incidental passes.

Status: local commit only, as above.

## Still open — the ~4.1k-token head on staging `vllm`

Unchanged by the above and still not root-caused. The observed relation is
`cached ≈ prompt − ~4100`, with a hard `0` below ~4.1k, while production reuses
`2560/2595` of the *same* 2.6k prompt. That rules out a shared gateway cause:
identical gateway code produces both. What remains is the staging engine's own
configuration, or a difference in what staging sends.

Discriminators that would settle it, in order of cheapness:

1. **Head-only perturbation.** Hold the size above 4.1k and change only the first
   ~100 tokens. `cached` staying at `prompt − 4100` means the head is excluded
   from matching (engine-side); `cached` collapsing to 0 means the head is what
   diverges (request-side).
2. **Direct to the engine, bypassing the gateway.** The same small prefix sent
   straight to `local-8004`. Caching there but not through the gateway localizes
   the head to the gateway; not caching there either makes it engine-side.
3. **Engine facts**, which this session cannot read (the engine is reachable from
   the staging host, not from here): the vLLM version, its startup flags
   (`--enable-prefix-caching`, `--block-size`, `--max-num-batched-tokens`,
   `--num-gpu-blocks-override`), and its prefix-cache counters
   (`gpu_prefix_cache_hit_rate` on `/metrics`) next to production's. These are
   non-secret and are the exact telemetry needed.

No server or configuration correction is claimed here: without those three
readings, a physical reuse threshold and a cache-*accounting* threshold are still
indistinguishable, and the objective forbids reporting a configuration finding as
a fix.
