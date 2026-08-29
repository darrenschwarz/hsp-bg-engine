# bg-engine wire contract, version 1

The HSP game server talks to this sidecar over HTTP JSON. This page is the
contract the HSP client (`src/backgammon/bot/core/wildbgEngine.ts`) validates
on every reply before it reads a single result (GP-476). Canonical examples
live in `crates/bg-engine/fixtures/contract-v1/`, held equal to the real
response types by `cargo test -p bg-engine`.

**Scope.** This API serves live single-position traffic: one table's next
checker play or cube decision, a handful of positions per request at most.
It is not a bulk analysis interface. Post-match analysis of a whole game is
a different workload with different capacity, and needs its own deployment
rather than a larger batch here (GP-477).

## Access and limits (GP-477)

| Rule | Detail |
|---|---|
| Authentication | `POST /rank`, `/evaluate` and `/cube` require `Authorization: Bearer <token>`, where the token is the sidecar's `BG_ENGINE_AUTH_TOKEN`. The sidecar refuses to start without one. Token grammar, enforced identically by the sidecar at startup and by the HSP server at its composition boundary: at least 16 characters, every one visible ASCII (`!`..`~`, 0x21..0x7E) -- no spaces, no control characters, nothing outside ASCII (`openssl rand -hex 32` produces one). A missing, malformed or wrong credential is answered `401 {"error":"unauthorized"}` with `WWW-Authenticate: Bearer realm="bg-engine"` before the body is read -- the reply never says which of the three it was. The comparison is constant time. `GET /health` is public. |
| Body limit | 16 KiB. A larger body is answered `413` (after the credential check) and never parsed. |
| Batch | 1 to 8 positions per request, on every evaluation route. An empty batch or a batch of 9 or more is `400 {"error":...}` before any evaluation. |
| Plies | `plies` 0 or 1 run the net's own evaluation (`plies.1`); 2 runs the 2-ply evaluator (`plies.2`); anything else is `400` before any evaluation. |
| Concurrency | Neural evaluation runs off the request threads on a blocking pool bounded by `BG_ENGINE_MAX_CONCURRENT_EVALS` (default 1, at most 4). A request waits at most 100 ms for a slot; if none frees up it is answered `429 {"error":...}` with `Retry-After: 1` and no evaluation is started. A client that has gone away does not free its slot early: the slot returns when the evaluation it started finishes. |
| Not ready | While the startup readiness check has failed, the evaluation routes answer `503` with the `/health` body (GP-476). |

Order of refusals for one request: 401 (credentials) -> 413 (body) -> 400/415/422
(body not the request type) -> 400 (batch, plies) -> 503 (not ready) -> 429
(capacity). Every refusal before 429 is answered without waiting for capacity.

`/health` is unaffected by evaluation load: it never queues behind inference.

## Metadata on every reply

Every `/health` reply and every `/rank`, `/evaluate` and `/cube` batch reply
carries the same three fields, built once at startup and never changed:

| Field | Value |
|---|---|
| `apiVersion` | `1`. Clients require an exact match. A newer number is not compatible. |
| `capabilities` | `["rank.v1", "evaluate.v1", "cube.money.v1", "plies.1", "plies.2"]` — only what this build implements. A client must find the capability for the operation it made (`rank.v1`, `cube.money.v1`) and, for `/rank`, the `plies.<n>` it asked for. |
| `engineId` | `wildbg@<source revision>+contact@<sha256>+race@<sha256>`: the full commit hash the binary was built from and the SHA-256 of the exact `neural-nets/contact.onnx` and `race.onnx` bytes compiled into it. Baked in at build time; the build fails without a full commit hash, and there is no runtime override. |

## Health

`GET /health` evaluates the opening position with the loaded 1-ply evaluator
and checks that the side on roll's win probability is finite and inside
`0.45..=0.55` (the true figure is a shade over 0.5). Ready:

    HTTP 200  {"ok":true, ...metadata, "openingWinProbability":0.51}

Not ready — wrong nets, a failed load, a mis-oriented board:

    HTTP 503  {"ok":false, ...metadata, "openingWinProbability":0.875,
               "error":"opening win probability 0.875 outside plausible range 0.45..=0.55"}

The container's `HEALTHCHECK` runs `bg-engine --health`, an HTTP client that
performs this same request and exits 0 only on HTTP 200 with `"ok":true`.
Render's health-check path should be `/health` for the same reason.

## Batch replies

    { "results": [ ... one per request item, in order ... ],
      "apiVersion": 1, "capabilities": [...], "engineId": "wildbg@...",
      "evalMs": <f64>, "positionsEvaluated": <n> }

`results.length` always equals the number of request items; a client that
receives a different count must discard the reply.

## What a client must do

Reject the reply — and answer from an identified fallback rather than from
this engine — when any of these holds: `apiVersion` missing or not exactly
`1`; `engineId` missing or not of the structure above; the operation's
capability (and requested ply) absent; result count different from the
request. A client must never fill a missing `engineId` with a label of its
own.

Send the bearer token on every evaluation request and never on `/health`.
Treat `401` and `403` as a configuration failure — the token on the client
does not match the sidecar's — not as an outage: do not retry, do not log
the token, answer from the fallback. Treat `429` as a capacity signal: do
not retry the same decision, answer it from the fallback and say so in its
provenance; the sidecar is up. The HSP client
(`src/backgammon/bot/core/wildbgEngine.ts` with `resilientEngine.ts`) does
exactly this.
