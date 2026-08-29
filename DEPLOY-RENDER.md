# Deploying the bg-engine sidecar on Render

This repo builds one small HTTP service: the neural-net backgammon engine
the HSP server consults when `BG_ENGINE_URL` is set. Without it the server
falls back to its built-in stub engine — bots still play legally, but the
six difficulty tiers are not calibrated.

## One-time setup

1. Push this folder to its own GitHub repo (e.g. `hsp-bg-engine`):

       git init && git add -A && git commit -m "bg-engine sidecar"
       git remote add origin <your-repo-url> && git push -u origin main

2. Render → New → **Web Service** → pick that repo
   - Runtime: **Docker**
   - Dockerfile path: `Dockerfile.bg-engine`
   - Instance: Starter. (A free instance spins down when idle; the HSP
     server's 1.5s engine timeout then falls back to the stub until the
     service wakes, so bots quietly lose their calibration.)

3. Generate a bearer token and give it to BOTH services (GP-477). The
   sidecar refuses to start without one, and the HSP server refuses to
   start with `BG_ENGINE_URL` set but no token:

       openssl rand -hex 32

   On the **engine service** → Environment:

       BG_ENGINE_AUTH_TOKEN         = <that value>
       BG_ENGINE_MAX_CONCURRENT_EVALS = 1        # optional; 1 (default) .. 4

   On your existing **HSP API service** → Environment:

       BG_ENGINE_URL        = https://<your-engine-service>.onrender.com
       BG_ENGINE_AUTH_TOKEN = <the same value>

   (If you use a paid Private Service instead, use the internal address
   Render shows in its dashboard — same region as the API service. The
   token is still required: the sidecar has no unauthenticated mode.)

4. Redeploy the API service.

## Capacity (GP-477)

This service is sized for live play -- one table's next decision at a
time -- not for analysing whole matches. The evaluation routes accept at
most 8 positions and 16 KiB per request, run inference on a bounded
blocking pool, and answer `429 Retry-After: 1` rather than queue when every
slot has been busy for 100 ms; the HSP server then answers that one
decision from its own heuristic and says so in the decision record. Leave
`BG_ENGINE_MAX_CONCURRENT_EVALS` at 1 on a Starter instance (one core);
raise it, up to 4, only on an instance with that many cores. `/health`
never waits behind inference, so Render's health check stays truthful
under load. To see how the deployed service behaves at live-sized load,
run `crates/bg-engine/tools/live-load.py` against it (see the script's
header); it never prints the token.

## Identity and health (GP-476)

The binary's identity is baked in when the image is built:
`wildbg@<commit hash>+contact@<sha256>+race@<sha256>`. The Docker build reads
the commit hash from `RENDER_GIT_COMMIT` (which Render passes to Docker
builds) or from `--build-arg BG_ENGINE_SOURCE_REVISION=...`, and **fails**
without one -- confirm the first Render build after this change succeeds; if
it does not, add `BG_ENGINE_SOURCE_REVISION` to the service's environment.
There is no `BG_ENGINE_ID` variable any more and no fallback label.

Set the service's **health check path to `/health`**. It answers 200 only
when the loaded nets pass the opening-position sanity check, 503 otherwise,
so a broken build never receives traffic.

## Verify

    curl -i https://<your-engine-service>.onrender.com/health

should answer HTTP 200 with `"ok":true`, `"apiVersion":1`, the capabilities
and the full identity. From then on bot decisions carry that identity in
their provenance instead of `stub`; the HSP server's own `/health` shows it
as `backgammonBot.engine.lastEngineId`.

An evaluation request without the token is refused:

    curl -i -X POST https://<your-engine-service>.onrender.com/cube \
      -H 'content-type: application/json' -d '[]'
    # HTTP 401 {"error":"unauthorized"}

and with it (the empty batch is then refused for being empty, which proves
the credential was accepted):

    curl -i -X POST https://<your-engine-service>.onrender.com/cube \
      -H "authorization: Bearer $BG_ENGINE_AUTH_TOKEN" \
      -H 'content-type: application/json' -d '[]'
    # HTTP 400 {"error":"the batch is empty: send between 1 and 8 positions"}

If the HSP server's `/health` shows `backgammonBot.engine.lastFallback.reason`
containing `unauthorized`, the two services hold different tokens.

The binary binds whatever `PORT` Render injects, so no port config needed.
See `docs/api/contract-v1.md` for the full wire contract.
