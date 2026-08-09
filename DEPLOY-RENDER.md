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

3. On your existing **HSP API service** → Environment → add:

       BG_ENGINE_URL = https://<your-engine-service>.onrender.com

   (If you use a paid Private Service instead, use the internal address
   Render shows in its dashboard — same region as the API service.)

4. Redeploy the API service.

## Verify

    curl https://<your-engine-service>.onrender.com/health

should return the engine identity. From then on bot decisions carry
`engineId: wildbg` instead of `stub` in the decision log.

The binary binds whatever `PORT` Render injects, so no port config needed.
