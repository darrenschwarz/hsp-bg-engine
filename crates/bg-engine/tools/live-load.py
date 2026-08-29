#!/usr/bin/env python3
"""Live-sized load against a running bg-engine (GP-477).

Sends single-position requests -- batch size exactly 1, the shape of live
play -- at a chosen concurrency, and reports what came back: status counts,
latency p50/p95/p99/max, how the sidecar's own /health answered while the
load ran, and the Retry-After it asked for when it refused. Repeatable: the
positions and dice are chosen from a fixed seed, so two runs against the
same build send the same requests.

Usage (the token is read from the environment and is NEVER printed):

    export BG_ENGINE_AUTH_TOKEN=...            # PowerShell: $env:BG_ENGINE_AUTH_TOKEN = "..."
    python crates/bg-engine/tools/live-load.py --url http://127.0.0.1:8090 \
        --requests 200 --concurrency 8 --plies 1

    python crates/bg-engine/tools/live-load.py --route cube --requests 100
    python crates/bg-engine/tools/live-load.py --unauthenticated   # expect 401s
    python crates/bg-engine/tools/live-load.py --json > run.json

What to expect from a healthy sidecar with BG_ENGINE_MAX_CONCURRENT_EVALS=1:
a mix of 200 and 429 once concurrency exceeds what one slot serves within
100 ms, p99 of the 200s a few ms above p50 at 1-ply, and /health answering in
single-digit milliseconds throughout. Anything else is worth a look before
it reaches a table.

The exit status is a verdict, not a courtesy. Authenticated runs PASS (0)
only when every request was answered 200 or 429 -- no transport errors, no
timeouts, no other status -- with at least one 200, every 429 carrying a
valid Retry-After, and every /health sample answered 200 (and, with
--health-latency-ms, /health p99 under that ceiling). --unauthenticated runs
PASS only when every request was 401 and /health stayed 200. Every failed
condition is listed.

Standard library only; Python 3.8+.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

TOKEN_ENV = "BG_ENGINE_AUTH_TOKEN"

# The standard opening position in wildbg's 26-slot format (see main.rs).
OPENING = [0] * 26
OPENING[24] = 2
OPENING[13] = 5
OPENING[8] = 3
OPENING[6] = 5
OPENING[1] = -2
OPENING[12] = -5
OPENING[17] = -3
OPENING[19] = -5

DICE = [(a, b) for a in range(1, 7) for b in range(1, a + 1)]  # the 21 distinct rolls


class Redactor:
    """Every string that reaches stdout/stderr passes through here."""

    def __init__(self, token: Optional[str]):
        self.token = token

    def __call__(self, text: str) -> str:
        if self.token and self.token in text:
            text = text.replace(self.token, "[redacted]")
        return text


class Sample:
    __slots__ = ("status", "ms", "eval_ms", "retry_after", "error")

    def __init__(self, status: int, ms: float, eval_ms: Optional[float], retry_after: Optional[str], error: Optional[str]):
        self.status = status
        self.ms = ms
        self.eval_ms = eval_ms
        self.retry_after = retry_after
        self.error = error


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def one_request(url: str, path: str, body: Optional[bytes], token: Optional[str], timeout: float) -> Sample:
    headers = {"content-type": "application/json", "accept": "application/json"}
    if token is not None and path != "/health":
        headers["authorization"] = "Bearer " + token
    request = urllib.request.Request(url + path, data=body, headers=headers, method="POST" if body is not None else "GET")
    started = time.perf_counter()
    status = 0
    eval_ms = None
    retry_after = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = response.read()
            try:
                eval_ms = json.loads(payload).get("evalMs")
            except (ValueError, AttributeError):
                pass
    except urllib.error.HTTPError as e:
        status = e.code
        retry_after = e.headers.get("Retry-After")
        try:
            error = json.loads(e.read()).get("error")
        except (ValueError, AttributeError):
            error = None
    except Exception as e:  # connection refused, timeout, ...
        status = 0
        error = type(e).__name__ + ": " + str(e)
    return Sample(status, (time.perf_counter() - started) * 1000.0, eval_ms, retry_after, error)


def valid_retry_after(value: Optional[str]) -> bool:
    """RFC 9110: a non-negative integer of seconds, or an HTTP-date."""
    if value is None:
        return False
    text = value.strip()
    if text.isdigit():
        return True
    try:
        parsedate_to_datetime(text)
        return True
    except (TypeError, ValueError, IndexError):
        return False


def verdict(args: argparse.Namespace, samples: List[Sample], health: List[Sample], summary: Dict[str, object]) -> List[str]:
    """Every way the run fell short of the contract; empty means PASS.

    Authenticated: every request answered 200 or 429, nothing else -- no
    transport errors, no timeouts, no unexpected statuses -- with at least
    one 200 so the evaluator is known to have worked, and every 429 carrying a
    valid Retry-After. Unauthenticated: every request answered 401. In both
    modes every /health sample answered 200, and, when --health-latency-ms is
    given, /health p99 stayed under it.
    """
    failures: List[str] = []
    if args.unauthenticated:
        wrong = [s for s in samples if s.status != 401]
        if wrong:
            failures.append(f"{len(wrong)} of {len(samples)} unauthenticated requests were not 401")
    else:
        unexpected = [s for s in samples if s.status not in (200, 429)]
        if unexpected:
            counts: Dict[str, int] = {}
            for s in unexpected:
                key = str(s.status) if s.status else "no-response"
                counts[key] = counts.get(key, 0) + 1
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            failures.append(f"{len(unexpected)} of {len(samples)} requests were neither 200 nor 429 ({detail})")
        if not any(s.status == 200 for s in samples):
            failures.append("no request was answered 200: the evaluator served nothing")
        bad_retry = [s for s in samples if s.status == 429 and not valid_retry_after(s.retry_after)]
        if bad_retry:
            failures.append(f"{len(bad_retry)} of the 429 replies carried no valid Retry-After")
    if not health:
        failures.append("/health was never sampled")
    unhealthy = [s for s in health if s.status != 200]
    if unhealthy:
        failures.append(f"{len(unhealthy)} of {len(health)} /health samples were not 200")
    if args.health_latency_ms is not None:
        stats = summary["health"]["latencyMs"]  # type: ignore[index]
        p99 = stats.get("p99")  # type: ignore[union-attr]
        if p99 is not None and p99 > args.health_latency_ms:
            failures.append(f"/health p99 {p99:.1f}ms exceeded the {args.health_latency_ms:.1f}ms ceiling")
    return failures


def make_body(route: str, rng: random.Random, plies: int) -> bytes:
    if route == "rank":
        die1, die2 = rng.choice(DICE)
        item = {"pips": OPENING, "die1": die1, "die2": die2, "plies": plies}
    else:
        item = {"pips": OPENING}
    return json.dumps([item]).encode()  # batch size 1, always


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8090", help="sidecar base URL")
    parser.add_argument("--route", choices=["rank", "cube", "evaluate"], default="rank")
    parser.add_argument("--requests", type=positive_int, default=200, help="total single-position requests")
    parser.add_argument("--concurrency", type=positive_int, default=8, help="requests in flight at once")
    parser.add_argument("--plies", type=int, choices=(0, 1, 2), default=1, help="rank depth: 0/1 or 2")
    parser.add_argument("--seed", type=int, default=477, help="seed for the dice sequence")
    parser.add_argument("--timeout", type=positive_float, default=10.0, help="per-request timeout, seconds")
    parser.add_argument("--health-interval", type=positive_float, default=0.05, help="seconds between /health samples during the run")
    parser.add_argument("--unauthenticated", action="store_true", help="send no token (to see the 401 path)")
    parser.add_argument("--health-latency-ms", type=positive_float, default=None, help="fail the run if /health p99 exceeds this many milliseconds")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    token = None if args.unauthenticated else os.environ.get(TOKEN_ENV)
    redact = Redactor(token)
    say = lambda text: print(redact(text))  # noqa: E731
    # In --json mode stdout is the summary and nothing else.
    note = (lambda text: print(redact(text), file=sys.stderr)) if args.json else say  # noqa: E731

    if not args.unauthenticated and not token:
        note(f"{TOKEN_ENV} is not set; export it (or pass --unauthenticated to exercise the 401 path)")
        return 2

    rng = random.Random(args.seed)
    bodies = [make_body(args.route, rng, args.plies) for _ in range(args.requests)]
    path = "/" + args.route

    # /health is sampled from its own thread for the whole run, so the
    # report shows what an operator (and Render's health check) would have
    # seen while the evaluation slots were busy.
    health: List[Sample] = []
    stop = threading.Event()

    def poll_health() -> None:
        while not stop.is_set():
            health.append(one_request(args.url, "/health", None, None, args.timeout))
            stop.wait(args.health_interval)

    health_thread = threading.Thread(target=poll_health, daemon=True)

    note(f"bg-engine live load: {args.requests} x {args.route} (batch 1, plies {args.plies}) at concurrency {args.concurrency} -> {args.url}")
    health_thread.start()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        samples = list(pool.map(lambda body: one_request(args.url, path, body, token, args.timeout), bodies))
    elapsed = time.perf_counter() - started
    stop.set()
    health_thread.join(timeout=args.timeout + 1)

    statuses: Dict[str, int] = {}
    for s in samples:
        key = str(s.status) if s.status else "no-response"
        statuses[key] = statuses.get(key, 0) + 1
    ok = [s.ms for s in samples if s.status == 200]
    refused = [s for s in samples if s.status == 429]
    retry_after = sorted({s.retry_after for s in refused if s.retry_after})
    errors = sorted({s.error for s in samples if s.error and s.status != 429})
    health_statuses: Dict[str, int] = {}
    for s in health:
        key = str(s.status) if s.status else "no-response"
        health_statuses[key] = health_statuses.get(key, 0) + 1

    summary = {
        "url": args.url,
        "route": args.route,
        "batchSize": 1,
        "plies": args.plies,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "authenticated": token is not None,
        "elapsedSeconds": elapsed,
        "requestsPerSecond": args.requests / elapsed if elapsed > 0 else None,
        "statuses": statuses,
        "latencyMsAll": summarise([s.ms for s in samples]),
        "latencyMs200": summarise(ok),
        "evalMs200": summarise([s.eval_ms for s in samples if s.status == 200 and s.eval_ms is not None]),
        "retryAfterSeen": retry_after,
        "errorsSeen": errors[:5],
        "health": {"samples": len(health), "statuses": health_statuses, "latencyMs": summarise([s.ms for s in health])},
    }

    failures = verdict(args, samples, health, summary)
    summary["passed"] = not failures
    summary["failures"] = failures

    if args.json:
        say(json.dumps(summary, indent=2))
        return 0 if not failures else 1

    def line(name: str, stats: Dict[str, float]) -> str:
        if stats.get("count", 0) == 0:
            return f"  {name:<14} (none)"
        return (
            f"  {name:<14} n={stats['count']:<5} p50={stats['p50']:7.1f}  p95={stats['p95']:7.1f}  "
            f"p99={stats['p99']:7.1f}  max={stats['max']:7.1f}  mean={stats['mean']:7.1f}"
        )

    say(f"done in {elapsed:.2f}s ({summary['requestsPerSecond']:.1f} req/s)")
    say("statuses: " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))
    say("latency (ms):")
    say(line("all", summary["latencyMsAll"]))
    say(line("200 only", summary["latencyMs200"]))
    say(line("evalMs (200)", summary["evalMs200"]))
    if refused:
        say(f"429s: {len(refused)} (Retry-After seen: {', '.join(retry_after) or 'none'})")
    if errors:
        say("errors seen: " + "; ".join(errors[:5]))
    say(f"/health during the run: {len(health)} samples, statuses " + ", ".join(f"{k}: {v}" for k, v in sorted(health_statuses.items())))
    say(line("health", summary["health"]["latencyMs"]))
    if failures:
        say("FAIL:")
        for reason in failures:
            say(f"  - {reason}")
        return 1
    say("PASS: " + ("every request was 401 and /health stayed 200" if args.unauthenticated
                    else "every request was 200 or 429 (with a valid Retry-After), no transport errors, /health stayed 200"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
