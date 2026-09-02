# bg-engine wire contract v1 — canonical fixtures

Owned by this crate. `src/main.rs` (`fixture_tests`) builds the real response
types, serialises them and holds them equal to these files, so a fixture can
never drift from what the server actually sends.

| File | What it is |
|---|---|
| `health-ok.json` | `GET /health` when ready (HTTP 200) |
| `health-unhealthy-503.json` | `GET /health` when the loaded evaluator fails the opening sanity check (HTTP 503, `ok:false`, `error`) |
| `rank-ok.json` | `POST /rank` for one position (opening, 3-1, two candidates shown) — the canonical batch reply |
| `rank-missing-version.json` | canonical reply without `apiVersion` — clients must reject |
| `rank-future-version.json` | `apiVersion: 2` — clients must reject (exact match only) |
| `rank-missing-capability.json` | `rank.v1` absent from `capabilities` — clients must reject a rank reply |
| `rank-malformed-identity.json` | `engineId` is a mutable label, not `wildbg@<rev>+contact@<sha>+race@<sha>` — reject |
| `rank-missing-identity.json` | no `engineId` — reject; a client must never substitute its own label |
| `rank-result-count-mismatch.json` | two results for a one-position request — reject |

The additive `rank.match.v1` capability is present in every canonical reply
that advertises rank support. It covers 1-ply match-context scoring; a match
request at 2-ply is typed unsupported until the opponent reply is also chosen
in MWC. Match request/result examples live separately in
`../gp493-checker-context-v1.json`; the v1 canonical money result stays free of
the optional context/provenance fields so old money clients retain their
original result shape.

`MANIFEST.sha256` lists every fixture's SHA-256. The HSP repository carries
byte-identical copies under `src/backgammon/__tests__/fixtures/bg-engine-contract-v1/`
together with this manifest, and its tests re-hash the copies against it — so
the two sides are proven to be testing the same bytes.

The identity in the fixtures uses a made-up source revision; the two net
hashes are the real hashes of `neural-nets/contact.onnx` and `race.onnx`.
