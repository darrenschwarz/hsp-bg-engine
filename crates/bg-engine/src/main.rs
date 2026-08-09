//! bg-engine — reference sidecar for the game platform.
//!
//! Wraps wildbg behind a batch-first HTTP API. Three endpoints:
//!
//!   POST /rank      — N positions+dice -> ranked candidate moves with equity loss
//!   POST /evaluate  — N positions      -> cubeless probabilities
//!   POST /cube      — N positions      -> Janowski money-game cube decisions
//!   GET  /health    — liveness + engine identity
//!
//! Board format is wildbg's, unchanged, so the adapter lives in exactly one
//! place (the TypeScript client) rather than being half-translated here:
//!   [0] = opponent's bar, [1..24] = points, [25] = player's bar.
//!   Positive = side to move. No borne-off slot.

use axum::{Json, Router, http::StatusCode, routing::get, routing::post};
use engine::dice::Dice;
use engine::position::Position;
use engine::probabilities::Probabilities;
use logic::bg_move::BgMove;
use logic::cube::CubeInfo;
use logic::wildbg_api::{ScoreConfig, WildbgApi};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------- wire types

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RankRequest {
    /// 26-slot wildbg board, from the side-to-move's perspective.
    pips: Vec<i8>,
    die1: u32,
    die2: u32,
    /// 0/0 = money game, 1/1 = DMP. Anything else is rejected upstream.
    #[serde(default)]
    x_away: u32,
    #[serde(default)]
    o_away: u32,
    /// Search depth. 1 = the net's own evaluation (default, ~0.1ms/decision).
    /// 2 = look one roll further, averaging the opponent's best reply over all
    /// 21 dice (~500x slower, and the difference between "Advanced" and a
    /// genuine world-class bot).
    #[serde(default)]
    plies: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PositionRequest {
    pips: Vec<i8>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WireProbabilities {
    win: f32,
    win_gammon: f32,
    win_backgammon: f32,
    lose_gammon: f32,
    lose_backgammon: f32,
}

impl From<&Probabilities> for WireProbabilities {
    fn from(p: &Probabilities) -> Self {
        Self {
            win: p.win_normal + p.win_gammon + p.win_bg,
            win_gammon: p.win_gammon + p.win_bg,
            win_backgammon: p.win_bg,
            lose_gammon: p.lose_gammon + p.lose_bg,
            lose_backgammon: p.lose_bg,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WireMoveDetail {
    from: usize,
    to: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RankedMove {
    /// Checker movements, in wildbg's from/to pip notation.
    details: Vec<WireMoveDetail>,
    /// Resulting position, so the caller can verify against its own rules engine
    /// instead of trusting our move decomposition.
    resulting_pips: Vec<i8>,
    probabilities: WireProbabilities,
    /// Value under the requested score config (equity for money, win% for DMP).
    equity: f32,
    /// Equity given up versus the best move. Best move is always exactly 0.0.
    /// This is the number the difficulty sampler and the analyser both consume.
    equity_loss: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RankResult {
    moves: Vec<RankedMove>,
    /// Present when the position/dice were rejected; `moves` is then empty.
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchResponse<T> {
    results: Vec<T>,
    engine_id: String,
    eval_ms: f64,
    /// Total candidate positions evaluated across the batch. Useful for
    /// capacity planning: analysis batches are ~2000, live moves are ~20.
    positions_evaluated: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CubeResult {
    should_double: bool,
    should_accept: bool,
    equity_cubeless: f32,
    equity_no_double: f32,
    equity_double_take: f32,
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EvalResult {
    probabilities: Option<WireProbabilities>,
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Health {
    ok: bool,
    engine_id: String,
    /// Sanity signal: the opening position must evaluate near 50%.
    /// If this drifts far from 0.5 the wrong nets are loaded.
    opening_win_probability: f32,
}

// ------------------------------------------------------------------ plumbing

struct AppState {
    api: WildbgApi<engine::composite::CompositeEvaluator>,
    api2: WildbgApi<engine::multiply::MultiPlyEvaluator<engine::composite::CompositeEvaluator>>,
    engine_id: String,
}

fn parse_pips(pips: &[i8]) -> Result<Position, String> {
    let arr: [i8; 26] = pips
        .try_into()
        .map_err(|_| format!("expected 26 pips, got {}", pips.len()))?;
    Position::try_from(arr).map_err(|e| e.to_string())
}

fn score_config(x_away: u32, o_away: u32) -> Result<ScoreConfig, String> {
    ScoreConfig::try_from((x_away, o_away)).map_err(|e| e.to_string())
}

/// ORIENTATION TRAP — read before touching this.
///
/// wildbg has two APIs that return a position after a move, and they use
/// OPPOSITE conventions:
///
///   Evaluator::best_position()            -> sides ALREADY switched
///                                            (needs .sides_switched() to read back)
///   Evaluator::positions_and_probabilities -> "again from the perspective of x"
///     (what WildbgApi::all_moves calls)      i.e. NOT switched
///
/// Copying the `.sides_switched()` from wildbg's own `best_move()` into code
/// built on `all_moves()` double-flips the board. The failure is silent:
/// BgMove::new() can't diff two unrelated positions, so it returns ZERO move
/// details and every equity still looks perfectly sensible. This cost a
/// debugging cycle to find and is precisely why the plan insists on
/// property-based round-trip tests for the adapter.
///
/// Candidates from all_moves() are already in the mover's perspective, so we
/// serialise them straight out.
fn pips_of(position: &Position) -> Vec<i8> {
    (0..=25).map(|i| position.pip(i)).collect()
}

fn rank_one(state: &AppState, req: &RankRequest) -> (RankResult, usize) {
    let position = match parse_pips(&req.pips) {
        Ok(p) => p,
        Err(e) => return (RankResult { moves: vec![], error: Some(e) }, 0),
    };
    let dice = match Dice::try_from((req.die1 as usize, req.die2 as usize)) {
        Ok(d) => d,
        Err(e) => return (RankResult { moves: vec![], error: Some(e.to_string()) }, 0),
    };
    let config = match score_config(req.x_away, req.o_away) {
        Ok(c) => c,
        Err(e) => return (RankResult { moves: vec![], error: Some(e) }, 0),
    };

    // all_moves returns (resulting position, probabilities) for every legal
    // move sequence, best first under the given score config.
    let candidates = if req.plies >= 2 {
        state.api2.all_moves(&position, &dice, &config)
    } else {
        state.api.all_moves(&position, &dice, &config)
    };
    let n = candidates.len();
    let value = config.value();

    let best_equity = candidates.first().map(|(_, p)| value(p)).unwrap_or(0.0);

    let moves = candidates
        .iter()
        .map(|(new_position, probs)| {
            let equity = value(probs);
            // NO .sides_switched() here — see the note on pips_of().
            let bg_move = BgMove::new(&position, new_position, &dice);
            RankedMove {
                details: bg_move
                    .into_details()
                    .iter()
                    .map(|d| WireMoveDetail { from: d.from(), to: d.to() })
                    .collect(),
                resulting_pips: pips_of(new_position),
                probabilities: probs.into(),
                equity,
                equity_loss: best_equity - equity,
            }
        })
        .collect();

    (RankResult { moves, error: None }, n)
}

// ----------------------------------------------------------------- handlers

async fn rank(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(requests): Json<Vec<RankRequest>>,
) -> (StatusCode, Json<BatchResponse<RankResult>>) {
    let start = Instant::now();
    let mut evaluated = 0usize;
    let results = requests
        .iter()
        .map(|r| {
            let (res, n) = rank_one(&state, r);
            evaluated += n;
            res
        })
        .collect();
    (
        StatusCode::OK,
        Json(BatchResponse {
            results,
            engine_id: state.engine_id.clone(),
            eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            positions_evaluated: evaluated,
        }),
    )
}

async fn evaluate(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(requests): Json<Vec<PositionRequest>>,
) -> (StatusCode, Json<BatchResponse<EvalResult>>) {
    let start = Instant::now();
    let n = requests.len();
    let results = requests
        .iter()
        .map(|r| match parse_pips(&r.pips) {
            Ok(p) => EvalResult {
                probabilities: Some((&state.api.probabilities(&p)).into()),
                error: None,
            },
            Err(e) => EvalResult { probabilities: None, error: Some(e) },
        })
        .collect();
    (
        StatusCode::OK,
        Json(BatchResponse {
            results,
            engine_id: state.engine_id.clone(),
            eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            positions_evaluated: n,
        }),
    )
}

async fn cube(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(requests): Json<Vec<PositionRequest>>,
) -> (StatusCode, Json<BatchResponse<CubeResult>>) {
    let start = Instant::now();
    let n = requests.len();
    let results = requests
        .iter()
        .map(|r| match parse_pips(&r.pips) {
            Ok(p) => {
                let info: CubeInfo = state.api.cube_info(&p);
                CubeResult {
                    should_double: info.double(),
                    should_accept: info.accept(),
                    equity_cubeless: info.equity_cubeless(),
                    equity_no_double: info.equity_no_double(),
                    equity_double_take: info.equity_double_take(),
                    error: None,
                }
            }
            Err(e) => CubeResult {
                should_double: false,
                should_accept: false,
                equity_cubeless: 0.0,
                equity_no_double: 0.0,
                equity_double_take: 0.0,
                error: Some(e),
            },
        })
        .collect();
    (
        StatusCode::OK,
        Json(BatchResponse {
            results,
            engine_id: state.engine_id.clone(),
            eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            positions_evaluated: n,
        }),
    )
}

async fn health(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Json<Health> {
    let opening = Position::try_from(opening_pips()).expect("opening position is legal");
    let p = state.api.probabilities(&opening);
    Json(Health {
        ok: true,
        engine_id: state.engine_id.clone(),
        opening_win_probability: p.win_normal + p.win_gammon + p.win_bg,
    })
}

/// Standard opening position in wildbg's format.
fn opening_pips() -> [i8; 26] {
    let mut pips = [0i8; 26];
    pips[24] = 2;
    pips[13] = 5;
    pips[8] = 3;
    pips[6] = 5;
    pips[1] = -2;
    pips[12] = -5;
    pips[17] = -3;
    pips[19] = -5;
    pips
}

#[tokio::main]
async fn main() {
    let load_err = "failed to load neural nets — is neural-nets/ next to the binary, \
         and are they the STRONG nets from the `nets` branch rather than \
         the demo nets on `main`?";
    let api = WildbgApi::try_default().expect(load_err);
    // A second, independent copy wrapped for 2-ply. The nets are compiled into
    // the binary, so this costs memory, not load time.
    let api2 = WildbgApi::with_evaluator(engine::multiply::MultiPlyEvaluator {
        evaluator: engine::composite::CompositeEvaluator::try_default().expect(load_err),
    });

    let engine_id = std::env::var("BG_ENGINE_ID")
        .unwrap_or_else(|_| "wildbg@main+nets/unpinned".to_string());

    let state = Arc::new(AppState { api, api2, engine_id });

    let app = Router::new()
        .route("/health", get(health))
        .route("/rank", post(rank))
        .route("/evaluate", post(evaluate))
        .route("/cube", post(cube))
        .with_state(state);

    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8090);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    eprintln!("bg-engine listening on :{port}");
    axum::serve(listener, app).await.unwrap();
}
