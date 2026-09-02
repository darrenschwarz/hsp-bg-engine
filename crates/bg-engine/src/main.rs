//! bg-engine — reference sidecar for the game platform.
//!
//! Wraps wildbg behind a small HTTP API for LIVE play. Four endpoints:
//!
//!   POST /rank      — 1..8 positions+dice -> ranked candidate moves with equity loss
//!   POST /evaluate  — 1..8 positions      -> cubeless probabilities
//!   POST /cube      — 1..8 positions      -> Janowski money-game cube decisions
//!   GET  /health    — readiness + engine identity (200 ready, 503 not)
//!
//! This API serves live single-position traffic -- one table's next decision
//! -- and is NOT a bulk analysis interface (GP-477). The evaluation routes
//! require a bearer token, cap the body at 16 KiB and the batch at 8, accept
//! only the plies this build implements, and run the neural work on a bounded
//! blocking pool: a request that cannot start within 100 ms is refused with
//! 429 rather than queued behind the tables already being served. Post-match
//! analysis of a whole game is a different workload and needs a different
//! deployment, not a larger batch here.
//!
//! Board format is wildbg's, unchanged, so the adapter lives in exactly one
//! place (the TypeScript client) rather than being half-translated here:
//!   [0] = opponent's bar, [1..24] = points, [25] = player's bar.
//!   Positive = side to move. No borne-off slot.
//!
//! Every reply carries the same immutable `ContractMeta` (GP-476): the API
//! version, the capabilities this build implements, and an identity made of
//! the source commit and the two net hashes baked in at build time. See
//! `contract.rs`, and `fixtures/contract-v1/` for the canonical wire shapes
//! the HSP client is tested against.

mod auth;
mod capacity;
mod contract;
mod probe;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::header::{RETRY_AFTER, WWW_AUTHENTICATE};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, http::StatusCode, routing::get, routing::post};
use engine::dice::Dice;
use engine::position::Position;
use engine::probabilities::Probabilities;
use logic::bg_move::BgMove;
use logic::cube::CubeInfo;
use logic::match_equity::{
    CrawfordState, CubeOwner, MatchContext, supported as match_context_supported,
    validate as validate_match_context,
};
use logic::wildbg_api::{ScoreConfig, WildbgApi};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use auth::BearerToken;
use capacity::Capacity;
use contract::ContractMeta;

// ------------------------------------------------------------------- limits

/// The largest request body the evaluation routes read. Eight rank items
/// with 26 pips each serialise to well under 2 KiB; anything near this
/// limit is not live traffic.
pub const MAX_BODY_BYTES: usize = 16 * 1024;
/// The most positions one request may carry.
pub const MAX_BATCH: usize = 8;

// ---------------------------------------------------------------- wire types

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RankRequest {
    /// 26-slot wildbg board, from the side-to-move's perspective.
    pips: Vec<i8>,
    die1: u32,
    die2: u32,
    /// Legacy score fields: 0/0 = money and 1/1 = DMP when `context` is absent.
    /// With a match context these repeat its mover-relative points-away values.
    #[serde(default)]
    x_away: u32,
    #[serde(default)]
    o_away: u32,
    /// Additive GP-493 checker context. Absent retains the exact legacy money/DMP path.
    #[serde(default)]
    context: Option<WireCheckerContext>,
    /// Search depth. 0 or 1 = the net's own evaluation (default, ~0.1ms per
    /// decision). 2 = look one roll further, averaging the opponent's best
    /// reply over all 21 dice (~500x slower, and the difference between
    /// "Advanced" and a genuine world-class bot). Anything else is refused
    /// with 400 before any evaluation starts: see `Depth`.
    #[serde(default)]
    plies: u32,
}

#[cfg(test)]
mod checker_context_corpus_tests {
    use super::test_support::state_with;
    use super::*;
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn fixed_positions_rank_by_score_and_unsupported_live_cube_is_typed() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("gp493-checker-context-v1.json");
        let corpus: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let state = state_with(Ok(()), 1);
        for position in corpus["positions"].as_array().unwrap() {
            let mut tops = Vec::new();
            for case in position["cases"].as_array().unwrap() {
                let context: WireCheckerContext =
                    serde_json::from_value(case["context"].clone()).unwrap();
                let req = RankRequest {
                    pips: serde_json::from_value(position["pips"].clone()).unwrap(),
                    die1: position["dice"][0].as_u64().unwrap() as u32,
                    die2: position["dice"][1].as_u64().unwrap() as u32,
                    x_away: context.points_away_us.unwrap(),
                    o_away: context.points_away_them.unwrap(),
                    context: Some(context.clone()),
                    plies: 1,
                };
                let (result, _) = rank_one(&state, &req, Depth::OnePly);
                if let Some(code) = case.get("expectedErrorCode") {
                    assert_eq!(result.error_code.as_deref(), code.as_str());
                    assert!(result.moves.is_empty());
                    continue;
                }
                assert_eq!(result.error, None, "{} / {}", position["id"], case["label"]);
                assert_eq!(result.evaluated_context.as_ref(), Some(&context));
                assert_eq!(result.equity_units.as_deref(), Some("mwc"));
                assert_eq!(
                    result.evaluation_model.as_deref(),
                    Some("wildbg+kazaross-xg2-cubeless-mwc/v1")
                );
                let actual = serde_json::to_value(&result.moves[0].details).unwrap();
                assert_eq!(
                    actual, case["expectedTop"],
                    "{} / {}",
                    position["id"], case["label"]
                );
                if case["label"] == "gammon-go" || case["label"] == "gammon-save" {
                    tops.push(actual);
                }
            }
            assert_eq!(tops.len(), 2);
            assert_ne!(
                tops[0], tops[1],
                "{} must have a fixed score-dependent ranking change",
                position["id"]
            );
        }
    }

    #[test]
    fn explicit_money_context_preserves_the_legacy_rank_result() {
        let state = state_with(Ok(()), 1);
        let legacy = RankRequest {
            pips: opening_pips().to_vec(),
            die1: 3,
            die2: 1,
            x_away: 0,
            o_away: 0,
            context: None,
            plies: 1,
        };
        let explicit = RankRequest {
            context: Some(WireCheckerContext {
                mode: "money".into(),
                match_length: None,
                score_us: None,
                score_them: None,
                points_away_us: None,
                points_away_them: None,
                cube_enabled: None,
                cube_value: None,
                cube_owner: None,
                crawford_state: None,
            }),
            ..RankRequest {
                pips: legacy.pips.clone(),
                die1: legacy.die1,
                die2: legacy.die2,
                x_away: legacy.x_away,
                o_away: legacy.o_away,
                context: None,
                plies: legacy.plies,
            }
        };

        let (legacy_result, legacy_positions) = rank_one(&state, &legacy, Depth::OnePly);
        let (explicit_result, explicit_positions) = rank_one(&state, &explicit, Depth::OnePly);
        assert_eq!(legacy_positions, explicit_positions);
        assert_eq!(
            serde_json::to_value(&legacy_result).unwrap(),
            serde_json::to_value(&explicit_result).unwrap()
        );
        let wire = serde_json::to_value(&explicit_result).unwrap();
        assert!(wire.get("evaluatedContext").is_none());
        assert!(wire.get("equityUnits").is_none());
        assert!(wire.get("evaluationModel").is_none());
    }

    #[test]
    fn two_ply_match_context_is_typed_unsupported_until_every_ply_is_match_aware() {
        let state = state_with(Ok(()), 1);
        let context = WireCheckerContext {
            mode: "match".into(),
            match_length: Some(5),
            score_us: Some(3),
            score_them: Some(1),
            points_away_us: Some(2),
            points_away_them: Some(4),
            cube_enabled: Some(false),
            cube_value: Some(1),
            cube_owner: Some("centred".into()),
            crawford_state: Some("pre-crawford".into()),
        };
        let request = RankRequest {
            pips: opening_pips().to_vec(),
            die1: 3,
            die2: 1,
            x_away: 2,
            o_away: 4,
            context: Some(context),
            plies: 2,
        };
        let (result, positions) = rank_one(&state, &request, Depth::TwoPly);
        assert_eq!(positions, 0);
        assert_eq!(
            result.error_code.as_deref(),
            Some("unsupported_checker_context")
        );
        assert!(result.moves.is_empty());
        assert!(result.evaluated_context.is_none());
        assert!(result.equity_units.is_none());
        assert!(result.evaluation_model.is_none());
    }

    #[test]
    fn incoherent_crawford_context_is_typed_invalid_without_evaluation() {
        let state = state_with(Ok(()), 1);
        let request = RankRequest {
            pips: opening_pips().to_vec(),
            die1: 3,
            die2: 1,
            x_away: 5,
            o_away: 5,
            context: Some(WireCheckerContext {
                mode: "match".into(),
                match_length: Some(5),
                score_us: Some(0),
                score_them: Some(0),
                points_away_us: Some(5),
                points_away_them: Some(5),
                cube_enabled: Some(false),
                cube_value: Some(1),
                cube_owner: Some("centred".into()),
                crawford_state: Some("crawford".into()),
            }),
            plies: 1,
        };
        let (result, positions) = rank_one(&state, &request, Depth::OnePly);
        assert_eq!(positions, 0);
        assert_eq!(
            result.error_code.as_deref(),
            Some("invalid_checker_context")
        );
        assert!(result.moves.is_empty());
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct WireCheckerContext {
    mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    match_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score_us: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score_them: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    points_away_us: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    points_away_them: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cube_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cube_value: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cube_owner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    crawford_state: Option<String>,
}

impl WireCheckerContext {
    fn match_context(&self) -> Result<MatchContext, String> {
        if self.mode != "match" {
            return Err("context.mode must be money or match".into());
        }
        let cube_owner = match self.cube_owner.as_deref() {
            Some("us") => CubeOwner::Us,
            Some("them") => CubeOwner::Them,
            Some("centred") => CubeOwner::Centred,
            _ => return Err("context.cubeOwner must be us, them or centred".into()),
        };
        let crawford_state = match self.crawford_state.as_deref() {
            Some("pre-crawford") => CrawfordState::PreCrawford,
            Some("crawford") => CrawfordState::Crawford,
            Some("post-crawford") => CrawfordState::PostCrawford,
            _ => return Err("context.crawfordState is invalid".into()),
        };
        let ctx = MatchContext {
            match_length: self.match_length.ok_or("context.matchLength is required")?,
            score_us: self.score_us.ok_or("context.scoreUs is required")?,
            score_them: self.score_them.ok_or("context.scoreThem is required")?,
            points_away_us: self
                .points_away_us
                .ok_or("context.pointsAwayUs is required")?,
            points_away_them: self
                .points_away_them
                .ok_or("context.pointsAwayThem is required")?,
            cube_enabled: self.cube_enabled.ok_or("context.cubeEnabled is required")?,
            cube_value: self.cube_value.ok_or("context.cubeValue is required")?,
            cube_owner,
            crawford_state,
        };
        validate_match_context(&ctx)?;
        Ok(ctx)
    }
}

/// The evaluators this build implements, and the only depths a request may
/// name. `plies.1` and `plies.2` in `contract::CAPABILITIES` are exactly this
/// list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    OnePly,
    TwoPly,
}

impl Depth {
    /// 0 and 1 are the net's own evaluation; 2 is the 2-ply evaluator. There
    /// is no 3-ply and there is no "as deep as you can": a depth this build
    /// does not implement is a client error, not a request to do its best.
    fn for_plies(plies: u32) -> Result<Depth, String> {
        match plies {
            0 | 1 => Ok(Depth::OnePly),
            2 => Ok(Depth::TwoPly),
            other => Err(format!(
                "plies must be 0, 1 or 2 (got {other}): this build implements plies.1 and plies.2 only"
            )),
        }
    }
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
    /// Value under the requested score config: money equity, DMP win chance, or MWC.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evaluated_context: Option<WireCheckerContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    equity_units: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evaluation_model: Option<String>,
}

/// One reply to a batch. `contract` is the same `ContractMeta` on every
/// reply this process ever sends; the HSP client validates it before it
/// reads a single result.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchResponse<T> {
    results: Vec<T>,
    #[serde(flatten)]
    contract: ContractMeta,
    /// Time spent in the evaluator itself, excluding any wait for capacity.
    eval_ms: f64,
    /// Total candidate positions evaluated across the batch. Useful for
    /// capacity planning: a live move is ~20 at 1-ply, a batch is at most 8 moves.
    positions_evaluated: usize,
}

/// The body of every refusal this service writes itself: 400, 401, 429, 500.
/// (axum's own 413 for an oversized body and 415/422 for a body it cannot
/// read as the request type are plain text.)
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct ErrorBody {
    error: String,
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

/// `/health`. `ok` is the readiness verdict of `contract::health_check` over
/// the loaded evaluator's opening-position evaluation, and the HTTP status
/// agrees with it: 200 when ready, 503 when not, with `error` saying why.
/// The same body answers the evaluation routes while not ready.
#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Health {
    ok: bool,
    #[serde(flatten)]
    contract: ContractMeta,
    /// The loaded evaluator's win probability for the side on roll from the
    /// opening position. Must sit inside `contract::OPENING_WIN_RANGE`.
    opening_win_probability: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ------------------------------------------------------------------ plumbing

/// The readiness verdict: the loaded evaluator's opening-position evaluation,
/// judged by `contract::health_check`. Computed ONCE at startup and cached
/// here, and authoritative for every route -- `/health` reports it, and the
/// evaluation routes refuse to serve while it is `Err` (GP-476). The
/// evaluator is immutable after load, so re-running the check could never
/// say anything different.
struct Readiness {
    opening_win_probability: f32,
    verdict: Result<(), String>,
}

struct AppState {
    api: WildbgApi<engine::composite::CompositeEvaluator>,
    api2: WildbgApi<engine::multiply::MultiPlyEvaluator<engine::composite::CompositeEvaluator>>,
    /// Built once at startup; every reply clones this same value.
    contract: ContractMeta,
    readiness: Readiness,
    /// The bearer token the evaluation routes require (GP-477).
    auth: BearerToken,
    /// The bound on concurrent neural evaluation (GP-477).
    capacity: Capacity,
}

impl AppState {
    /// The `/health` body for the cached verdict.
    fn health_report(&self) -> Health {
        Health {
            ok: self.readiness.verdict.is_ok(),
            contract: self.contract.clone(),
            opening_win_probability: self.readiness.opening_win_probability,
            error: self.readiness.verdict.clone().err(),
        }
    }
}

/// Every way an evaluation route declines to evaluate, in the order a
/// request meets them: credentials (middleware, before the body is read),
/// then the request itself, then readiness, then capacity. Each is a
/// complete HTTP reply.
#[derive(Debug)]
enum Refusal {
    /// 401, from the middleware. A fixed body: the reply never says whether
    /// the header was missing, malformed or wrong.
    Unauthorized,
    /// 400: an empty or oversized batch, or a depth this build does not implement.
    BadRequest(String),
    /// 503 with the same body `/health` gives, so there is one message to read.
    NotReady(Health),
    /// 429 with `Retry-After`: no evaluation slot came free within the queue wait.
    Saturated {
        bound: usize,
        in_use: usize,
        waited_ms: u128,
    },
    /// 500: the evaluator did not return.
    Internal(String),
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let error = |text: String| Json(ErrorBody { error: text });
        match self {
            Refusal::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                [(WWW_AUTHENTICATE, "Bearer realm=\"bg-engine\"")],
                error("unauthorized".to_string()),
            )
                .into_response(),
            Refusal::BadRequest(text) => (StatusCode::BAD_REQUEST, error(text)).into_response(),
            Refusal::NotReady(health) => (StatusCode::SERVICE_UNAVAILABLE, Json(health)).into_response(),
            Refusal::Saturated {
                bound,
                in_use,
                waited_ms,
            } => (
                StatusCode::TOO_MANY_REQUESTS,
                [(RETRY_AFTER, capacity::RETRY_AFTER_SECONDS.to_string())],
                error(format!(
                    "evaluation capacity saturated: {in_use} of {bound} evaluation slot(s) still busy after \
                     waiting {waited_ms}ms; this API serves live single-position traffic -- retry after {}s \
                     or send less at once",
                    capacity::RETRY_AFTER_SECONDS
                )),
            )
                .into_response(),
            Refusal::Internal(text) => (StatusCode::INTERNAL_SERVER_ERROR, error(text)).into_response(),
        }
    }
}

impl From<capacity::Refused> for Refusal {
    fn from(refused: capacity::Refused) -> Self {
        match refused {
            capacity::Refused::Saturated {
                bound,
                in_use,
                waited,
            } => Refusal::Saturated {
                bound,
                in_use,
                waited_ms: waited.as_millis(),
            },
            capacity::Refused::Failed(text) => Refusal::Internal(text),
        }
    }
}

/// The gate every evaluation handler passes through before it evaluates. An
/// evaluator that failed its invariant at startup serves no evaluation
/// traffic at all.
fn require_ready(state: &AppState) -> Result<(), Refusal> {
    match &state.readiness.verdict {
        Ok(()) => Ok(()),
        Err(_) => Err(Refusal::NotReady(state.health_report())),
    }
}

/// A batch must carry between one and `MAX_BATCH` positions. Checked before
/// readiness and before any wait for capacity: a request this API will not
/// serve is answered without costing the tables it does serve anything.
fn require_batch(len: usize) -> Result<(), Refusal> {
    if len == 0 {
        return Err(Refusal::BadRequest(
            "the batch is empty: send between 1 and 8 positions".to_string(),
        ));
    }
    if len > MAX_BATCH {
        return Err(Refusal::BadRequest(format!(
            "the batch carries {len} positions, more than the maximum of {MAX_BATCH}: \
             this API serves live single-position traffic, not bulk analysis"
        )));
    }
    Ok(())
}

/// The auth middleware on the evaluation routes: runs before the handler
/// and its extractors, so a refused request has its headers read and
/// nothing else -- no JSON, no evaluation, no wait for capacity.
async fn require_bearer(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    match state.auth.check(request.headers()) {
        Ok(()) => next.run(request).await,
        Err(auth::Unauthorized) => Refusal::Unauthorized.into_response(),
    }
}

/// The whole service. `/health` is public; the evaluation routes sit behind
/// the bearer check and the body limit, in that order.
fn app(state: Arc<AppState>) -> Router {
    let evaluation = Router::new()
        .route("/rank", post(rank))
        .route("/evaluate", post(evaluate))
        .route("/cube", post(cube))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));
    Router::new()
        .route("/health", get(health))
        .merge(evaluation)
        .with_state(state)
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

fn rank_score_config(
    req: &RankRequest,
) -> Result<
    (
        ScoreConfig,
        Option<WireCheckerContext>,
        Option<String>,
        Option<String>,
    ),
    (String, Option<String>),
> {
    let Some(wire) = req.context.as_ref() else {
        return score_config(req.x_away, req.o_away)
            .map(|config| (config, None, None, None))
            .map_err(|e| (e, None));
    };
    if wire.mode == "money" {
        if req.x_away != 0 || req.o_away != 0 {
            return Err((
                "money context requires xAway=0 and oAway=0".into(),
                Some("invalid_checker_context".into()),
            ));
        }
        return Ok((ScoreConfig::MoneyGame, None, None, None));
    }
    let ctx = wire
        .match_context()
        .map_err(|e| (e, Some("invalid_checker_context".into())))?;
    if req.x_away != ctx.points_away_us || req.o_away != ctx.points_away_them {
        return Err((
            "xAway/oAway must match the canonical context points-away fields".into(),
            Some("invalid_checker_context".into()),
        ));
    }
    if !match_context_supported(&ctx) {
        let reason = if !ctx.cube_enabled
            && ctx.crawford_state == CrawfordState::PreCrawford
            && ctx.match_length > 1
            && (ctx.points_away_us == 1 || ctx.points_away_them == 1)
        {
            "global cube-disabled one-away context needs a dedicated no-cube MET"
        } else {
            "live-cube checker context is not modelled; no money-equity substitution was used"
        };
        return Err((reason.into(), Some("unsupported_checker_context".into())));
    }
    Ok((
        ScoreConfig::Match(ctx),
        Some(wire.clone()),
        Some("mwc".into()),
        Some("wildbg+kazaross-xg2-cubeless-mwc/v1".into()),
    ))
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

fn rank_one(state: &AppState, req: &RankRequest, depth: Depth) -> (RankResult, usize) {
    let position = match parse_pips(&req.pips) {
        Ok(p) => p,
        Err(e) => {
            return (
                RankResult {
                    moves: vec![],
                    error: Some(e),
                    error_code: None,
                    evaluated_context: None,
                    equity_units: None,
                    evaluation_model: None,
                },
                0,
            );
        }
    };
    let dice = match Dice::try_from((req.die1 as usize, req.die2 as usize)) {
        Ok(d) => d,
        Err(e) => {
            return (
                RankResult {
                    moves: vec![],
                    error: Some(e.to_string()),
                    error_code: None,
                    evaluated_context: None,
                    equity_units: None,
                    evaluation_model: None,
                },
                0,
            );
        }
    };
    let (config, evaluated_context, equity_units, evaluation_model) = match rank_score_config(req) {
        Ok(c) => c,
        Err((e, error_code)) => {
            return (
                RankResult {
                    moves: vec![],
                    error: Some(e),
                    error_code,
                    evaluated_context: None,
                    equity_units: None,
                    evaluation_model: None,
                },
                0,
            );
        }
    };

    if depth == Depth::TwoPly && matches!(&config, ScoreConfig::Match(_)) {
        return (
            RankResult {
                moves: vec![],
                error: Some(
                    "2-ply opponent reply selection is money-equity-only; match context requires plies=1"
                        .into(),
                ),
                error_code: Some("unsupported_checker_context".into()),
                evaluated_context: None,
                equity_units: None,
                evaluation_model: None,
            },
            0,
        );
    }

    // all_moves returns (resulting position, probabilities) for every legal
    // move sequence, best first under the given score config.
    let candidates = match depth {
        Depth::TwoPly => state.api2.all_moves(&position, &dice, &config),
        Depth::OnePly => state.api.all_moves(&position, &dice, &config),
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
                    .map(|d| WireMoveDetail {
                        from: d.from(),
                        to: d.to(),
                    })
                    .collect(),
                resulting_pips: pips_of(new_position),
                probabilities: probs.into(),
                equity,
                equity_loss: best_equity - equity,
            }
        })
        .collect();

    (
        RankResult {
            moves,
            error: None,
            error_code: None,
            evaluated_context,
            equity_units,
            evaluation_model,
        },
        n,
    )
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

/// The readiness check, run once at startup against the loaded 1-ply
/// evaluator: evaluate the opening position and judge the answer.
fn assess(api: &WildbgApi<engine::composite::CompositeEvaluator>) -> Readiness {
    let opening = Position::try_from(opening_pips()).expect("opening position is legal");
    let p = api.probabilities(&opening);
    let win = p.win_normal + p.win_gammon + p.win_bg;
    Readiness {
        opening_win_probability: win,
        verdict: contract::health_check(win),
    }
}

// ----------------------------------------------------------------- handlers
//
// Each evaluation handler runs the same sequence: the request is checked
// (batch size, depth) and readiness is checked, both without touching the
// evaluator; only then does it wait -- briefly -- for an evaluation slot and
// hand the neural work to the blocking pool. The reply is assembled from
// what comes back. Nothing on the async runtime ever runs inference.

/// What the blocking pool hands back: the results, the candidate positions
/// evaluated, and the evaluator's own wall time.
struct Evaluated<T> {
    results: Vec<T>,
    positions: usize,
    eval_ms: f64,
}

/// The reply for a completed batch.
fn reply<T>(state: &AppState, evaluated: Evaluated<T>) -> (StatusCode, Json<BatchResponse<T>>) {
    (
        StatusCode::OK,
        Json(BatchResponse {
            results: evaluated.results,
            contract: state.contract.clone(),
            eval_ms: evaluated.eval_ms,
            positions_evaluated: evaluated.positions,
        }),
    )
}

async fn rank(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<RankRequest>>,
) -> Result<(StatusCode, Json<BatchResponse<RankResult>>), Refusal> {
    require_batch(requests.len())?;
    let depths = requests
        .iter()
        .map(|r| Depth::for_plies(r.plies))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Refusal::BadRequest)?;
    require_ready(&state)?;

    let worker = state.clone();
    let evaluated = state
        .capacity
        .run_blocking(move || {
            let start = Instant::now();
            let mut positions = 0usize;
            let results = requests
                .iter()
                .zip(depths)
                .map(|(r, depth)| {
                    let (res, n) = rank_one(&worker, r, depth);
                    positions += n;
                    res
                })
                .collect();
            Evaluated {
                results,
                positions,
                eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            }
        })
        .await?;
    Ok(reply(&state, evaluated))
}

async fn evaluate(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<PositionRequest>>,
) -> Result<(StatusCode, Json<BatchResponse<EvalResult>>), Refusal> {
    require_batch(requests.len())?;
    require_ready(&state)?;

    let worker = state.clone();
    let evaluated = state
        .capacity
        .run_blocking(move || {
            let start = Instant::now();
            let results = requests
                .iter()
                .map(|r| match parse_pips(&r.pips) {
                    Ok(p) => EvalResult {
                        probabilities: Some((&worker.api.probabilities(&p)).into()),
                        error: None,
                    },
                    Err(e) => EvalResult {
                        probabilities: None,
                        error: Some(e),
                    },
                })
                .collect();
            Evaluated {
                results,
                positions: requests.len(),
                eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            }
        })
        .await?;
    Ok(reply(&state, evaluated))
}

async fn cube(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<PositionRequest>>,
) -> Result<(StatusCode, Json<BatchResponse<CubeResult>>), Refusal> {
    require_batch(requests.len())?;
    require_ready(&state)?;

    let worker = state.clone();
    let evaluated = state
        .capacity
        .run_blocking(move || {
            let start = Instant::now();
            let results = requests
                .iter()
                .map(|r| match parse_pips(&r.pips) {
                    Ok(p) => {
                        let info: CubeInfo = worker.api.cube_info(&p);
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
            Evaluated {
                results,
                positions: requests.len(),
                eval_ms: start.elapsed().as_secs_f64() * 1000.0,
            }
        })
        .await?;
    Ok(reply(&state, evaluated))
}

async fn health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Health>) {
    let report = state.health_report();
    let status = if report.ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report))
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8090);

    // `bg-engine --health [url]`: the container's health probe, not a server.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--health") {
        std::process::exit(probe::run(args.get(2).map(String::as_str), port));
    }

    // Configuration first, before the nets are loaded: a misconfigured
    // service fails in under a second with one line saying what to set.
    // There is no unauthenticated mode and no silent clamp of the bound.
    let auth = BearerToken::from_env().unwrap_or_else(|e| {
        eprintln!("bg-engine: refusing to start: {e}");
        std::process::exit(2);
    });
    let bound = capacity::max_concurrent_from_env().unwrap_or_else(|e| {
        eprintln!("bg-engine: refusing to start: {e}");
        std::process::exit(2);
    });

    let load_err = "failed to load the neural nets compiled into this binary -- \
         the build embeds neural-nets/contact.onnx and race.onnx; if they load \
         and this still fails, the nets are not the ONNX shapes the engine expects";
    let api = WildbgApi::try_default().expect(load_err);
    // A second, independent copy wrapped for 2-ply. The nets are compiled into
    // the binary, so this costs memory, not load time.
    let api2 = WildbgApi::with_evaluator(engine::multiply::MultiPlyEvaluator {
        evaluator: engine::composite::CompositeEvaluator::try_default().expect(load_err),
    });

    // The readiness verdict, once, against the evaluator we just loaded. It
    // is cached in the state and authoritative: /health reports it, and the
    // evaluation routes refuse to serve while it says not ready.
    let readiness = assess(&api);
    let state = Arc::new(AppState {
        api,
        api2,
        contract: ContractMeta::from_build(),
        readiness,
        auth,
        capacity: Capacity::new(bound),
    });

    eprintln!(
        "bg-engine {} (api v{})",
        state.contract.engine_id, state.contract.api_version
    );
    match &state.readiness.verdict {
        Ok(()) => eprintln!(
            "opening win probability {:.4}: ready",
            state.readiness.opening_win_probability
        ),
        Err(e) => eprintln!(
            "NOT READY: {e} -- /health answers 503 and /rank, /evaluate and /cube refuse to serve"
        ),
    }
    // The token itself is never logged, here or anywhere.
    eprintln!(
        "live-traffic limits: bearer token required on /rank, /evaluate, /cube (/health public); \
         body <= {} KiB; batch 1..={MAX_BATCH}; plies 0-2; {} concurrent evaluation(s), \
         queue wait {}ms then 429",
        MAX_BODY_BYTES / 1024,
        state.capacity.bound(),
        capacity::QUEUE_WAIT.as_millis()
    );

    let app = app(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .unwrap();
    eprintln!("bg-engine listening on :{port}");
    axum::serve(listener, app).await.unwrap();
}

// ------------------------------------------------------------------ fixtures
//
// The canonical wire shapes live in fixtures/contract-v1/ and are owned by
// this crate. These tests build the ACTUAL response types above, serialise
// them, and hold them equal to the fixtures; the rejection fixtures are held
// equal to the canonical one with exactly the intended defect; and the rank
// fixture's resulting positions are checked to be legal 3-1 plays from the
// opening with the real engine. The HSP repository carries byte-identical
// copies plus the manifest, verified by hash on its side.

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use contract::Identity;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::PathBuf;

    const FIXTURE_IDENTITY: Identity = Identity {
        source_revision: "f1d2d2f924e986ac86fdf7b36c94bcdf32beec15",
        contact_sha256: "cdb4763d4a1ba2967bcafe97799e6a7a89d3fb43f7f584dabd2a7602779e06f0",
        race_sha256: "f16c0265c8d1097a6fcf7ec97b5e40c42857612c9d7019622476c1207c1c293b",
    };

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("contract-v1")
    }

    fn fixture(name: &str) -> Value {
        let path = fixture_dir().join(name);
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn meta() -> ContractMeta {
        ContractMeta::for_identity(&FIXTURE_IDENTITY)
    }

    fn move_a() -> RankedMove {
        // 8/5 6/5 from the opening, white (the mover) to play 3-1.
        RankedMove {
            details: vec![
                WireMoveDetail { from: 8, to: 5 },
                WireMoveDetail { from: 6, to: 5 },
            ],
            resulting_pips: vec![
                0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
            ],
            probabilities: WireProbabilities {
                win: 0.5625,
                win_gammon: 0.125,
                win_backgammon: 0.0625,
                lose_gammon: 0.125,
                lose_backgammon: 0.0625,
            },
            equity: 0.25,
            equity_loss: 0.0,
        }
    }

    fn move_b() -> RankedMove {
        // 24/21 24/23.
        RankedMove {
            details: vec![
                WireMoveDetail { from: 24, to: 21 },
                WireMoveDetail { from: 24, to: 23 },
            ],
            resulting_pips: vec![
                0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 1, 0, 1, 0, 0,
            ],
            probabilities: WireProbabilities {
                win: 0.5,
                win_gammon: 0.125,
                win_backgammon: 0.0625,
                lose_gammon: 0.1875,
                lose_backgammon: 0.0625,
            },
            equity: 0.125,
            equity_loss: 0.125,
        }
    }

    fn canonical_rank() -> BatchResponse<RankResult> {
        BatchResponse {
            results: vec![RankResult {
                moves: vec![move_a(), move_b()],
                error: None,
                error_code: None,
                evaluated_context: None,
                equity_units: None,
                evaluation_model: None,
            }],
            contract: meta(),
            eval_ms: 0.5,
            positions_evaluated: 2,
        }
    }

    #[test]
    fn health_ok_fixture_is_the_real_health_type() {
        let health = Health {
            ok: true,
            contract: meta(),
            opening_win_probability: 0.5,
            error: None,
        };
        assert_eq!(
            serde_json::to_value(&health).unwrap(),
            fixture("health-ok.json")
        );
    }

    #[test]
    fn health_503_fixture_is_the_real_health_type_with_the_real_verdict() {
        let error = contract::health_check(0.875).unwrap_err();
        let health = Health {
            ok: false,
            contract: meta(),
            opening_win_probability: 0.875,
            error: Some(error),
        };
        assert_eq!(
            serde_json::to_value(&health).unwrap(),
            fixture("health-unhealthy-503.json")
        );
    }

    #[test]
    fn rank_ok_fixture_is_the_real_batch_type() {
        assert_eq!(
            serde_json::to_value(canonical_rank()).unwrap(),
            fixture("rank-ok.json")
        );
    }

    #[test]
    fn rank_ok_fixture_has_the_full_capability_set_and_a_structured_identity() {
        let ok = fixture("rank-ok.json");
        assert_eq!(ok["apiVersion"], json!(1));
        assert_eq!(ok["capabilities"], json!(contract::CAPABILITIES.to_vec()));
        assert_eq!(
            ok["engineId"],
            json!(contract::engine_id(&FIXTURE_IDENTITY))
        );
    }

    #[test]
    fn each_rejection_fixture_is_the_canonical_reply_with_exactly_one_defect() {
        let ok = fixture("rank-ok.json");

        let mut missing_version = ok.clone();
        missing_version
            .as_object_mut()
            .unwrap()
            .remove("apiVersion");
        assert_eq!(fixture("rank-missing-version.json"), missing_version);

        let mut future_version = ok.clone();
        future_version["apiVersion"] = json!(2);
        assert_eq!(fixture("rank-future-version.json"), future_version);

        let mut missing_capability = ok.clone();
        missing_capability["capabilities"] =
            json!(["evaluate.v1", "cube.money.v1", "plies.1", "plies.2"]);
        assert_eq!(fixture("rank-missing-capability.json"), missing_capability);

        let mut malformed_identity = ok.clone();
        malformed_identity["engineId"] = json!("wildbg@main+nets/unpinned");
        assert_eq!(fixture("rank-malformed-identity.json"), malformed_identity);

        let mut missing_identity = ok.clone();
        missing_identity.as_object_mut().unwrap().remove("engineId");
        assert_eq!(fixture("rank-missing-identity.json"), missing_identity);

        let mut count_mismatch = ok.clone();
        let one = ok["results"][0].clone();
        count_mismatch["results"] = json!([one.clone(), one]);
        assert_eq!(fixture("rank-result-count-mismatch.json"), count_mismatch);
    }

    #[test]
    fn rank_ok_fixture_moves_are_legal_3_1_plays_from_the_opening_per_the_real_engine() {
        let api = WildbgApi::try_default().expect("nets load");
        let opening = Position::try_from(opening_pips()).unwrap();
        let dice = Dice::try_from((3usize, 1usize)).unwrap();
        let legal: Vec<Position> = api
            .all_moves(&opening, &dice, &score_config(0, 0).unwrap())
            .into_iter()
            .map(|(position, _)| position)
            .collect();
        for wire in [move_a(), move_b()] {
            let arr: [i8; 26] = wire.resulting_pips.as_slice().try_into().unwrap();
            let position = Position::try_from(arr).expect("fixture position is well formed");
            assert!(
                legal.contains(&position),
                "fixture move {:?} is not a legal 3-1 play",
                wire.details
                    .iter()
                    .map(|d| (d.from, d.to))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_manifest_matches_every_fixture_byte_for_byte() {
        let manifest = fs::read_to_string(fixture_dir().join("MANIFEST.sha256")).unwrap();
        let mut listed = 0;
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let (expected, name) = line.split_once("  ").expect("`<sha256>  <file>` lines");
            let bytes =
                fs::read(fixture_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let actual: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(actual, expected, "{name} does not match the manifest");
            listed += 1;
        }
        let on_disk = fs::read_dir(fixture_dir())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "json")
            })
            .count();
        assert_eq!(listed, on_disk, "every fixture must be in the manifest");
        assert_eq!(listed, 9);
    }
}

// ------------------------------------------------------------- test support
//
// Handler- and router-level tests share one way of building the state: the
// nets loaded for real, as at startup, with the readiness verdict, the token
// and the evaluation bound under the test's control.

#[cfg(test)]
mod test_support {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::Value;

    /// The token every test state is configured with (32 hex characters).
    pub const TEST_TOKEN: &str = "0123456789abcdef0123456789abcdef";

    pub fn state_with(verdict: Result<(), String>, bound: usize) -> Arc<AppState> {
        let api = WildbgApi::try_default().expect("nets load");
        let api2 = WildbgApi::with_evaluator(engine::multiply::MultiPlyEvaluator {
            evaluator: engine::composite::CompositeEvaluator::try_default().expect("nets load"),
        });
        Arc::new(AppState {
            api,
            api2,
            contract: ContractMeta::from_build(),
            readiness: Readiness {
                opening_win_probability: 0.875,
                verdict,
            },
            auth: BearerToken::new(TEST_TOKEN).unwrap(),
            capacity: Capacity::new(bound),
        })
    }

    pub async fn body_of(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    pub async fn text_of(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn rank_request() -> Vec<RankRequest> {
        vec![RankRequest {
            pips: opening_pips().to_vec(),
            die1: 3,
            die2: 1,
            x_away: 0,
            o_away: 0,
            context: None,
            plies: 1,
        }]
    }

    pub fn position_request() -> Vec<PositionRequest> {
        vec![PositionRequest {
            pips: opening_pips().to_vec(),
        }]
    }
}

// ------------------------------------------------------------------- gating
//
// The readiness verdict is data on AppState, so the gate can be tested by
// calling the handlers with a state whose verdict is forced -- no server,
// no network. The nets are loaded for real, as at startup.

#[cfg(test)]
mod gating_tests {
    use super::test_support::*;
    use super::*;
    use serde_json::Value;

    fn state_with(verdict: Result<(), String>) -> Arc<AppState> {
        super::test_support::state_with(verdict, 1)
    }

    #[tokio::test]
    async fn a_failed_startup_verdict_closes_every_evaluation_route_and_health_says_why() {
        let state = state_with(Err("forced: evaluator failed its invariant".to_string()));

        let rank = rank(State(state.clone()), Json(rank_request()))
            .await
            .into_response();
        let evaluate = evaluate(State(state.clone()), Json(position_request()))
            .await
            .into_response();
        let cube = cube(State(state.clone()), Json(position_request()))
            .await
            .into_response();
        let health = health(State(state.clone())).await.into_response();

        for (name, response) in [
            ("rank", rank),
            ("evaluate", evaluate),
            ("cube", cube),
            ("health", health),
        ] {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{name}");
            let body = body_of(response).await;
            assert_eq!(body["ok"], Value::Bool(false), "{name}");
            assert_eq!(
                body["error"],
                Value::String("forced: evaluator failed its invariant".into()),
                "{name}"
            );
            assert_eq!(
                body["engineId"],
                Value::String(state.contract.engine_id.clone()),
                "{name}"
            );
            assert!(
                body.get("results").is_none(),
                "{name} must serve no results"
            );
        }
    }

    #[tokio::test]
    async fn a_passing_verdict_serves_evaluation_traffic_and_health_is_200() {
        let state = state_with(Ok(()));

        let rank = rank(State(state.clone()), Json(rank_request()))
            .await
            .into_response();
        assert_eq!(rank.status(), StatusCode::OK);
        let body = body_of(rank).await;
        assert_eq!(body["results"].as_array().map(Vec::len), Some(1));
        assert!(!body["results"][0]["moves"].as_array().unwrap().is_empty());
        assert_eq!(body["apiVersion"], Value::from(1));

        let evaluate = evaluate(State(state.clone()), Json(position_request()))
            .await
            .into_response();
        assert_eq!(evaluate.status(), StatusCode::OK);
        let cube = cube(State(state.clone()), Json(position_request()))
            .await
            .into_response();
        assert_eq!(cube.status(), StatusCode::OK);

        let health = health(State(state)).await.into_response();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(body_of(health).await["ok"], Value::Bool(true));
    }

    #[test]
    fn the_real_startup_assessment_passes_on_the_compiled_nets() {
        let api = WildbgApi::try_default().expect("nets load");
        let readiness = assess(&api);
        assert!(
            readiness.verdict.is_ok(),
            "{:?} (opening win {})",
            readiness.verdict,
            readiness.opening_win_probability
        );
        assert!(contract::OPENING_WIN_RANGE.contains(&readiness.opening_win_probability));
    }
}

// ------------------------------------------------------------------- access
//
// GP-477. These go through the real Router (middleware, body limit, JSON
// extraction, handler) with tower's `oneshot`, so the ORDER of refusals is
// what is tested: credentials before the body is read as JSON, the body
// limit before the batch is looked at, the batch and depth before readiness
// and before any wait for capacity. A state whose only permit is already
// held by a blocked worker makes "before capacity" observable: a request that
// reached the semaphore would wait 100 ms and answer 429, so a prompt 400 or
// 401 proves it never got there.

#[cfg(test)]
mod access_tests {
    use super::test_support::*;
    use super::*;
    use axum::body::Body;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::http::{Method, Request};
    use serde_json::{Value, json};
    use std::sync::mpsc;
    use tower::ServiceExt;

    fn request(
        method: Method,
        path: &str,
        auth: Option<&str>,
        body: impl Into<Body>,
    ) -> Request<Body> {
        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header(CONTENT_TYPE, "application/json");
        if let Some(value) = auth {
            builder = builder.header(AUTHORIZATION, value);
        }
        builder.body(body.into()).unwrap()
    }

    fn bearer() -> String {
        format!("Bearer {TEST_TOKEN}")
    }

    fn rank_body(items: usize, plies: u32) -> String {
        let item = json!({
            "pips": opening_pips().to_vec(), "die1": 3, "die2": 1, "plies": plies,
        });
        serde_json::to_string(&vec![item; items]).unwrap()
    }

    /// A state whose single evaluation permit is held by a worker that blocks
    /// until the returned sender is dropped. Any request that reaches the
    /// semaphore now waits the full queue wait and answers 429.
    async fn saturated_state() -> (Arc<AppState>, mpsc::Sender<()>) {
        let state = state_with(Ok(()), 1);
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder = state.clone();
        tokio::spawn(async move {
            holder
                .capacity
                .run_blocking(move || {
                    let _ = release_rx.recv();
                })
                .await
                .unwrap();
        });
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        while state.capacity.available() != 0 && Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(state.capacity.available(), 0);
        (state, release_tx)
    }

    #[tokio::test]
    async fn missing_malformed_and_wrong_credentials_are_refused_before_the_body_is_parsed() {
        let (state, _hold) = saturated_state().await;
        let malformed_json = "{this is not json";
        for (name, auth) in [
            ("missing", None),
            ("empty", Some("")),
            ("not bearer", Some(&*format!("Basic {TEST_TOKEN}"))),
            ("bare token", Some(TEST_TOKEN)),
            (
                "wrong token",
                Some("Bearer ffffffffffffffffffffffffffffffff"),
            ),
            (
                "prefix of the token",
                Some(&*format!("Bearer {}", &TEST_TOKEN[..31])),
            ),
            (
                "token plus a character",
                Some(&*format!("Bearer {TEST_TOKEN}0")),
            ),
        ] {
            for path in ["/rank", "/evaluate", "/cube"] {
                let started = Instant::now();
                let response = app(state.clone())
                    .oneshot(request(Method::POST, path, auth, malformed_json))
                    .await
                    .unwrap();
                let elapsed = started.elapsed();
                assert_eq!(
                    response.status(),
                    StatusCode::UNAUTHORIZED,
                    "{name} on {path}"
                );
                assert_eq!(
                    response
                        .headers()
                        .get(WWW_AUTHENTICATE)
                        .map(|v| v.to_str().unwrap()),
                    Some("Bearer realm=\"bg-engine\""),
                    "{name} on {path}"
                );
                let text = text_of(response).await;
                assert_eq!(text, r#"{"error":"unauthorized"}"#, "{name} on {path}");
                assert!(!text.contains(TEST_TOKEN));
                // Prompt: neither a JSON parse error (400) nor the queue wait (429) was reached.
                assert!(
                    elapsed < capacity::QUEUE_WAIT,
                    "{name} on {path} took {elapsed:?}: the refusal must not touch capacity"
                );
            }
        }
    }

    #[tokio::test]
    async fn health_stays_public() {
        let state = state_with(Ok(()), 1);
        let response = app(state.clone())
            .oneshot(request(Method::GET, "/health", None, Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_of(response).await;
        assert_eq!(body["ok"], Value::Bool(true));
        assert_eq!(
            body["engineId"],
            Value::String(state.contract.engine_id.clone())
        );

        // A wrong token on /health is simply ignored: there is nothing to protect.
        let response = app(state)
            .oneshot(request(
                Method::GET,
                "/health",
                Some("Bearer nonsense"),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn an_oversized_body_is_413_with_the_right_token_and_401_without_it() {
        let (state, _hold) = saturated_state().await;
        // One item padded past 16 KiB with an unknown field serde would have
        // ignored -- the limit acts on bytes, before deserialisation.
        let padded = format!(
            r#"[{{"pips":{},"die1":3,"die2":1,"plies":1,"padding":"{}"}}]"#,
            serde_json::to_string(&opening_pips().to_vec()).unwrap(),
            "x".repeat(MAX_BODY_BYTES)
        );
        assert!(padded.len() > MAX_BODY_BYTES);

        let response = app(state.clone())
            .oneshot(request(
                Method::POST,
                "/rank",
                Some(&bearer()),
                padded.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let response = app(state.clone())
            .oneshot(request(
                Method::POST,
                "/rank",
                Some("Bearer wrong-token-wrong-token"),
                padded,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // Just under the limit, the same request shape is read and evaluated
        // -- which here means it reaches the semaphore and is refused with 429
        // (the permit is held): the body was accepted.
        let under = format!(
            r#"[{{"pips":{},"die1":3,"die2":1,"plies":1,"padding":"{}"}}]"#,
            serde_json::to_string(&opening_pips().to_vec()).unwrap(),
            "x".repeat(MAX_BODY_BYTES - 200)
        );
        assert!(under.len() <= MAX_BODY_BYTES);
        let response = app(state)
            .oneshot(request(Method::POST, "/rank", Some(&bearer()), under))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn an_empty_batch_a_batch_of_9_and_an_unimplemented_depth_are_400_before_any_evaluation()
    {
        let (state, _hold) = saturated_state().await;
        let cases: Vec<(&str, &str, String, &str)> = vec![
            ("/rank", "empty batch", "[]".to_string(), "empty"),
            (
                "/rank",
                "batch of 9",
                rank_body(9, 1),
                "more than the maximum of 8",
            ),
            (
                "/rank",
                "plies 3",
                rank_body(1, 3),
                "plies must be 0, 1 or 2",
            ),
            (
                "/rank",
                "plies 7 in the 8th item",
                {
                    let mut items: Vec<Value> = serde_json::from_str(&rank_body(8, 1)).unwrap();
                    items[7]["plies"] = json!(7);
                    serde_json::to_string(&items).unwrap()
                },
                "plies must be 0, 1 or 2",
            ),
            ("/evaluate", "empty batch", "[]".to_string(), "empty"),
            (
                "/evaluate",
                "batch of 9",
                serde_json::to_string(&vec![json!({"pips": opening_pips().to_vec()}); 9]).unwrap(),
                "more than the maximum of 8",
            ),
            ("/cube", "empty batch", "[]".to_string(), "empty"),
            (
                "/cube",
                "batch of 9",
                serde_json::to_string(&vec![json!({"pips": opening_pips().to_vec()}); 9]).unwrap(),
                "more than the maximum of 8",
            ),
        ];
        for (path, name, body, expected) in cases {
            let started = Instant::now();
            let response = app(state.clone())
                .oneshot(request(Method::POST, path, Some(&bearer()), body))
                .await
                .unwrap();
            let elapsed = started.elapsed();
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{name} on {path}"
            );
            let body: ErrorBody = serde_json::from_value(body_of(response).await).unwrap();
            assert!(
                body.error.contains(expected),
                "{name} on {path}: {}",
                body.error
            );
            assert!(
                elapsed < capacity::QUEUE_WAIT,
                "{name} on {path} took {elapsed:?}: the refusal must come before the wait for capacity"
            );
        }
    }

    #[tokio::test]
    async fn a_body_that_is_not_the_request_type_is_a_client_error_once_the_credentials_are_right()
    {
        let (state, _hold) = saturated_state().await;
        for body in [
            "{this is not json",
            r#"{"pips": "not a list"}"#,
            r#"[{"die1": 3}]"#,
        ] {
            let response = app(state.clone())
                .oneshot(request(Method::POST, "/rank", Some(&bearer()), body))
                .await
                .unwrap();
            assert!(
                response.status().is_client_error()
                    && response.status() != StatusCode::UNAUTHORIZED,
                "{body}: {}",
                response.status()
            );
        }
    }

    #[tokio::test]
    async fn saturation_is_429_with_retry_after_and_no_results() {
        let (state, hold) = saturated_state().await;
        let started = Instant::now();
        let response = app(state.clone())
            .oneshot(request(
                Method::POST,
                "/rank",
                Some(&bearer()),
                rank_body(1, 1),
            ))
            .await
            .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(RETRY_AFTER)
                .map(|v| v.to_str().unwrap()),
            Some("1")
        );
        let body = body_of(response).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("capacity saturated")
        );
        assert!(body.get("results").is_none());
        assert!(elapsed >= capacity::QUEUE_WAIT, "{elapsed:?}");
        assert!(elapsed < std::time::Duration::from_secs(1), "{elapsed:?}");

        // /health answers at once while the evaluation slot is busy.
        let started = Instant::now();
        let response = app(state.clone())
            .oneshot(request(Method::GET, "/health", None, Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(started.elapsed() < capacity::QUEUE_WAIT);

        // Release the worker: the next request is served.
        drop(hold);
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        while state.capacity.available() == 0 && Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let response = app(state)
            .oneshot(request(
                Method::POST,
                "/rank",
                Some(&bearer()),
                rank_body(1, 1),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn authorized_replies_carry_the_unchanged_gp476_identity_and_metadata() {
        let state = state_with(Ok(()), 1);
        let health = body_of(
            app(state.clone())
                .oneshot(request(Method::GET, "/health", None, Body::empty()))
                .await
                .unwrap(),
        )
        .await;
        let cube_body =
            serde_json::to_string(&vec![json!({"pips": opening_pips().to_vec()})]).unwrap();
        for (path, body) in [
            ("/rank", rank_body(1, 1)),
            ("/evaluate", cube_body.clone()),
            ("/cube", cube_body),
        ] {
            let response = app(state.clone())
                .oneshot(request(Method::POST, path, Some(&bearer()), body))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let reply = body_of(response).await;
            assert_eq!(reply["apiVersion"], json!(1), "{path}");
            assert_eq!(
                reply["capabilities"],
                json!(contract::CAPABILITIES.to_vec()),
                "{path}"
            );
            assert_eq!(reply["engineId"], health["engineId"], "{path}");
            assert_eq!(reply["engineId"], json!(state.contract.engine_id), "{path}");
            assert_eq!(reply["results"].as_array().map(Vec::len), Some(1), "{path}");
            assert!(reply["evalMs"].as_f64().is_some(), "{path}");
        }
        assert_eq!(state.capacity.available(), 1, "every permit came back");
    }

    #[test]
    fn plies_0_and_1_are_1_ply_2_is_2_ply_and_nothing_else_is_a_depth() {
        assert_eq!(Depth::for_plies(0), Ok(Depth::OnePly));
        assert_eq!(Depth::for_plies(1), Ok(Depth::OnePly));
        assert_eq!(Depth::for_plies(2), Ok(Depth::TwoPly));
        for plies in [3, 4, 10, u32::MAX] {
            let err = Depth::for_plies(plies).unwrap_err();
            assert!(err.contains(&plies.to_string()), "{err}");
            assert!(err.contains("plies.1 and plies.2"), "{err}");
        }
    }

    #[tokio::test]
    async fn plies_0_and_1_produce_the_same_1_ply_ranking() {
        let state = state_with(Ok(()), 1);
        let mut rankings = Vec::new();
        for plies in [0, 1] {
            let response = app(state.clone())
                .oneshot(request(
                    Method::POST,
                    "/rank",
                    Some(&bearer()),
                    rank_body(1, plies),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            rankings.push(body_of(response).await["results"][0]["moves"].clone());
        }
        assert_eq!(rankings[0], rankings[1]);
        assert!(!rankings[0].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn plies_2_runs_the_2_ply_evaluator() {
        let state = state_with(Ok(()), 1);
        let mut best_equity = Vec::new();
        for plies in [1, 2] {
            let response = app(state.clone())
                .oneshot(request(
                    Method::POST,
                    "/rank",
                    Some(&bearer()),
                    rank_body(1, plies),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "plies {plies}");
            let body = body_of(response).await;
            assert_eq!(body["capabilities"], json!(contract::CAPABILITIES.to_vec()));
            best_equity.push(body["results"][0]["moves"][0]["equity"].as_f64().unwrap());
        }
        // A deeper search values the same best play differently: the 2-ply
        // reply came from the other evaluator, not from a re-labelled 1-ply.
        assert_ne!(best_equity[0], best_equity[1]);
    }
}

// ----------------------------------------------------------------- capacity
//
// GP-477, handler level: the evaluation handlers run their neural work
// through `Capacity::run_blocking`, so the bound, the orphaned-permit rule
// and the prompt refusal proven in capacity.rs apply to them. These tests
// show the handlers actually go through it, and that the async side stays
// responsive while the blocking side is busy.

#[cfg(test)]
mod capacity_route_tests {
    use super::test_support::*;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_rank_requests_never_run_more_evaluations_than_the_bound() {
        let bound = 2;
        let state = state_with(Ok(()), bound);
        let served = Arc::new(AtomicUsize::new(0));
        let refused = Arc::new(AtomicUsize::new(0));

        // Sample the permits out while the requests run: that is the number
        // of evaluations on the blocking pool at that instant.
        let peak = Arc::new(AtomicUsize::new(0));
        let sampling = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let sampler = {
            let state = state.clone();
            let peak = peak.clone();
            let sampling = sampling.clone();
            tokio::spawn(async move {
                while sampling.load(Ordering::SeqCst) {
                    peak.fetch_max(bound - state.capacity.available(), Ordering::SeqCst);
                    tokio::task::yield_now().await;
                }
            })
        };

        let mut tasks = Vec::new();
        for _ in 0..16 {
            let state = state.clone();
            let served = served.clone();
            let refused = refused.clone();
            tasks.push(tokio::spawn(async move {
                let response = rank(State(state.clone()), Json(rank_request()))
                    .await
                    .into_response();
                match response.status() {
                    StatusCode::OK => served.fetch_add(1, Ordering::SeqCst),
                    StatusCode::TOO_MANY_REQUESTS => refused.fetch_add(1, Ordering::SeqCst),
                    other => panic!("unexpected {other}"),
                };
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        sampling.store(false, Ordering::SeqCst);
        sampler.await.unwrap();

        assert!(served.load(Ordering::SeqCst) >= 1);
        assert_eq!(
            served.load(Ordering::SeqCst) + refused.load(Ordering::SeqCst),
            16
        );
        let peak = peak.load(Ordering::SeqCst);
        assert!(
            (1..=bound).contains(&peak),
            "peak concurrent evaluations {peak}, bound {bound}"
        );
        assert_eq!(state.capacity.available(), bound, "every permit came back");
    }

    #[tokio::test]
    async fn health_answers_while_an_evaluation_is_running_on_the_blocking_pool() {
        let state = state_with(Ok(()), 1);
        let busy = state.clone();
        let worker = tokio::spawn(async move {
            busy.capacity
                .run_blocking(|| std::thread::sleep(Duration::from_millis(300)))
                .await
                .unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while state.capacity.available() != 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(state.capacity.available(), 0);
        assert!(!worker.is_finished());

        let started = Instant::now();
        let (status, Json(report)) = health(State(state.clone())).await;
        assert_eq!(status, StatusCode::OK);
        assert!(report.ok);
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "/health must not queue behind evaluation: {:?}",
            started.elapsed()
        );
        assert!(
            !worker.is_finished(),
            "the evaluation was still running when /health answered"
        );
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn the_handler_refuses_promptly_with_429_when_the_only_slot_is_held() {
        let state = state_with(Ok(()), 1);
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let holder = state.clone();
        let worker = tokio::spawn(async move {
            holder
                .capacity
                .run_blocking(move || {
                    let _ = release_rx.recv();
                })
                .await
                .unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while state.capacity.available() != 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        for path in ["rank", "evaluate", "cube"] {
            let started = Instant::now();
            let response = match path {
                "rank" => rank(State(state.clone()), Json(rank_request()))
                    .await
                    .into_response(),
                "evaluate" => evaluate(State(state.clone()), Json(position_request()))
                    .await
                    .into_response(),
                _ => cube(State(state.clone()), Json(position_request()))
                    .await
                    .into_response(),
            };
            let elapsed = started.elapsed();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS, "{path}");
            assert_eq!(
                response
                    .headers()
                    .get(RETRY_AFTER)
                    .map(|v| v.to_str().unwrap()),
                Some("1"),
                "{path}"
            );
            assert!(
                elapsed >= capacity::QUEUE_WAIT && elapsed < Duration::from_secs(1),
                "{path}: {elapsed:?}"
            );
        }

        release_tx.send(()).unwrap();
        worker.await.unwrap();
        assert_eq!(state.capacity.available(), 1);
    }
}
