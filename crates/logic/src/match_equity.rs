//! Cubeless match-winning-chance scoring for checker ranking (GP-493).
//!
//! Kazaross-XG2 values are copied verbatim from GNU Backgammon
//! `met/Kazaross-XG2.xml` at 0143cad3c8f7826f4cd33c6627c9c5485bf1b0ae.
//! Table by Neil Kazaross (2011); copying/distribution permitted when this notice is preserved.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrawfordState {
    PreCrawford,
    Crawford,
    PostCrawford,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeOwner {
    Us,
    Them,
    Centred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchContext {
    pub match_length: u32,
    pub score_us: u32,
    pub score_them: u32,
    pub points_away_us: u32,
    pub points_away_them: u32,
    pub cube_enabled: bool,
    pub cube_value: u32,
    pub cube_owner: CubeOwner,
    pub crawford_state: CrawfordState,
}
const PRE: [[f32; 25]; 25] = [
    [
        0.500000, 0.677360, 0.750760, 0.814360, 0.841790, 0.887310, 0.907240, 0.932500, 0.944020,
        0.959275, 0.966442, 0.975534, 0.979845, 0.985273, 0.987893, 0.991140, 0.992730, 0.994670,
        0.995630, 0.996790, 0.997370, 0.998070, 0.998420, 0.998840, 0.999050,
    ],
    [
        0.322640, 0.500000, 0.599470, 0.668700, 0.743590, 0.799400, 0.842250, 0.875390, 0.901970,
        0.923034, 0.939311, 0.952470, 0.962495, 0.970701, 0.976887, 0.981960, 0.985800, 0.988930,
        0.991290, 0.993220, 0.994660, 0.995850, 0.996750, 0.997460, 0.998020,
    ],
    [
        0.249240, 0.400530, 0.500000, 0.571500, 0.647950, 0.711230, 0.762090, 0.804680, 0.840170,
        0.870638, 0.894417, 0.914831, 0.930702, 0.944426, 0.954931, 0.963990, 0.970930, 0.976870,
        0.981390, 0.985220, 0.988140, 0.990620, 0.992480, 0.994070, 0.995270,
    ],
    [
        0.185640, 0.331300, 0.428500, 0.500000, 0.577320, 0.642850, 0.699240, 0.745770, 0.787990,
        0.824059, 0.853955, 0.879141, 0.900233, 0.918040, 0.932657, 0.944950, 0.954990, 0.963410,
        0.970210, 0.975890, 0.980440, 0.984220, 0.987260, 0.989750, 0.991740,
    ],
    [
        0.158210, 0.256410, 0.352050, 0.422680, 0.500000, 0.566350, 0.626380, 0.677860, 0.725400,
        0.767055, 0.802732, 0.833654, 0.859934, 0.882866, 0.902013, 0.918470, 0.932230, 0.943970,
        0.953670, 0.961890, 0.968640, 0.974320, 0.978960, 0.982830, 0.986000,
    ],
    [
        0.112690, 0.200600, 0.288770, 0.357150, 0.433650, 0.500000, 0.562610, 0.616360, 0.667870,
        0.713057, 0.753427, 0.788634, 0.819569, 0.846648, 0.869999, 0.890210, 0.907560, 0.922460,
        0.935080, 0.945830, 0.954880, 0.962540, 0.968940, 0.974320, 0.978790,
    ],
    [
        0.092760, 0.157750, 0.237910, 0.300760, 0.373620, 0.437390, 0.500000, 0.554800, 0.608540,
        0.656283, 0.700209, 0.739054, 0.774121, 0.805203, 0.832566, 0.856590, 0.877610, 0.895910,
        0.911710, 0.925350, 0.937020, 0.947030, 0.955530, 0.962760, 0.968870,
    ],
    [
        0.067500, 0.124610, 0.195320, 0.254230, 0.322140, 0.383640, 0.445200, 0.500000, 0.554420,
        0.603718, 0.649899, 0.691356, 0.729447, 0.763593, 0.794397, 0.821580, 0.845780, 0.867140,
        0.885890, 0.902300, 0.916580, 0.928980, 0.939680, 0.948910, 0.956820,
    ],
    [
        0.055980, 0.098030, 0.159830, 0.212010, 0.274600, 0.332130, 0.391460, 0.445580, 0.500000,
        0.550196, 0.597926, 0.641481, 0.682119, 0.718927, 0.752814, 0.783010, 0.810370, 0.834830,
        0.856620, 0.875910, 0.892940, 0.907910, 0.920980, 0.932400, 0.942300,
    ],
    [
        0.040725, 0.076966, 0.129362, 0.175941, 0.232945, 0.286943, 0.343717, 0.396282, 0.449804,
        0.500000, 0.548547, 0.593459, 0.635880, 0.674830, 0.711113, 0.743710, 0.773750, 0.800930,
        0.825430, 0.847410, 0.867030, 0.884480, 0.899910, 0.913530, 0.925500,
    ],
    [
        0.033558, 0.060689, 0.105583, 0.146045, 0.197268, 0.246573, 0.299791, 0.350101, 0.402074,
        0.451453, 0.500000, 0.545552, 0.589242, 0.629736, 0.667927, 0.703030, 0.735300, 0.764940,
        0.791980, 0.816480, 0.838620, 0.858490, 0.876290, 0.892140, 0.906220,
    ],
    [
        0.024466, 0.047530, 0.085169, 0.120859, 0.166346, 0.211366, 0.260946, 0.308644, 0.358519,
        0.406541, 0.454448, 0.500000, 0.544068, 0.585701, 0.625259, 0.661780, 0.696100, 0.727780,
        0.757030, 0.783810, 0.808260, 0.830440, 0.850510, 0.868560, 0.884760,
    ],
    [
        0.020155, 0.037505, 0.069298, 0.099767, 0.140066, 0.180431, 0.225879, 0.270553, 0.317881,
        0.364120, 0.410758, 0.455932, 0.500000, 0.541943, 0.582545, 0.620360, 0.656190, 0.689660,
        0.720810, 0.749630, 0.776190, 0.800540, 0.822760, 0.842950, 0.861230,
    ],
    [
        0.014727, 0.029299, 0.055574, 0.081960, 0.117134, 0.153352, 0.194797, 0.236407, 0.281073,
        0.325170, 0.370264, 0.414299, 0.458057, 0.500000, 0.540750, 0.579420, 0.616340, 0.651170,
        0.683910, 0.714480, 0.742900, 0.769170, 0.793390, 0.815590, 0.835860,
    ],
    [
        0.012107, 0.023113, 0.045069, 0.067343, 0.097987, 0.130001, 0.167434, 0.205603, 0.247186,
        0.288887, 0.332073, 0.374741, 0.417455, 0.459250, 0.500000, 0.539160, 0.576790, 0.612610,
        0.646590, 0.678590, 0.708620, 0.736640, 0.762650, 0.786690, 0.808830,
    ],
    [
        0.008860, 0.018040, 0.036010, 0.055050, 0.081530, 0.109790, 0.143410, 0.178420, 0.216990,
        0.256290, 0.296970, 0.338220, 0.379640, 0.420580, 0.460840, 0.500000, 0.537960, 0.574410,
        0.609290, 0.642410, 0.673760, 0.703230, 0.730840, 0.756570, 0.780460,
    ],
    [
        0.007270, 0.014200, 0.029070, 0.045010, 0.067770, 0.092440, 0.122390, 0.154220, 0.189630,
        0.226250, 0.264700, 0.303900, 0.343810, 0.383660, 0.423210, 0.462040, 0.500000, 0.536760,
        0.572220, 0.606180, 0.638560, 0.669250, 0.698220, 0.725420, 0.750870,
    ],
    [
        0.005330, 0.011070, 0.023130, 0.036590, 0.056030, 0.077540, 0.104090, 0.132860, 0.165170,
        0.199070, 0.235060, 0.272220, 0.310340, 0.348830, 0.387390, 0.425590, 0.463240, 0.500000,
        0.535740, 0.570230, 0.603360, 0.635010, 0.665100, 0.693560, 0.720380,
    ],
    [
        0.004370, 0.008710, 0.018610, 0.029790, 0.046330, 0.064920, 0.088290, 0.114110, 0.143380,
        0.174570, 0.208020, 0.242970, 0.279190, 0.316090, 0.353410, 0.390710, 0.427780, 0.464260,
        0.500000, 0.534750, 0.568380, 0.600730, 0.631710, 0.661220, 0.689210,
    ],
    [
        0.003210, 0.006780, 0.014780, 0.024110, 0.038110, 0.054170, 0.074650, 0.097700, 0.124090,
        0.152590, 0.183520, 0.216190, 0.250370, 0.285520, 0.321410, 0.357590, 0.393820, 0.429770,
        0.465250, 0.500000, 0.533870, 0.566670, 0.598300, 0.628640, 0.657600,
    ],
    [
        0.002630, 0.005340, 0.011860, 0.019560, 0.031360, 0.045120, 0.062980, 0.083420, 0.107060,
        0.132970, 0.161380, 0.191740, 0.223810, 0.257100, 0.291380, 0.326240, 0.361440, 0.396640,
        0.431620, 0.466130, 0.500000, 0.533030, 0.565080, 0.596030, 0.625760,
    ],
    [
        0.001930, 0.004150, 0.009380, 0.015780, 0.025680, 0.037460, 0.052970, 0.071020, 0.092090,
        0.115520, 0.141510, 0.169560, 0.199460, 0.230830, 0.263360, 0.296770, 0.330750, 0.364990,
        0.399270, 0.433330, 0.466970, 0.500000, 0.532260, 0.563600, 0.593910,
    ],
    [
        0.001580, 0.003250, 0.007520, 0.012740, 0.021040, 0.031060, 0.044470, 0.060320, 0.079020,
        0.100090, 0.123710, 0.149490, 0.177240, 0.206610, 0.237350, 0.269160, 0.301780, 0.334900,
        0.368290, 0.401700, 0.434920, 0.467740, 0.500000, 0.531530, 0.562210,
    ],
    [
        0.001160, 0.002540, 0.005930, 0.010250, 0.017170, 0.025680, 0.037240, 0.051090, 0.067600,
        0.086470, 0.107860, 0.131440, 0.157050, 0.184410, 0.213310, 0.243430, 0.274580, 0.306440,
        0.338780, 0.371360, 0.403970, 0.436400, 0.468470, 0.500000, 0.530860,
    ],
    [
        0.000950, 0.001980, 0.004730, 0.008260, 0.014000, 0.021210, 0.031130, 0.043180, 0.057700,
        0.074500, 0.093780, 0.115240, 0.138770, 0.164140, 0.191170, 0.219540, 0.249130, 0.279620,
        0.310790, 0.342400, 0.374240, 0.406090, 0.437790, 0.469140, 0.500000,
    ],
];

const POST: [f32; 25] = [
    0.500000, 0.488030, 0.322640, 0.310020, 0.190120, 0.180720, 0.115590, 0.109060, 0.069530,
    0.065161, 0.042069, 0.039060, 0.025371, 0.023428, 0.015304, 0.014050, 0.009240, 0.008420,
    0.005560, 0.005050, 0.003360, 0.003030, 0.002030, 0.001820, 0.001230,
];

pub fn validate(ctx: &MatchContext) -> Result<(), String> {
    if !(1..=25).contains(&ctx.match_length) {
        return Err("matchLength must be in 1..25".into());
    }
    if ctx.score_us >= ctx.match_length || ctx.score_them >= ctx.match_length {
        return Err("score must be below matchLength".into());
    }
    if ctx.points_away_us != ctx.match_length - ctx.score_us
        || ctx.points_away_them != ctx.match_length - ctx.score_them
    {
        return Err("points-away fields do not match matchLength and score".into());
    }
    if ctx.cube_value == 0 || !ctx.cube_value.is_power_of_two() {
        return Err("cubeValue must be a positive power of two".into());
    }
    if ctx.cube_value == 1 && ctx.cube_owner != CubeOwner::Centred {
        return Err("a 1-cube must be centred".into());
    }
    if ctx.cube_value > 1 && ctx.cube_owner == CubeOwner::Centred {
        return Err("a turned cube must have an owner".into());
    }
    if !ctx.cube_enabled && (ctx.cube_value != 1 || ctx.cube_owner != CubeOwner::Centred) {
        return Err("a disabled cube must be centred at 1".into());
    }
    let us_one_away = ctx.points_away_us == 1;
    let them_one_away = ctx.points_away_them == 1;
    match ctx.crawford_state {
        CrawfordState::PreCrawford => {
            let one_point_dmp = ctx.match_length == 1 && us_one_away && them_one_away;
            if (us_one_away || them_one_away) && !one_point_dmp && ctx.cube_enabled {
                return Err("pre-Crawford one-away state requires a cube-disabled match".into());
            }
        }
        CrawfordState::Crawford => {
            if us_one_away == them_one_away {
                return Err("Crawford requires exactly one player one-away".into());
            }
            if ctx.cube_enabled || ctx.cube_value != 1 || ctx.cube_owner != CubeOwner::Centred {
                return Err("the Crawford game requires a disabled centred 1-cube".into());
            }
        }
        CrawfordState::PostCrawford => {
            if !us_one_away && !them_one_away {
                return Err("post-Crawford requires at least one player one-away".into());
            }
        }
    }
    Ok(())
}

pub fn supported(ctx: &MatchContext) -> bool {
    let global_no_cube_one_away = !ctx.cube_enabled
        && ctx.crawford_state == CrawfordState::PreCrawford
        && ctx.match_length > 1
        && (ctx.points_away_us == 1 || ctx.points_away_them == 1);
    !global_no_cube_one_away
        && (!ctx.cube_enabled
            || (ctx.points_away_us == 1 && ctx.points_away_them == 1)
            || (ctx.cube_value >= ctx.points_away_us && ctx.cube_value >= ctx.points_away_them))
}

fn met(ours: i32, theirs: i32, phase: CrawfordState) -> f32 {
    if ours <= 0 {
        return 1.0;
    }
    if theirs <= 0 {
        return 0.0;
    }
    let o = ours as usize;
    let t = theirs as usize;
    if o == 1 && t == 1 {
        return 0.5;
    }
    if phase == CrawfordState::PostCrawford && (o == 1 || t == 1) {
        return if o == 1 {
            1.0 - POST[t - 1]
        } else {
            POST[o - 1]
        };
    }
    PRE[o - 1][t - 1]
}

/// Expected MWC for cumulative wildbg probabilities.
pub fn mwc(prob: [f32; 6], ctx: &MatchContext) -> f32 {
    let [wn, wg, wbg, ln, lg, lbg] = prob;
    let stake = ctx.cube_value as i32;
    let next_phase = match ctx.crawford_state {
        CrawfordState::PreCrawford => CrawfordState::Crawford,
        CrawfordState::Crawford | CrawfordState::PostCrawford => CrawfordState::PostCrawford,
    };
    let after = |ours: i32, theirs: i32| {
        if ours <= 0 {
            return 1.0;
        }
        if theirs <= 0 {
            return 0.0;
        }
        let phase = if ours == 1 || theirs == 1 {
            next_phase
        } else {
            CrawfordState::PreCrawford
        };
        met(ours, theirs, phase)
    };
    let ours = ctx.points_away_us as i32;
    let theirs = ctx.points_away_them as i32;
    wn * after(ours - stake, theirs)
        + wg * after(ours - 2 * stake, theirs)
        + wbg * after(ours - 3 * stake, theirs)
        + ln * after(ours, theirs - stake)
        + lg * after(ours, theirs - 2 * stake)
        + lbg * after(ours, theirs - 3 * stake)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ctx(us: u32, them: u32) -> MatchContext {
        MatchContext {
            match_length: 5,
            score_us: 5 - us,
            score_them: 5 - them,
            points_away_us: us,
            points_away_them: them,
            cube_enabled: false,
            cube_value: 1,
            cube_owner: CubeOwner::Centred,
            crawford_state: CrawfordState::PreCrawford,
        }
    }
    #[test]
    fn gammon_value_changes_with_score() {
        // Two fixed neural-style exclusive outcome vectors. Candidate A is
        // preferred at 2-away/4-away; candidate B at the mirrored score.
        // One board/candidate set can therefore rank differently by score.
        let candidate_a = [0.4337, 0.1668, 0.0217, 0.3025, 0.0730, 0.0023];
        let candidate_b = [0.3186, 0.0409, 0.2902, 0.0471, 0.0033, 0.2999];
        assert!(mwc(candidate_a, &ctx(2, 4)) > mwc(candidate_b, &ctx(2, 4)));
        assert!(mwc(candidate_a, &ctx(4, 2)) < mwc(candidate_b, &ctx(4, 2)));
    }
    #[test]
    fn live_cube_is_explicitly_unsupported() {
        let mut c = ctx(4, 4);
        c.cube_enabled = true;
        assert!(!supported(&c));
    }

    #[test]
    fn impossible_crawford_phases_and_cube_states_are_rejected() {
        let mut c = ctx(4, 4);
        c.crawford_state = CrawfordState::Crawford;
        assert!(validate(&c).unwrap_err().contains("exactly one"));

        c = ctx(1, 4);
        c.crawford_state = CrawfordState::Crawford;
        c.cube_value = 4;
        c.cube_owner = CubeOwner::Us;
        assert!(validate(&c).unwrap_err().contains("disabled cube"));

        c = ctx(4, 4);
        c.crawford_state = CrawfordState::PostCrawford;
        assert!(validate(&c).unwrap_err().contains("at least one"));

        c = ctx(1, 4);
        c.cube_enabled = false;
        assert!(validate(&c).is_ok());
        assert!(!supported(&c));

        c.cube_enabled = true;
        assert!(validate(&c).unwrap_err().contains("cube-disabled"));

        c = ctx(1, 4);
        c.crawford_state = CrawfordState::PostCrawford;
        c.cube_enabled = true;
        assert!(validate(&c).is_ok());
        assert!(!supported(&c));
    }
}
