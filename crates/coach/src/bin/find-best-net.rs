use clap::{Parser, ValueEnum};
use coach::duel::Duel;
use coach::unwrap::UnwrapHelper;
use engine::composite::CompositeEvaluator;
use engine::dice_gen::FastrandDice;
use engine::probabilities::{Probabilities, ResultCounter};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{Write, stdout};

/// The whole file was vibecoded, it might be deleted soon. Maybe also extended, who knows.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version)]
#[command(about = "Find the best neural net in the folder `training-data` using multi-armed bandit", long_about = None)]
struct Args {
    #[arg(long)]
    phase: Phase,
    /// Number of games per evaluation batch.
    #[arg(long, default_value_t = 1000)]
    batch_size: u32,
    /// Exploration constant for UCB1. Lower values (0.1-0.3) favor exploitation, higher favor exploration.
    #[arg(long, default_value_t = 0.4)]
    exploration: f64,
}

#[derive(clap::ValueEnum, Clone, Serialize, Debug)]
enum Phase {
    Contact,
    Race,
}

/// UCB1 (Upper Confidence Bound) algorithm with tunable exploration constant.
/// Better than Thompson Sampling when reward differences are very small (e.g., equity around ±0.01).
struct UCB1 {
    /// Sum of rewards for each arm
    reward_sums: Vec<f64>,
    /// Number of times each arm has been pulled
    pull_counts: Vec<u32>,
    /// Total number of pulls across all arms
    total_pulls: u32,
    /// Exploration constant (c). Lower values favor exploitation, higher favor exploration.
    exploration_constant: f64,
}

impl UCB1 {
    fn new(num_arms: usize, exploration_constant: f64) -> Self {
        Self {
            reward_sums: vec![0.0; num_arms],
            pull_counts: vec![0; num_arms],
            total_pulls: 0,
            exploration_constant,
        }
    }

    /// Select an arm using UCB1 formula with configurable exploration constant
    /// Formula: avg_reward + c * sqrt(2 * ln(total_pulls) / arm_pulls)
    fn select_arm(&self) -> usize {
        // First, make sure each arm is pulled at least once
        for (idx, &count) in self.pull_counts.iter().enumerate() {
            if count == 0 {
                return idx;
            }
        }

        // Calculate UCB value for each arm
        let ucb_values: Vec<f64> = (0..self.reward_sums.len())
            .map(|i| {
                let avg_reward = self.reward_sums[i] / self.pull_counts[i] as f64;
                let exploration_bonus = self.exploration_constant
                    * (2.0 * (self.total_pulls as f64).ln() / self.pull_counts[i] as f64).sqrt();
                avg_reward + exploration_bonus
            })
            .collect();

        ucb_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Update the arm's statistics based on observed reward
    fn update(&mut self, arm: usize, reward: f64, counts: u32) {
        self.reward_sums[arm] += reward * counts as f64;
        self.pull_counts[arm] += counts;
        self.total_pulls += counts;
    }
}

/// Find the best neural net in the folder `training-data` using a multi-armed bandit approach.
/// This treats each ONNX file as an "arm" and uses UCB1 (Upper Confidence Bound) to allocate
/// more duels to promising candidates while exploring less promising ones efficiently.
/// Runs until interrupted with Ctrl+C.
fn main() {
    let args = Args::parse();

    let folder_name = "training-data";
    println!("Start finding best neural net in {folder_name}");
    println!("Batch size: {} games", args.batch_size);
    println!("Exploration constant: {}", args.exploration);
    println!("Press Ctrl+C to stop and show results.\n");

    // Set up Ctrl+C handler
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Find all neural nets in the folder
    let net_files: Vec<String> = fs::read_dir(folder_name)
        .unwrap()
        .map(|x| x.unwrap().file_name().into_string().unwrap())
        .filter(|x| x.starts_with(args.phase.to_possible_value().unwrap().get_name()))
        .filter(|x| x.ends_with(".onnx"))
        .collect();

    if net_files.is_empty() {
        eprintln!(
            "No neural nets found in {folder_name} for phase {:?}",
            args.phase
        );
        return;
    }

    println!("Found {} neural nets to evaluate", net_files.len());
    if !net_files.is_empty() {
        let mut sorted_files = net_files.clone();
        sorted_files.sort();
        println!("  First: {}", sorted_files.first().unwrap());
        if sorted_files.len() > 1 {
            println!("  Last:  {}", sorted_files.last().unwrap());
        }
    }
    println!();

    // Load all neural nets and create Duel structs once (outside all loops)
    println!("Loading neural nets...");
    let duels: Vec<Duel<CompositeEvaluator, CompositeEvaluator>> = net_files
        .iter()
        .map(|net_file| {
            let path_string = folder_name.to_string() + "/" + net_file;

            let contender = match args.phase {
                Phase::Contact => {
                    CompositeEvaluator::from_file_paths(&path_string, "neural-nets/race.onnx")
                }
                Phase::Race => {
                    CompositeEvaluator::from_file_paths("neural-nets/contact.onnx", &path_string)
                }
            }
            .unwrap_or_exit_with_message();

            let current = CompositeEvaluator::from_file_paths(
                "neural-nets/contact.onnx",
                "neural-nets/race.onnx",
            )
            .unwrap_or_exit_with_message();

            Duel::new(contender, current)
        })
        .collect();
    println!("All neural nets loaded.\n");

    // Initialize UCB1 bandit algorithm with exploration constant
    let mut bandit = UCB1::new(net_files.len(), args.exploration);

    // Track statistics for each arm
    let mut total_games: HashMap<usize, u32> = HashMap::new();
    let mut equity_sum: HashMap<usize, f64> = HashMap::new();
    let mut equity_count: HashMap<usize, u32> = HashMap::new();

    let mut games_played = 0;

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Select which neural net to evaluate next using UCB1
        let arm_idx = bandit.select_arm();
        let net_file = &net_files[arm_idx];

        // Get the pre-created Duel for this arm
        let duel = &duels[arm_idx];

        // Evaluate this neural net by running the duel
        let mut dice_gen = FastrandDice::new();
        let seeds: Vec<u64> = (0..args.batch_size / 2).map(|_| dice_gen.seed()).collect();
        let counter = seeds
            .into_par_iter()
            .map(|seed| duel.duel(&mut FastrandDice::with_seed(seed)))
            .reduce(ResultCounter::default, |a, b| a.combine(&b));
        let probabilities = Probabilities::from(&counter);
        let equity = probabilities.equity() as f64;

        // Update games played counter
        games_played += args.batch_size;
        *total_games.entry(arm_idx).or_insert(0) += args.batch_size;
        *equity_sum.entry(arm_idx).or_insert(0.0) += equity;
        *equity_count.entry(arm_idx).or_insert(0) += 1;

        // Convert equity to reward in [0, 1] range for UCB1
        // Equity typically ranges from -1 to 1, so we normalize to [0, 1]
        let reward = (equity + 1.0) / 2.0;

        // Update the bandit with this observation
        bandit.update(arm_idx, reward, args.batch_size);

        // Get all average equities for ranking and median calculation
        let mut all_equities: Vec<(usize, f64)> = equity_sum
            .iter()
            .map(|(idx, sum)| {
                let count = equity_count.get(idx).copied().unwrap_or(1);
                let avg_equity = sum / count as f64;
                (*idx, avg_equity)
            })
            .collect();
        all_equities.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Calculate median equity
        let median_equity = if all_equities.is_empty() {
            0.0
        } else {
            let mid = all_equities.len() / 2;
            if all_equities.len().is_multiple_of(2) && all_equities.len() > 1 {
                // Even number of elements: average of two middle values
                (all_equities[mid - 1].1 + all_equities[mid].1) / 2.0
            } else {
                // Odd number of elements: middle value
                all_equities[mid].1
            }
        };

        // Get top 3 nets
        let top_nets: Vec<(usize, f64)> = all_equities.iter().take(3).copied().collect();

        // Print progress (overwrite line)
        // Add space before positive equity to prevent shifting when sign changes
        let equity_str = if equity >= 0.0 {
            format!(" {:.4}", equity)
        } else {
            format!("{:.4}", equity)
        };
        print!(
            "\r[{:>4.0}k games] Evaluated {}: equity = {} | Top 3: ",
            games_played as f64 / 1_000.0,
            net_file.strip_suffix(".onnx").unwrap(),
            equity_str,
        );

        for (i, (idx, eq)) in top_nets.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            let games = total_games.get(idx).copied().unwrap_or(0);
            print!(
                "{}: {:.4} ({} games)",
                net_files[*idx].strip_suffix(".onnx").unwrap(),
                eq,
                games
            );
        }

        print!(" | Median: {:.4}", median_equity);

        // Pad with spaces to clear previous output
        print!("  ");
        stdout().flush().unwrap();
    }

    println!("\n========================================");
    println!("Final Results:");
    println!("========================================\n");

    // Calculate average equity for each net
    let mut results: Vec<(usize, &String, u32, f64)> = net_files
        .iter()
        .enumerate()
        .map(|(idx, net_file)| {
            let games = total_games.get(&idx).copied().unwrap_or(0);
            let avg_equity = if let Some(&sum) = equity_sum.get(&idx) {
                let count = equity_count.get(&idx).copied().unwrap_or(1);
                sum / count as f64
            } else {
                0.0
            };
            (idx, net_file, games, avg_equity)
        })
        .collect();

    // Sort by equity (best first)
    results.sort_by(|(_, _, _, eq1), (_, _, _, eq2)| eq2.partial_cmp(eq1).unwrap());

    // Print detailed stats for each net (sorted by equity)
    for (_, net_file, games, avg_equity) in &results {
        if *games > 0 {
            println!(
                "{}: After {} games the equity is {:7.4}.",
                net_file.strip_suffix(".onnx").unwrap(),
                games,
                avg_equity,
            );
        }
    }
}
