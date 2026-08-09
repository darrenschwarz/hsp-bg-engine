The following doc was created by an AI agent and may be deleted.

# find-best-net

Continuous UCB1 evaluation with tunable exploration for finding the best neural network in the `training-data` folder.

## Overview

This binary uses UCB1 (Upper Confidence Bound) with a configurable exploration constant. It runs continuously until you press Ctrl+C, adaptively allocating more evaluations to promising candidates while still exploring all options.

## Algorithm

**UCB1 with exploration constant** works by:
1. Calculating an "upper confidence bound" for each neural net: `avg_reward + c * sqrt(2 * ln(total_pulls) / arm_pulls)`
2. The constant `c` controls the exploration/exploitation trade-off
3. Selecting the neural net with the highest UCB value
4. Over time, better-performing nets get evaluated more frequently

**Key feature:** The exploration constant `c` allows you to tune the algorithm's behavior:
- Lower `c` → more exploitation (focus on best performers)
- Higher `c` → more exploration (more thorough search)

This is particularly effective for small equity differences (±0.01 range) because you can reduce exploration to quickly identify the best performers.

## Usage

```bash
# Run with default exploration constant (0.3 - favors exploitation)
cargo run -p coach --bin find-best-net --release -- --phase race

# Use more exploitation (faster convergence to best)
cargo run -p coach --bin find-best-net --release -- --phase race --exploration 0.1

# Use standard UCB1 behavior (balanced)
cargo run -p coach --bin find-best-net --release -- --phase race --exploration 1.0

# Use more exploration (more thorough search)
cargo run -p coach --bin find-best-net --release -- --phase race --exploration 2.0

# Adjust batch size as well
cargo run -p coach --bin find-best-net --release -- --phase race --batch-size 2000 --exploration 0.5
```

## Parameters

- `--phase`: Either `contact` or `race` - determines which type of neural nets to evaluate
- `--batch-size`: Number of games to play for each evaluation (default: 1,000)
- `--exploration`: Exploration constant for UCB1 (default: 0.3)
  - **0.1-0.5**: Favors exploitation (faster convergence to best arms)
  - **1.0**: Standard UCB1 behavior (sqrt(2) is the theoretical optimum)
  - **2.0+**: Favors exploration (more thorough search)

### Choosing the Exploration Constant

- **Use 0.1-0.3** when you want to quickly identify the best net (aggressive exploitation)
- **Use 0.5-1.0** for balanced exploration and exploitation
- **Use 1.5-2.0** when you want thorough exploration of all candidates
- **Default 0.3** works well for neural nets with small equity differences (±0.01)

## How It Works

1. **Loads all neural nets** once at startup from the `training-data` folder
2. **UCB1 selection** - intelligently picks which net to evaluate next using confidence bounds
3. **Tracks statistics** - maintains running average equity and pull count for each net
4. **Adaptive allocation** - better nets get evaluated more based on the UCB formula
5. **Runs until Ctrl+C** - press Ctrl+C to stop and see final results

## Output

The program displays:
- **Progress**: Total millions of games played
- **Current evaluation**: Which net is being evaluated and its equity
- **Top 3 nets**: Best performers with their average equity and total games
- **Median equity**: Overall performance across all nets

### During Execution

```
[  0.10M games] Evaluated race290: equity =  0.0523 | Top 3: race290: 0.0523 (15000 games), race294: 0.0456 (8000 games), race183: 0.0234 (3000 games) | Median: 0.0189
```

Note: UCB1 allocates more games to better-performing nets, so the game counts will vary.

### Final Results

After pressing Ctrl+C:
```
========================================
Final Results:
========================================

race290 is winning. After 450000 games (45.0%) the equity is  0.0521.
race294 is winning. After 280000 games (28.0%) the equity is  0.0489.
race183 is winning. After 120000 games (12.0%) the equity is  0.0245.
race300 is  losing. After 80000 games (8.0%) the equity is -0.0123.
race149 is  losing. After 70000 games (7.0%) the equity is -0.0234.
...
```

Results are sorted by equity (best first), showing:
- Neural net name
- Whether it's winning (positive equity) or losing (negative equity)
- Total games played and percentage of total
- Average equity across all evaluations

## Expected Behavior

With default exploration constant (0.3):

- **Initial phase**: All nets get evaluated at least once
- **Early phase**: Rapid convergence starts as UCB1 learns performance
- **Middle phase**: Better nets receive progressively more evaluations
- **Late phase**: Top 2-3 nets typically get 70-90% of total evaluations

With higher exploration (e.g., 1.0+):
- Slower convergence, more balanced distribution
- Better for thorough exploration
- Standard UCB1 theoretical guarantees apply

With lower exploration (e.g., 0.1):
- Very fast convergence to apparent best
- Risk of premature convergence
- Good when you're confident early results are reliable

## Advantages

1. **Tunable behavior**: Adjust exploration constant to match your needs
2. **Simple and effective**: Standard UCB1 with proven performance guarantees
3. **Efficient**: Doesn't waste time on poor performers
4. **Adaptive**: Automatically adjusts based on observed performance
5. **No budget needed**: Runs until you decide to stop
6. **Good for small differences**: Works well with equity values around ±0.01
7. **Fast convergence**: With c=0.3, quickly focuses on best performers

## Comparison with compare-folder

| Feature | compare-folder | find-best-net |
|---------|---------------|---------------|
| Selection | Sequential | UCB1 (adaptive) |
| Execution | Fixed games per net | Continuous until Ctrl+C |
| Game distribution | Equal | Adaptive (more for better nets) |
| Use case | One-shot comparison | Continuous intelligent search |

## Example Workflow

Run this during neural net training to monitor which nets are performing best:

```bash
cargo run -p coach --bin find-best-net --release -- --phase race
```

The algorithm will continuously evaluate nets, giving more attention to top performers. When you're satisfied (or want to check progress), press Ctrl+C to see the results.

