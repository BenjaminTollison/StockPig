# StockPig

<p align="center">
  <img src="assets/StockPig-logo.png" alt="StockPig logo" width="160">
</p>

**StockPig** is a Streamlit-based college football analytics project focused on SEC score prediction, weekly team rankings, and pick'em survival planning.

The name is a combination of **Stockfish**, the chess engine, and **pigskin**.

## What StockPig Does

StockPig brings several SEC analysis tools into one app:

- **Score prediction** for SEC matchups
- **Win probability simulation**
- **Weekly SEC rankings**
- **Pick'em optimization**
- **Season survival planning**
- **Historical week replay** to see what the model would have recommended at an earlier point in the season
- **Printable survival reports** with rankings, recommended picks, score predictions, and simulated score distributions

The project is designed so predictions for a selected week only use information that would have been available before that week.

## Quick Start

### Requirements

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) recommended for environment management

Clone the repository:

```bash
git clone https://github.com/BenjaminTollison/StockPig.git
cd StockPig
```

Create and synchronize the environment:

```bash
uv sync
```

Run the app:

```bash
uv run streamlit run app.py
```

## GPU Acceleration

StockPig can use PyTorch acceleration for Monte Carlo simulations.

For the AMD Radeon `gfx1201` / ROCm environment used during development:

```bash
uv sync --extra amd-gfx1201
```

For a standard PyTorch environment:

```bash
uv sync --extra torch
```

The prediction tools fall back to CPU where supported if GPU acceleration is unavailable.

## Main Tools

### SEC Survival Planner

The primary planning workflow combines rankings, score predictions, win probabilities, and pick'em optimization into one report.

It can:

- Plan from the current week through the end of the season
- Optimize for a shorter target survival week
- Respect previously used teams and double-pick weeks
- Replay earlier weeks using only information available at that point
- Generate simulated score distributions for recommended matchups
- Export a print-friendly report

### Score Predictor

Estimates matchup scores and converts those predictions into simulated score distributions and win probabilities.

### SEC Weekly Rankings

Ranks SEC teams using recent offensive and defensive performance.

### Pick'em Optimizer

Finds a season-long sequence of SEC picks while accounting for team reuse restrictions and required double-pick weeks.

## Project Layout

```text
StockPig/
├── app.py
├── pages/
│   ├── home.py
│   ├── survival_planner.py
│   ├── score_predictor_gpu.py
│   ├── pickem_optimizer_gpu.py
│   └── sec_ranking.py
├── predictor_core_gpu.py
├── stat_processing.py
├── assets/
├── fantasy_football/
├── pyproject.toml
└── uv.lock
```

Generated historical feature data is stored locally under:

```text
data/feature_store/
```

This allows later runs to reuse previously built features instead of recalculating the full historical dataset.

## Data

StockPig uses college football data provided through **SportsDataverse**.

Model outputs are analytical estimates, not sportsbook odds or guarantees of game outcomes.

## Development Status

StockPig is an actively developed personal analytics project. Features, model behavior, and interfaces may change as the project evolves.

## License

See [`LICENSE`](LICENSE) for license information.