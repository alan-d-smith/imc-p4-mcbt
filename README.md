# IMC Prosperity 4 Monte Carlo Backtester

Round-aware replay and empirical block-bootstrap Monte Carlo tooling for IMC Prosperity strategies.

This checkout currently powers the local workflow in the parent strategy repo. It provides:

- `python -m prosperity4mcbt` for Monte Carlo simulation
- `python -m prosperity4mcbt.replay` for exact historical replay
- a local dashboard bundle plus visualizer integration
- dynamic product handling based on the supplied CSVs rather than hard-coded tutorial products

You do not need to rewrite your trader for Monte Carlo mode. If your file exposes the normal `Trader.run(state)` interface, it should run.

## What Works Right Now

- historical replay for the parent repo's round 1 / round 2 CSV layout
- empirical block-bootstrap Monte Carlo calibrated from real price and trade data
- multi-day synthetic sessions with persistent positions, cash, and `traderData`
- optional PNG plots for persisted sample sessions
- local visualizer integration through `--vis`
- dynamic product names in the dashboard and visualizer

In the parent repo, this is currently used for `ASH_COATED_OSMIUM` and `INTARIAN_PEPPER_ROOT`.

## Prerequisites

You need:

- Python `3.9+`
- Node / npm for the visualizer

If you run this vendored inside the parent repo, the parent Python environment is usually the easiest way to satisfy the dependencies.

## Typical Workflow

The examples below use `python -m ...` so they work even if your shell has not refreshed console-script entrypoints yet.

If you are running from the vendored checkout in the parent repo, start from:

```bash
cd external/imc-p4-mcbt/backtester
```

Then use absolute paths or the vendored relative paths shown below.

## Monte Carlo CLI

Quick smoke test:

```bash
python -m prosperity4mcbt /abs/path/to/trader.py --round 2 --data-root /abs/path/to/data --quick
```

Vendored example from `external/imc-p4-mcbt/backtester`:

```bash
python -m prosperity4mcbt ../../../src/trader.py --round 2 --data-root ../../../data --quick
```

More standard run:

```bash
python -m prosperity4mcbt /abs/path/to/trader.py --round 2 --data-root /abs/path/to/data --sessions 100 --sample-sessions 10
```

Useful flags:

- `--round 1|2`: choose the round to calibrate from
- `--data-root PATH` or `--data PATH`: root data directory
- `--sessions N`: number of Monte Carlo sessions
- `--sample-sessions N`: number of full sample paths to persist
- `--session-days N`: synthetic day-equivalents per session
- `--block-size N`: mean contiguous bootstrap block length in ticks
- `--workers N`: worker processes for independent sessions
- `--plot-samples`: save PNG plots for persisted sample sessions
- `--vis`: open the dashboard in the local visualizer
- `--out PATH`: dashboard JSON path

### Presets

- default: `100` sessions, `10` sample sessions
- `--quick`: `25` sessions, `5` sample sessions
- `--heavy`: `300` sessions, `20` sample sessions

## Replay CLI

Replay all available days in a round:

```bash
python -m prosperity4mcbt.replay 2 --algorithm /abs/path/to/trader.py --data-root /abs/path/to/data
```

Vendored example:

```bash
python -m prosperity4mcbt.replay 2 --algorithm ../../../src/trader.py --data-root ../../../data
```

Replay one specific day:

```bash
python -m prosperity4mcbt.replay 2 --algorithm /abs/path/to/trader.py --data-root /abs/path/to/data --day -1
```

Useful flags:

- `--day D`: replay one day; repeat the flag to replay multiple days
- `--out PATH`: save merged replay log to a custom path
- `--no-out`: skip writing the merged replay log
- `--print`: print trader output inline
- `--no-progress`: disable the progress bar
- `--no-merge-pnl`: keep per-day PnL separate in merged output
- `--original-timestamps`: keep original timestamps across merged days
- `--match-trades all|worse|none`: replay market-trade matching mode

Replay logs default to:

```text
backtests/results/mcbt_replay/
```

relative to the parent strategy repo.

## Visualizer

Start the visualizer:

```bash
cd ../visualizer
npm install
npm run dev
```

Then run Monte Carlo with `--vis`, for example:

```bash
cd ../backtester
python -m prosperity4mcbt ../../../src/trader.py --round 2 --data-root ../../../data --quick --vis
```

The local dashboard is served through:

```text
http://127.0.0.1:5555/
```

The frontend proxies `/dashboard.json` and related sidecar files to the local dashboard server on port `8001`.

## How Monte Carlo Works

The current Monte Carlo engine is an empirical, time-aligned block bootstrap built from the supplied historical CSVs.

At a high level it:

1. discovers the available days for the selected round
2. loads real order books and market trades for every product
3. normalizes sparse or empty-book moments so synthetic mids stay usable
4. samples short contiguous blocks from real history while keeping products aligned on the same ticks
5. reanchors only products with meaningful day-to-day level shifts
6. carries positions, cash, and `traderData` across synthetic day boundaries
7. writes a dashboard bundle with distributions, path boards, and sample sessions

This makes it a robustness tool for nearby plausible worlds built from real data. It is not a first-principles market simulator.

## Output Bundle

A Monte Carlo run writes:

- `dashboard.json`
- `session_summary.csv`
- `run_summary.csv`
- `sample_paths/`
- `sessions/`
- `plots/` when `--plot-samples` is enabled
- `run.log`

If `--out` is omitted, the dashboard defaults to:

```text
backtests/results/mcbt_monte_carlo/<timestamp>/dashboard.json
```

relative to the parent strategy repo.

## Strategy Contract

Your strategy only needs the normal Prosperity interface:

```python
class Trader:
    def run(self, state):
        return orders, conversions, trader_data
```

`prosperity4mcbt` handles:

- state preparation
- replay-style order matching
- cash and position accounting
- mark-to-market PnL
- dashboard bundle generation

No special visualizer logger is required for Monte Carlo mode.

Compatible import styles include:

- `from datamodel import ...`
- `from prosperity3bt.datamodel import ...`
- `from prosperity4mcbt.datamodel import ...`

Monte Carlo still uses minimal observations and does not currently simulate conversions, which is fine for the current round 1 / round 2 products in the parent repo.

## Repo Layout

```text
imc-p4-mcbt/
|-- backtester/         # Python CLIs and dashboard bundle builder
|-- rust_simulator/     # Rust simulator experiments
|-- visualizer/         # local visualizer frontend
|-- scripts/            # helper scripts
|-- data/               # legacy local data and calibration artifacts
`-- docs/               # screenshots and notes
```

## Attribution

This project includes adapted components from Jasper van Merle's open-source IMC Prosperity 3 tooling:

- backtester lineage: https://github.com/jmerle/imc-prosperity-3-backtester
- visualizer lineage: https://github.com/jmerle/imc-prosperity-3-visualizer

The historical replay CLI and parts of the visualizer shell started from those projects and were then extended for the current Prosperity 4 research workflow.

## Open-Source Hygiene

Do not commit:

- `.env.local`
- `tmp/`
- `rust_simulator/target/`
- `visualizer/dist/`

Local auth tokens live outside the repo in `~/.prosperity4mcbt/`.

## Status

This checkout is tuned for fast local strategy iteration inside the parent repo. The replay path is intended for exact historical measurement, and the Monte Carlo path is intended for robustness testing on synthetic sessions derived from real data.
