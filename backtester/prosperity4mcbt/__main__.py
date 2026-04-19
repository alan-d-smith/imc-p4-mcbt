import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated, Optional

from typer import Argument, Option, Typer

from prosperity4mcbt.monte_carlo import default_dashboard_path, normalize_dashboard_path, run_monte_carlo_mode
from prosperity4mcbt.open import open_dashboard


def version_callback(value: bool) -> None:
    if value:
        try:
            version = metadata.version("prosperity4mcbt")
        except metadata.PackageNotFoundError:
            version = "0.0.0+local"
        print(f"prosperity4mcbt {version}")
        raise SystemExit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[
        Path,
        Argument(
            help="Path to the Python file containing the strategy to simulate.",
            show_default=False,
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    round_num: Annotated[int, Option("--round", help="Round number to simulate, for example 1 or 2.")] = 1,
    vis: Annotated[bool, Option("--vis", help="Open the Monte Carlo dashboard in the local visualizer when done.")] = False,
    out: Annotated[
        Optional[Path],
        Option(
            help="Path to dashboard JSON file (defaults to backtests/results/mcbt_monte_carlo/<timestamp>/dashboard.json).",
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    data: Annotated[
        Optional[Path],
        Option(
            help="Path to the repo data directory. Both data/ and data/raw/ layouts are supported.",
            show_default=False,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    quick: Annotated[
        bool,
        Option("--quick", help="Preset for a fast run: 25 sessions and 5 sample sessions."),
    ] = False,
    heavy: Annotated[
        bool,
        Option("--heavy", help="Preset for a larger run: 300 sessions and 20 sample sessions."),
    ] = False,
    sessions: Annotated[int, Option("--sessions", help="Number of Monte Carlo sessions to run.")] = 100,
    sample_sessions: Annotated[
        int,
        Option("--sample-sessions", help="Number of sessions to persist with full trace data for charts."),
    ] = 10,
    block_size: Annotated[int, Option("--block-size", help="Mean contiguous block length in ticks.")] = 250,
    seed: Annotated[int, Option("--seed", help="RNG seed for the Monte Carlo bootstrap.")] = 20260419,
    workers: Annotated[int, Option("--workers", help="Worker processes used for independent sessions.")] = 4,
    match_trades: Annotated[
        str,
        Option("--match-trades", help="Replay-engine market-trade matching mode."),
    ] = "all",
    no_progress: Annotated[bool, Option("--no-progress", help="Disable the session progress bar.")] = False,
    version: Annotated[
        bool,
        Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback),
    ] = False,
) -> None:
    _ = version
    if quick and heavy:
        print("Error: --quick and --heavy are mutually exclusive")
        raise SystemExit(1)

    if quick:
        sessions = 25
        sample_sessions = 5
    elif heavy:
        sessions = 300
        sample_sessions = 20

    dashboard_path = normalize_dashboard_path(out, False) or default_dashboard_path(round_num)

    dashboard = run_monte_carlo_mode(
        algorithm=algorithm,
        dashboard_path=dashboard_path,
        data_root=data,
        round_num=round_num,
        sessions=sessions,
        sample_sessions=sample_sessions,
        block_size=block_size,
        seed=seed,
        workers=max(1, workers),
        trade_matching_mode=match_trades,
        show_progress=not no_progress,
    )

    total_stats = dashboard["overall"]["totalPnl"]
    print(f"Round: {round_num}")
    print(f"Sessions: {int(total_stats['count'])}")
    print(f"Mean total PnL: {total_stats['mean']:,.2f}")
    print(f"Std total PnL: {total_stats['std']:,.2f}")
    print(f"Median total PnL: {total_stats['p50']:,.2f}")
    print(f"5%-95% range: {total_stats['p05']:,.2f} to {total_stats['p95']:,.2f}")
    print(f"Saved Monte Carlo dashboard to {dashboard_path}")

    if vis:
        open_dashboard(dashboard_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
