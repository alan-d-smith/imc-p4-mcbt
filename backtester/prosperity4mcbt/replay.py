from __future__ import annotations

import argparse
from collections import defaultdict
from functools import reduce
from pathlib import Path

from prosperity4mcbt.repo import (
    DEFAULT_DATA_ROOT,
    DEFAULT_REPLAY_RESULTS_DIR,
    REPO_ROOT,
    RepoFileReader,
    discover_days,
    format_path,
    load_trader_module,
)

from prosperity3bt.models import BacktestResult, TradeMatchingMode
from prosperity3bt.runner import run_backtest


def default_output_path(round_num: int, days: list[int]) -> Path:
    if len(days) == 1:
        suffix = f"round_{round_num}_day_{days[0]}"
    else:
        suffix = f"round_{round_num}"
    return DEFAULT_REPLAY_RESULTS_DIR / f"{suffix}.log"


def print_day_summary(result: BacktestResult) -> None:
    last_timestamp = result.activity_logs[-1].timestamp

    product_lines = []
    total_profit = 0

    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break

        product = row.columns[2]
        profit = row.columns[-1]

        product_lines.append(f"{product}: {profit:,.0f}")
        total_profit += profit

    print(*reversed(product_lines), sep="\n")
    print(f"Total profit: {total_profit:,.0f}")


def print_overall_summary(results: list[BacktestResult]) -> None:
    print("Profit summary:")

    total_profit = 0
    for result in results:
        last_timestamp = result.activity_logs[-1].timestamp

        profit = 0
        for row in reversed(result.activity_logs):
            if row.timestamp != last_timestamp:
                break

            profit += row.columns[-1]

        print(f"Round {result.round_num} day {result.day_num}: {profit:,.0f}")
        total_profit += profit

    print(f"Total profit: {total_profit:,.0f}")


def merge_results(
    a: BacktestResult, b: BacktestResult, merge_profit_loss: bool, merge_timestamps: bool
) -> BacktestResult:
    sandbox_logs = a.sandbox_logs[:]
    activity_logs = a.activity_logs[:]
    trades = a.trades[:]

    if merge_timestamps:
        a_last_timestamp = a.activity_logs[-1].timestamp
        timestamp_offset = a_last_timestamp + 100
    else:
        timestamp_offset = 0

    sandbox_logs.extend([row.with_offset(timestamp_offset) for row in b.sandbox_logs])
    trades.extend([row.with_offset(timestamp_offset) for row in b.trades])

    if merge_profit_loss:
        profit_loss_offsets = defaultdict(float)
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_timestamp:
                break
            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            [row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]]) for row in b.activity_logs]
        )
    else:
        activity_logs.extend([row.with_offset(timestamp_offset, 0) for row in b.activity_logs])

    return BacktestResult(a.round_num, a.day_num, sandbox_logs, activity_logs, trades)


def write_output(output_file: Path, merged_results: BacktestResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w+", encoding="utf-8") as file:
        file.write("Sandbox logs:\n")
        for row in merged_results.sandbox_logs:
            file.write(str(row))

        file.write("\n\n\nActivities log:\n")
        file.write(
            "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
        )
        file.write("\n".join(map(str, merged_results.activity_logs)))

        file.write("\n\n\n\n\nTrade History:\n")
        file.write("[\n")
        file.write(",\n".join(map(str, merged_results.trades)))
        file.write("]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay round 1/2 historical data through the vendored imc-p4-mcbt backtester."
    )
    parser.add_argument("round", type=int, help="Round number to replay, for example 1 or 2.")
    parser.add_argument(
        "--day",
        dest="days",
        action="append",
        type=int,
        help="Specific day to replay. Repeat the flag to select multiple days. Defaults to all available days in the round.",
    )
    parser.add_argument("--algorithm", default=str(REPO_ROOT / "src" / "trader.py"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-out", action="store_true")
    parser.add_argument("--no-merge-pnl", action="store_true")
    parser.add_argument("--print", dest="print_output", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--original-timestamps", action="store_true")
    parser.add_argument(
        "--match-trades",
        choices=[mode.value for mode in TradeMatchingMode],
        default=TradeMatchingMode.worse.value,
        help="Market-trade matching mode passed to the replay engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algorithm = Path(args.algorithm).resolve()
    data_root = Path(args.data_root).resolve()

    if not algorithm.is_file():
        raise FileNotFoundError(f"Algorithm file not found: {algorithm}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    file_reader = RepoFileReader(data_root)
    days = discover_days(file_reader, args.round, args.days)

    results = []
    for day_num in days:
        print(f"Backtesting {algorithm} on round {args.round} day {day_num}")
        trader_module = load_trader_module(algorithm)
        if not hasattr(trader_module, "Trader"):
            raise ValueError(f"{algorithm} does not expose a Trader class")

        result = run_backtest(
            trader_module.Trader(),
            file_reader,
            args.round,
            day_num,
            args.print_output,
            TradeMatchingMode(args.match_trades),
            True,
            not args.no_progress and not args.print_output,
        )

        print_day_summary(result)
        if len(days) > 1:
            print()

        results.append(result)

    if len(days) > 1:
        print_overall_summary(results)

    if not args.no_out:
        output_file = args.out.resolve() if args.out else default_output_path(args.round, days)
        merged_results = reduce(
            lambda a, b: merge_results(a, b, not args.no_merge_pnl, not args.original_timestamps),
            results,
        )
        write_output(output_file, merged_results)
        print(f"\nSaved replay log to {format_path(output_file)}")


if __name__ == "__main__":
    main()
