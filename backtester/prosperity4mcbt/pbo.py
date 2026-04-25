from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from prosperity3bt import data as bt_data
from prosperity3bt.models import TradeMatchingMode
from prosperity4mcbt.monte_carlo import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SEED,
    bootstrap_day,
    build_day_path,
    load_round_dataset,
    run_backtest_with_data,
    sample_std,
    summarize_distribution,
    worker_default_count,
)
from prosperity4mcbt.repo import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PBO_RESULTS_DIR,
    POSITION_LIMITS,
    REPO_ROOT,
    RepoFileReader,
    discover_days,
    format_path,
    load_trader_module,
)


DEFAULT_TRIALS = 32
DEFAULT_GROUPS = 8
DEFAULT_METRIC = "mean_pnl"
GENERATED_OUTPUT_FILES = {
    "pbo.json",
    "algorithm_summary.csv",
    "trial_summary.csv",
    "split_summary.csv",
    "run.log",
}
GENERATED_OUTPUT_DIRS = {"plots"}


def default_output_path(round_num: int | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_round_{round_num}" if round_num is not None else ""
    return DEFAULT_PBO_RESULTS_DIR / f"{timestamp}{suffix}" / "pbo.json"


def normalize_output_path(out: Optional[Path]) -> Path:
    if out is None:
        return default_output_path()
    if out.suffix.lower() == ".json":
        return out.resolve()
    return (out / "pbo.json").resolve()


def discover_default_algorithms(round_num: int) -> list[Path]:
    candidates = [REPO_ROOT / "src" / "trader.py"]
    submissions_dir = REPO_ROOT / "submissions" / f"round{round_num}"
    if submissions_dir.is_dir():
        candidates.extend(sorted(submissions_dir.glob("*.py")))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def normalize_algorithm_paths(paths: list[str], round_num: int) -> list[Path]:
    algorithms = [Path(path).resolve() for path in paths] if paths else discover_default_algorithms(round_num)
    if not algorithms:
        raise ValueError("No candidate algorithms found. Pass --algorithm or add traders under submissions/roundX/.")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for algorithm in algorithms:
        if not algorithm.is_file():
            raise FileNotFoundError(f"Algorithm file not found: {algorithm}")
        if algorithm not in seen:
            deduped.append(algorithm)
            seen.add(algorithm)

    if len(deduped) < 2:
        raise ValueError("PBO requires at least two candidate algorithms.")
    return deduped


def algorithm_label(path: Path) -> str:
    return format_path(path.resolve())


def metric_score(values: list[float], metric: str) -> float:
    if not values:
        return 0.0

    if metric == "mean_pnl":
        return statistics.fmean(values)
    if metric == "sum_pnl":
        return float(sum(values))
    if metric == "median_pnl":
        return statistics.median(values)
    if metric == "sharpe_like":
        mean = statistics.fmean(values)
        std = sample_std(values)
        return mean / std if std > 1e-12 else 0.0
    if metric == "sortino_like":
        mean = statistics.fmean(values)
        downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in values) / len(values))
        return mean / downside if downside > 1e-12 else 0.0

    raise ValueError(f"Unsupported metric: {metric}")


def best_score_index(scores: list[float], labels: list[str]) -> int:
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], labels[index]))
    return ordered[0]


def ascending_midrank(values: list[float], target_index: int) -> float:
    target = values[target_index]
    strictly_lower = sum(value < target for value in values)
    equal = sum(value == target for value in values)
    return strictly_lower + (equal + 1) / 2.0


def short_label(label: str, max_length: int = 42) -> str:
    if len(label) <= max_length:
        return label
    return f"...{label[-(max_length - 3):]}"


def evaluate_algorithm_trials(task: tuple[str, str, int, tuple[int, ...], int, int, int, str]) -> dict[str, Any]:
    algorithm_str, data_root_str, round_num, source_days, trial_count, block_size, seed, trade_matching_mode = task
    algorithm = Path(algorithm_str).resolve()
    data_root = Path(data_root_str).resolve()
    dataset = load_round_dataset(data_root, round_num, source_days)
    bt_data.LIMITS.update(POSITION_LIMITS)

    trial_rows: list[dict[str, Any]] = []
    for trial_id in range(1, trial_count + 1):
        trader_module = load_trader_module(algorithm)
        if not hasattr(trader_module, "Trader"):
            raise ValueError(f"{algorithm} does not expose a Trader class")
        trader = trader_module.Trader()
        rng = random.Random(seed + trial_id * 9_973)
        synthetic = bootstrap_day(
            dataset=dataset,
            target_day=trial_id - 1,
            rng=rng,
            block_size=block_size,
            keep_session_files=False,
            initial_mid_prices=None,
        )
        backtest_data = bt_data.create_backtest_data(
            round_num=round_num,
            day_num=trial_id - 1,
            prices=synthetic["priceRows"],
            trades=synthetic["trades"],
            observations=[],
        )
        output = run_backtest_with_data(
            trader=trader,
            data=backtest_data,
            trade_matching_mode=TradeMatchingMode(trade_matching_mode),
        )
        day_path = build_day_path(
            result=output.result,
            products=dataset.products,
            timestamp_offset=0,
            initial_positions={product: 0 for product in dataset.products},
            initial_cash={product: 0.0 for product in dataset.products},
            write_trace=False,
        )
        product_pnls = dict(day_path["finalProductPnls"])
        trial_rows.append(
            {
                "trialId": trial_id,
                "totalPnl": float(sum(product_pnls.values())),
                "productPnls": product_pnls,
            }
        )

    return {
        "algorithmPath": str(algorithm),
        "algorithmLabel": algorithm_label(algorithm),
        "trialRows": trial_rows,
    }


def build_trial_groups(trial_count: int, groups: int) -> list[list[int]]:
    if groups < 2 or groups % 2 != 0:
        raise ValueError("--groups must be an even integer >= 2")
    if trial_count < groups:
        raise ValueError("--trials must be at least as large as --groups")
    if trial_count % groups != 0:
        raise ValueError("--trials must be divisible by --groups for CSCV")

    group_size = trial_count // groups
    return [list(range(group_index * group_size, (group_index + 1) * group_size)) for group_index in range(groups)]


def cscv_pbo(
    algorithm_rows: list[dict[str, Any]],
    trial_count: int,
    groups: int,
    metric: str,
) -> dict[str, Any]:
    labels = [row["algorithmLabel"] for row in algorithm_rows]
    scores_by_label = {
        row["algorithmLabel"]: [trial_row["totalPnl"] for trial_row in row["trialRows"]]
        for row in algorithm_rows
    }
    trial_groups = build_trial_groups(trial_count, groups)
    split_rows: list[dict[str, Any]] = []
    selection_counts = {label: 0 for label in labels}

    full_sample_scores = {label: metric_score(values, metric) for label, values in scores_by_label.items()}
    best_full_sample_label = sorted(labels, key=lambda label: (-full_sample_scores[label], label))[0]

    split_id = 0
    half = groups // 2
    all_group_indices = tuple(range(groups))
    for in_sample_groups in combinations(all_group_indices, half):
        split_id += 1
        out_sample_groups = tuple(group for group in all_group_indices if group not in in_sample_groups)
        in_sample_indices = [index for group in in_sample_groups for index in trial_groups[group]]
        out_sample_indices = [index for group in out_sample_groups for index in trial_groups[group]]

        in_sample_scores = [metric_score([scores_by_label[label][index] for index in in_sample_indices], metric) for label in labels]
        out_sample_scores = [metric_score([scores_by_label[label][index] for index in out_sample_indices], metric) for label in labels]

        selected_index = best_score_index(in_sample_scores, labels)
        selected_label = labels[selected_index]
        selection_counts[selected_label] += 1

        oos_rank = ascending_midrank(out_sample_scores, selected_index)
        oos_percentile = oos_rank / (len(labels) + 1.0)
        lambda_logit = math.log(oos_percentile / (1.0 - oos_percentile))
        split_rows.append(
            {
                "splitId": split_id,
                "inSampleGroups": list(in_sample_groups),
                "outOfSampleGroups": list(out_sample_groups),
                "selectedAlgorithm": selected_label,
                "selectedInSampleScore": in_sample_scores[selected_index],
                "selectedOutOfSampleScore": out_sample_scores[selected_index],
                "outOfSampleRank": oos_rank,
                "outOfSamplePercentile": oos_percentile,
                "lambdaLogit": lambda_logit,
                "overfit": lambda_logit < 0.0,
                "inSampleScores": {label: in_sample_scores[index] for index, label in enumerate(labels)},
                "outOfSampleScores": {label: out_sample_scores[index] for index, label in enumerate(labels)},
            }
        )

    lambda_values = [row["lambdaLogit"] for row in split_rows]
    oos_percentiles = [row["outOfSamplePercentile"] for row in split_rows]
    summary = {
        "pbo": sum(row["overfit"] for row in split_rows) / len(split_rows) if split_rows else 0.0,
        "splitCount": len(split_rows),
        "meanLambda": statistics.fmean(lambda_values) if lambda_values else 0.0,
        "medianLambda": statistics.median(lambda_values) if lambda_values else 0.0,
        "meanOutOfSamplePercentile": statistics.fmean(oos_percentiles) if oos_percentiles else 0.0,
        "medianOutOfSamplePercentile": statistics.median(oos_percentiles) if oos_percentiles else 0.0,
        "bestFullSampleAlgorithm": best_full_sample_label,
        "fullSampleScores": full_sample_scores,
        "selectionCounts": selection_counts,
    }
    return {
        "summary": summary,
        "splitRows": split_rows,
        "fullSampleScores": full_sample_scores,
        "selectionCounts": selection_counts,
    }


def render_plots(output_dir: Path, report: dict[str, Any], algorithm_rows: list[dict[str, Any]]) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    lambda_values = [row["lambdaLogit"] for row in report["splitRows"]]
    lambda_path = plot_dir / "lambda_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lambda_values, bins=min(20, max(5, len(lambda_values) // 3)), color="#4c6ef5", edgecolor="white")
    ax.axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title("PBO Lambda Distribution")
    ax.set_xlabel("logit(OOS percentile)")
    ax.set_ylabel("Split count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(lambda_path, dpi=160)
    plt.close(fig)
    saved_paths.append(lambda_path)

    percentile_values = [row["outOfSamplePercentile"] for row in report["splitRows"]]
    percentile_path = plot_dir / "oos_percentile_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(percentile_values, bins=10, range=(0.0, 1.0), color="#12b886", edgecolor="white")
    ax.axvline(0.5, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title("Selected Strategy OOS Percentile")
    ax.set_xlabel("Out-of-sample percentile")
    ax.set_ylabel("Split count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(percentile_path, dpi=160)
    plt.close(fig)
    saved_paths.append(percentile_path)

    labels = [row["algorithmLabel"] for row in algorithm_rows]
    mean_pnls = [statistics.fmean(trial["totalPnl"] for trial in row["trialRows"]) for row in algorithm_rows]
    score_path = plot_dir / "algorithm_mean_pnl.png"
    fig, ax = plt.subplots(figsize=(12, max(4.5, 1.2 * len(labels))))
    positions = list(range(len(labels)))
    ax.barh(positions, mean_pnls, color="#fd7e14")
    ax.set_yticks(positions)
    ax.set_yticklabels([short_label(label) for label in labels])
    ax.set_title("Candidate Mean Synthetic-Day PnL")
    ax.set_xlabel("Mean total PnL")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(score_path, dpi=160)
    plt.close(fig)
    saved_paths.append(score_path)

    return saved_paths


def write_csv_outputs(output_dir: Path, algorithm_rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    with (output_dir / "algorithm_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "algorithm_label",
            "algorithm_path",
            "trial_count",
            "mean_total_pnl",
            "std_total_pnl",
            "median_total_pnl",
            "p05_total_pnl",
            "p95_total_pnl",
            "full_sample_score",
            "selection_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in algorithm_rows:
            values = [trial["totalPnl"] for trial in row["trialRows"]]
            summary = summarize_distribution(values)
            writer.writerow(
                {
                    "algorithm_label": row["algorithmLabel"],
                    "algorithm_path": row["algorithmPath"],
                    "trial_count": len(values),
                    "mean_total_pnl": summary.get("mean", 0.0),
                    "std_total_pnl": summary.get("std", 0.0),
                    "median_total_pnl": summary.get("p50", 0.0),
                    "p05_total_pnl": summary.get("p05", 0.0),
                    "p95_total_pnl": summary.get("p95", 0.0),
                    "full_sample_score": report["fullSampleScores"][row["algorithmLabel"]],
                    "selection_count": report["selectionCounts"].get(row["algorithmLabel"], 0),
                }
            )

    with (output_dir / "trial_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["trial_id", "algorithm_label", "algorithm_path", "total_pnl", "product_pnls_json"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in algorithm_rows:
            for trial in row["trialRows"]:
                writer.writerow(
                    {
                        "trial_id": trial["trialId"],
                        "algorithm_label": row["algorithmLabel"],
                        "algorithm_path": row["algorithmPath"],
                        "total_pnl": trial["totalPnl"],
                        "product_pnls_json": json.dumps(trial["productPnls"], separators=(",", ":")),
                    }
                )

    with (output_dir / "split_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split_id",
            "selected_algorithm",
            "selected_in_sample_score",
            "selected_out_of_sample_score",
            "out_of_sample_rank",
            "out_of_sample_percentile",
            "lambda_logit",
            "overfit",
            "in_sample_groups_json",
            "out_of_sample_groups_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["splitRows"]:
            writer.writerow(
                {
                    "split_id": row["splitId"],
                    "selected_algorithm": row["selectedAlgorithm"],
                    "selected_in_sample_score": row["selectedInSampleScore"],
                    "selected_out_of_sample_score": row["selectedOutOfSampleScore"],
                    "out_of_sample_rank": row["outOfSampleRank"],
                    "out_of_sample_percentile": row["outOfSamplePercentile"],
                    "lambda_logit": row["lambdaLogit"],
                    "overfit": row["overfit"],
                    "in_sample_groups_json": json.dumps(row["inSampleGroups"]),
                    "out_of_sample_groups_json": json.dumps(row["outOfSampleGroups"]),
                }
            )


def write_run_log(
    output_dir: Path,
    round_num: int,
    algorithms: list[Path],
    trial_count: int,
    groups: int,
    metric: str,
    block_size: int,
    seed: int,
    workers: int,
    trade_matching_mode: str,
    report: dict[str, Any],
) -> None:
    lines = [
        f"round={round_num}",
        f"algorithm_count={len(algorithms)}",
        f"algorithms={json.dumps([format_path(path) for path in algorithms])}",
        f"trial_count={trial_count}",
        f"groups={groups}",
        f"metric={metric}",
        f"block_size={block_size}",
        f"seed={seed}",
        f"workers={workers}",
        f"trade_matching_mode={trade_matching_mode}",
        f"pbo={report['summary']['pbo']:.6f}",
        f"mean_lambda={report['summary']['meanLambda']:.6f}",
        f"best_full_sample_algorithm={report['summary']['bestFullSampleAlgorithm']}",
    ]
    (output_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pbo_mode(
    algorithms: list[Path],
    output_path: Path,
    data_root: Optional[Path],
    round_num: int,
    trials: int,
    groups: int,
    metric: str,
    block_size: int,
    seed: int,
    workers: int,
    trade_matching_mode: str,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir = output_path.parent.resolve()
    data_root = (data_root or DEFAULT_DATA_ROOT).resolve()

    if output_dir.exists():
        for name in GENERATED_OUTPUT_FILES:
            path = output_dir / name
            if path.is_file():
                path.unlink()
        for name in GENERATED_OUTPUT_DIRS:
            path = output_dir / name
            if path.is_dir():
                shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_reader = RepoFileReader(data_root)
    source_days = tuple(discover_days(file_reader, round_num, None))
    build_trial_groups(trials, groups)

    tasks = [
        (str(algorithm), str(data_root), round_num, source_days, trials, block_size, seed, trade_matching_mode)
        for algorithm in algorithms
    ]

    algorithm_rows: list[dict[str, Any]] = []
    max_workers = max(1, min(workers, len(tasks)))
    if max_workers <= 1:
        iterator = tqdm(tasks, disable=not show_progress, desc="PBO algorithms", ascii=True)
        for task in iterator:
            algorithm_rows.append(evaluate_algorithm_trials(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_algorithm_trials, task) for task in tasks]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="PBO algorithms", ascii=True)
            for future in iterator:
                algorithm_rows.append(future.result())

    algorithm_rows.sort(key=lambda row: row["algorithmLabel"])
    report = cscv_pbo(algorithm_rows, trial_count=trials, groups=groups, metric=metric)
    plot_files = render_plots(output_dir, report, algorithm_rows)
    write_csv_outputs(output_dir, algorithm_rows, report)
    write_run_log(
        output_dir=output_dir,
        round_num=round_num,
        algorithms=algorithms,
        trial_count=trials,
        groups=groups,
        metric=metric,
        block_size=block_size,
        seed=seed,
        workers=max_workers,
        trade_matching_mode=trade_matching_mode,
        report=report,
    )

    payload = {
        "kind": "pbo_report",
        "meta": {
            "round": round_num,
            "dataRoot": str(data_root),
            "sourceDays": list(source_days),
            "algorithmCount": len(algorithms),
            "algorithms": [{"label": algorithm_label(path), "path": str(path)} for path in algorithms],
            "trialCount": trials,
            "groups": groups,
            "metric": metric,
            "blockSize": block_size,
            "seed": seed,
            "workers": max_workers,
            "tradeMatchingMode": trade_matching_mode,
            "plotFiles": [str(path) for path in plot_files],
        },
        "summary": report["summary"],
        "algorithms": [
            {
                "label": row["algorithmLabel"],
                "path": row["algorithmPath"],
                "trialStats": summarize_distribution([trial["totalPnl"] for trial in row["trialRows"]]),
                "fullSampleScore": report["fullSampleScores"][row["algorithmLabel"]],
                "selectionCount": report["selectionCounts"].get(row["algorithmLabel"], 0),
            }
            for row in algorithm_rows
        ],
        "trials": algorithm_rows,
        "splits": report["splitRows"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Probability of Backtest Overfitting (PBO) with CSCV over multiple candidate traders."
    )
    parser.add_argument("--round", type=int, default=1, help="Round number used to source historical data.")
    parser.add_argument(
        "--algorithm",
        dest="algorithms",
        action="append",
        default=[],
        help="Candidate trader file. Repeat the flag to compare multiple algorithms. Defaults to src/trader.py plus submissions/roundX/*.py.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument(
        "--metric",
        choices=["mean_pnl", "sum_pnl", "median_pnl", "sharpe_like", "sortino_like"],
        default=DEFAULT_METRIC,
        help="Fold score used to rank candidate algorithms inside CSCV.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=worker_default_count())
    parser.add_argument(
        "--match-trades",
        choices=[mode.value for mode in TradeMatchingMode],
        default=TradeMatchingMode.all.value,
        help="Replay-engine market-trade matching mode.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = normalize_algorithm_paths(args.algorithms, args.round)
    output_path = normalize_output_path(args.out)

    report = run_pbo_mode(
        algorithms=algorithms,
        output_path=output_path,
        data_root=Path(args.data_root),
        round_num=args.round,
        trials=args.trials,
        groups=args.groups,
        metric=args.metric,
        block_size=args.block_size,
        seed=args.seed,
        workers=args.workers,
        trade_matching_mode=args.match_trades,
        show_progress=not args.no_progress,
    )

    summary = report["summary"]
    print(f"Round: {args.round}")
    print(f"Algorithms: {report['meta']['algorithmCount']}")
    print(f"Trials: {report['meta']['trialCount']}")
    print(f"Groups: {report['meta']['groups']}")
    print(f"Metric: {report['meta']['metric']}")
    print(f"PBO: {summary['pbo']:.2%}")
    print(f"Median lambda: {summary['medianLambda']:.4f}")
    print(f"Best full-sample algorithm: {summary['bestFullSampleAlgorithm']}")
    print(f"Saved PBO report to {format_path(output_path)}")
    if report["meta"]["plotFiles"]:
        print("Saved plots:")
        for plot_file in report["meta"]["plotFiles"]:
            print(f"  {format_path(Path(plot_file))}")


if __name__ == "__main__":
    main()
