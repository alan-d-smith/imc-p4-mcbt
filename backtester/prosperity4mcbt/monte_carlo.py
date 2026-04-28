from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from prosperity4mcbt.repo import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MONTE_CARLO_RESULTS_DIR,
    POSITION_LIMITS,
    REPO_ROOT,
    RepoFileReader,
    discover_days,
    format_path,
    load_trader_module,
)
from prosperity3bt import data as bt_data
from prosperity3bt.datamodel import Observation, Trade, TradingState
from prosperity3bt.models import (
    ActivityLogRow,
    BacktestResult,
    MarketTrade,
    SandboxLogRow,
    TradeMatchingMode,
    TradeRow,
)
from prosperity3bt.runner import (
    create_activity_logs,
    enforce_limits,
    match_orders,
    prepare_state,
    type_check_orders,
)


TIMESTAMP_STEP = 100
TICKS_PER_DAY = 10_000
TERMINAL_TIMESTAMP = 999_901
DAY_TIMESTAMP_OFFSET = 1_000_000
DEFAULT_SEED = 20_260_419
DEFAULT_BLOCK_SIZE = 250
LEVEL_REANCHOR_DAY_MEAN_RANGE_THRESHOLD = 100.0
DEFAULT_SESSIONS = 100
DEFAULT_SAMPLE_SESSIONS = 10
QUICK_SESSIONS = 25
QUICK_SAMPLE_SESSIONS = 5
HEAVY_SESSIONS = 300
HEAVY_SAMPLE_SESSIONS = 20
CHART_POINTS_PER_SERIES = 1500
STATIC_CHART_POINTS = 600
PRODUCT_COLORS = {
    "ASH_COATED_OSMIUM": "#12b886",
    "INTARIAN_PEPPER_ROOT": "#fd7e14",
}
GENERATED_OUTPUT_FILES = {
    "dashboard.json",
    "session_summary.csv",
    "run_summary.csv",
    "run.log",
}
GENERATED_OUTPUT_DIRS = {
    "plots",
    "sample_paths",
    "sessions",
}
DATASET_CACHE_VERSION = 1
DATASET_CACHE_DIR = REPO_ROOT / "backtests" / "results" / "mcbt_cache"

WORKER_CONTEXT: WorkerContext | None = None


@dataclass(frozen=True)
class HistoricalPriceRow:
    product: str
    bid_prices: tuple[int, ...]
    bid_volumes: tuple[int, ...]
    ask_prices: tuple[int, ...]
    ask_volumes: tuple[int, ...]
    mid_price: float

    def with_mid_price(self, mid_price: float) -> HistoricalPriceRow:
        return HistoricalPriceRow(
            product=self.product,
            bid_prices=self.bid_prices,
            bid_volumes=self.bid_volumes,
            ask_prices=self.ask_prices,
            ask_volumes=self.ask_volumes,
            mid_price=mid_price,
        )

    def shifted(self, price_delta: int, mid_price: Optional[float] = None) -> HistoricalPriceRow:
        return HistoricalPriceRow(
            product=self.product,
            bid_prices=tuple(price + price_delta for price in self.bid_prices),
            bid_volumes=self.bid_volumes,
            ask_prices=tuple(price + price_delta for price in self.ask_prices),
            ask_volumes=self.ask_volumes,
            mid_price=self.mid_price + price_delta if mid_price is None else mid_price,
        )

    def to_backtest_row(self, day: int, timestamp: int) -> bt_data.PriceRow:
        return bt_data.PriceRow(
            day=day,
            timestamp=timestamp,
            product=self.product,
            bid_prices=list(self.bid_prices),
            bid_volumes=list(self.bid_volumes),
            ask_prices=list(self.ask_prices),
            ask_volumes=list(self.ask_volumes),
            mid_price=self.mid_price,
            profit_loss=0.0,
        )

    def to_shifted_backtest_row(self, day: int, timestamp: int, price_delta: int) -> bt_data.PriceRow:
        if price_delta == 0:
            return self.to_backtest_row(day, timestamp)

        return bt_data.PriceRow(
            day=day,
            timestamp=timestamp,
            product=self.product,
            bid_prices=[price + price_delta for price in self.bid_prices],
            bid_volumes=list(self.bid_volumes),
            ask_prices=[price + price_delta for price in self.ask_prices],
            ask_volumes=list(self.ask_volumes),
            mid_price=self.mid_price + price_delta,
            profit_loss=0.0,
        )

    def to_csv_line(self, day: int, timestamp: int) -> str:
        values = [str(day), str(timestamp), self.product]

        for level in range(3):
            bid_price = self.bid_prices[level] if level < len(self.bid_prices) else ""
            bid_volume = self.bid_volumes[level] if level < len(self.bid_volumes) else ""
            values.extend([str(bid_price) if bid_price != "" else "", str(bid_volume) if bid_volume != "" else ""])

        for level in range(3):
            ask_price = self.ask_prices[level] if level < len(self.ask_prices) else ""
            ask_volume = self.ask_volumes[level] if level < len(self.ask_volumes) else ""
            values.extend([str(ask_price) if ask_price != "" else "", str(ask_volume) if ask_volume != "" else ""])

        values.extend([f"{self.mid_price}", "0.0"])
        return ";".join(values)

    def to_shifted_csv_line(self, day: int, timestamp: int, price_delta: int) -> str:
        if price_delta == 0:
            return self.to_csv_line(day, timestamp)

        values = [str(day), str(timestamp), self.product]

        for level in range(3):
            bid_price = self.bid_prices[level] + price_delta if level < len(self.bid_prices) else ""
            bid_volume = self.bid_volumes[level] if level < len(self.bid_volumes) else ""
            values.extend([str(bid_price) if bid_price != "" else "", str(bid_volume) if bid_volume != "" else ""])

        for level in range(3):
            ask_price = self.ask_prices[level] + price_delta if level < len(self.ask_prices) else ""
            ask_volume = self.ask_volumes[level] if level < len(self.ask_volumes) else ""
            values.extend([str(ask_price) if ask_price != "" else "", str(ask_volume) if ask_volume != "" else ""])

        values.extend([f"{self.mid_price + price_delta}", "0.0"])
        return ";".join(values)


@dataclass(frozen=True)
class HistoricalTradeRow:
    symbol: str
    price: int
    quantity: int
    buyer: str
    seller: str

    def shifted(self, price_delta: int) -> HistoricalTradeRow:
        return HistoricalTradeRow(
            symbol=self.symbol,
            price=self.price + price_delta,
            quantity=self.quantity,
            buyer=self.buyer,
            seller=self.seller,
        )

    def to_trade(self, timestamp: int) -> Trade:
        return Trade(
            symbol=self.symbol,
            price=self.price,
            quantity=self.quantity,
            buyer=self.buyer,
            seller=self.seller,
            timestamp=timestamp,
        )

    def to_shifted_trade(self, timestamp: int, price_delta: int) -> Trade:
        return Trade(
            symbol=self.symbol,
            price=self.price + price_delta,
            quantity=self.quantity,
            buyer=self.buyer,
            seller=self.seller,
            timestamp=timestamp,
        )

    def to_csv_line(self, timestamp: int) -> str:
        return ";".join(
            [
                str(timestamp),
                self.buyer,
                self.seller,
                self.symbol,
                "XIRECS",
                f"{float(self.price)}",
                str(self.quantity),
            ]
        )

    def to_shifted_csv_line(self, timestamp: int, price_delta: int) -> str:
        if price_delta == 0:
            return self.to_csv_line(timestamp)

        return ";".join(
            [
                str(timestamp),
                self.buyer,
                self.seller,
                self.symbol,
                "XIRECS",
                f"{float(self.price + price_delta)}",
                str(self.quantity),
            ]
        )


@dataclass(frozen=True)
class SourceTick:
    prices_by_product: dict[str, HistoricalPriceRow]
    trades: tuple[HistoricalTradeRow, ...]


@dataclass(frozen=True)
class SourceDay:
    day: int
    ticks: tuple[SourceTick, ...]


@dataclass(frozen=True)
class RoundDataset:
    round_num: int
    days: tuple[int, ...]
    products: tuple[str, ...]
    source_days: tuple[SourceDay, ...]
    reanchor_products: tuple[str, ...]


@dataclass(frozen=True)
class WorkerContext:
    algorithm: Path
    data_root: Path
    output_dir: Path
    round_num: int
    source_days: tuple[int, ...]
    session_day_labels: tuple[int, ...]
    block_size: int
    trade_matching_mode: TradeMatchingMode
    sample_sessions: int
    workers: int
    dataset: RoundDataset


@dataclass(frozen=True)
class BacktestRunOutput:
    result: BacktestResult
    trader_data: str
    final_positions: dict[str, int]
    final_cash: dict[str, float]


def default_dashboard_path(round_num: int | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_round_{round_num}" if round_num is not None else ""
    return DEFAULT_MONTE_CARLO_RESULTS_DIR / f"{timestamp}{suffix}" / "dashboard.json"


def normalize_dashboard_path(out: Optional[Path], no_out: bool) -> Optional[Path]:
    if no_out:
        return None

    if out is None:
        return default_dashboard_path()

    if out.suffix.lower() == ".json":
        return out

    return out / "dashboard.json"


def worker_default_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def parse_int_columns(columns: list[str], indices: tuple[int, int, int]) -> tuple[int, ...]:
    values: list[int] = []
    for index in indices:
        value = columns[index]
        if value == "":
            break
        values.append(int(value))
    return tuple(values)


def parse_price_row(row: dict[str, str]) -> HistoricalPriceRow:
    bid_prices = tuple(int(row[f"bid_price_{level}"]) for level in range(1, 4) if row[f"bid_price_{level}"] != "")
    bid_volumes = tuple(int(row[f"bid_volume_{level}"]) for level in range(1, 4) if row[f"bid_volume_{level}"] != "")
    ask_prices = tuple(int(row[f"ask_price_{level}"]) for level in range(1, 4) if row[f"ask_price_{level}"] != "")
    ask_volumes = tuple(int(row[f"ask_volume_{level}"]) for level in range(1, 4) if row[f"ask_volume_{level}"] != "")
    return HistoricalPriceRow(
        product=row["product"],
        bid_prices=bid_prices,
        bid_volumes=bid_volumes,
        ask_prices=ask_prices,
        ask_volumes=ask_volumes,
        mid_price=float(row["mid_price"]),
    )


def parse_price_columns(columns: list[str]) -> HistoricalPriceRow:
    return HistoricalPriceRow(
        product=columns[2],
        bid_prices=parse_int_columns(columns, (3, 5, 7)),
        bid_volumes=parse_int_columns(columns, (4, 6, 8)),
        ask_prices=parse_int_columns(columns, (9, 11, 13)),
        ask_volumes=parse_int_columns(columns, (10, 12, 14)),
        mid_price=float(columns[15]),
    )


def quoted_mid_price(row: HistoricalPriceRow) -> Optional[float]:
    if row.bid_prices and row.ask_prices:
        return (row.bid_prices[0] + row.ask_prices[0]) / 2.0
    return None


def observed_mark_price(row: HistoricalPriceRow) -> Optional[float]:
    if row.mid_price > 0:
        return row.mid_price
    return quoted_mid_price(row)


def normalize_mid_prices(prices_by_tick: list[dict[str, HistoricalPriceRow]]) -> list[dict[str, HistoricalPriceRow]]:
    normalized = [dict(tick_prices) for tick_prices in prices_by_tick]
    products = sorted({product for tick_prices in prices_by_tick for product in tick_prices})

    for product in products:
        rows = [normalized[tick_index][product] for tick_index in range(TICKS_PER_DAY)]
        observed = [observed_mark_price(row) for row in rows]
        next_valid: Optional[float] = None
        next_valid_by_tick: list[Optional[float]] = [None] * len(rows)
        for tick_index in range(len(rows) - 1, -1, -1):
            if observed[tick_index] is not None:
                next_valid = observed[tick_index]
            next_valid_by_tick[tick_index] = next_valid

        last_valid: Optional[float] = None
        for tick_index, row in enumerate(rows):
            mark_price = observed[tick_index]
            if mark_price is None:
                if last_valid is not None:
                    mark_price = last_valid
                elif next_valid_by_tick[tick_index] is not None:
                    mark_price = next_valid_by_tick[tick_index]
                else:
                    mark_price = 0.0

            normalized[tick_index][product] = row.with_mid_price(mark_price)
            if mark_price > 0:
                last_valid = mark_price

    return normalized


def parse_trade_row(row: dict[str, str]) -> HistoricalTradeRow:
    return HistoricalTradeRow(
        symbol=row["symbol"],
        price=int(float(row["price"])),
        quantity=int(row["quantity"]),
        buyer=row.get("buyer", "") or "",
        seller=row.get("seller", "") or "",
    )


def parse_trade_columns(columns: list[str]) -> HistoricalTradeRow:
    return HistoricalTradeRow(
        symbol=columns[3],
        price=int(float(columns[5])),
        quantity=int(columns[6]),
        buyer=columns[1],
        seller=columns[2],
    )


def dataset_cache_path(data_root: Path, round_num: int, target_days: tuple[int, ...]) -> Path:
    file_reader = RepoFileReader(data_root)
    parts = [f"version={DATASET_CACHE_VERSION}", f"data_root={data_root.resolve()}", f"round={round_num}"]
    for day_num in target_days:
        parts.append(f"day={day_num}")
        for kind in ("prices", "trades"):
            with file_reader.file([f"round{round_num}", f"{kind}_round_{round_num}_day_{day_num}.csv"]) as file:
                if file is None:
                    parts.append(f"{kind}=missing")
                    continue
                stat = file.stat()
                parts.append(f"{kind}={file.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")

    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    day_part = "_".join(str(day_num).replace("-", "m") for day_num in target_days)
    return DATASET_CACHE_DIR / f"round{round_num}_days_{day_part}_{digest}.pickle"


def load_round_dataset(data_root: Path, round_num: int, target_days: tuple[int, ...]) -> RoundDataset:
    cache_path = dataset_cache_path(data_root, round_num, target_days)
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, RoundDataset):
                return cached
        except Exception:
            cache_path.unlink(missing_ok=True)

    dataset = load_round_dataset_uncached(data_root, round_num, target_days)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(cache_path)
    return dataset


def load_round_dataset_uncached(data_root: Path, round_num: int, target_days: tuple[int, ...]) -> RoundDataset:
    file_reader = RepoFileReader(data_root)
    source_days: list[SourceDay] = []
    products: set[str] = set()
    day_mean_by_product: dict[str, list[float]] = defaultdict(list)

    for day_num in target_days:
        with file_reader.file([f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.csv"]) as file:
            if file is None:
                raise FileNotFoundError(f"Missing prices for round {round_num} day {day_num}")

            prices_by_tick: list[dict[str, HistoricalPriceRow]] = [dict() for _ in range(TICKS_PER_DAY)]
            with file.open("r", encoding="utf-8") as handle:
                next(handle, None)
                for line in handle:
                    columns = line.rstrip("\n").split(";")
                    timestamp = int(columns[1])
                    tick_index = timestamp // TIMESTAMP_STEP
                    if tick_index < 0 or tick_index >= TICKS_PER_DAY:
                        raise ValueError(f"Unexpected timestamp {timestamp} in round {round_num} day {day_num}")
                    parsed = parse_price_columns(columns)
                    prices_by_tick[tick_index][parsed.product] = parsed
                    products.add(parsed.product)

        with file_reader.file([f"round{round_num}", f"trades_round_{round_num}_day_{day_num}.csv"]) as file:
            trades_by_tick: list[list[HistoricalTradeRow]] = [[] for _ in range(TICKS_PER_DAY)]
            if file is not None:
                with file.open("r", encoding="utf-8") as handle:
                    next(handle, None)
                    for line in handle:
                        columns = line.rstrip("\n").split(";")
                        timestamp = int(columns[0])
                        tick_index = timestamp // TIMESTAMP_STEP
                        if 0 <= tick_index < TICKS_PER_DAY:
                            trades_by_tick[tick_index].append(parse_trade_columns(columns))

        normalized_prices = normalize_mid_prices(prices_by_tick)
        for product in normalized_prices[0]:
            mids = [tick_prices[product].mid_price for tick_prices in normalized_prices if tick_prices[product].mid_price > 0]
            if mids:
                day_mean_by_product[product].append(statistics.fmean(mids))
        ticks: list[SourceTick] = []
        for tick_index in range(TICKS_PER_DAY):
            tick_prices = normalized_prices[tick_index]
            if not tick_prices:
                raise ValueError(f"Missing price rows at tick {tick_index} for round {round_num} day {day_num}")
            ticks.append(SourceTick(prices_by_product=tick_prices, trades=tuple(trades_by_tick[tick_index])))

        source_days.append(SourceDay(day=day_num, ticks=tuple(ticks)))

    product_tuple = tuple(sorted(products))
    for source_day in source_days:
        for tick_index, tick in enumerate(source_day.ticks):
            if set(tick.prices_by_product) != set(product_tuple):
                missing = sorted(set(product_tuple) - set(tick.prices_by_product))
                raise ValueError(
                    f"Round {round_num} day {source_day.day} tick {tick_index} is missing products: {', '.join(missing)}"
                )

    reanchor_products = tuple(
        sorted(
            product
            for product, means in day_mean_by_product.items()
            if means and (max(means) - min(means)) >= LEVEL_REANCHOR_DAY_MEAN_RANGE_THRESHOLD
        )
    )

    return RoundDataset(
        round_num=round_num,
        days=target_days,
        products=product_tuple,
        source_days=tuple(source_days),
        reanchor_products=reanchor_products,
    )


def sample_block_length(rng: random.Random, mean_length: int) -> int:
    if mean_length <= 1:
        return 1

    probability = 1.0 / mean_length
    length = 1
    while rng.random() > probability:
        length += 1
    return length


def bootstrap_day(
    dataset: RoundDataset,
    target_day: int,
    rng: random.Random,
    block_size: int,
    keep_session_files: bool,
    initial_mid_prices: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    price_rows: list[bt_data.PriceRow] = []
    trades: list[Trade] = []
    price_lines: list[str] | None = [] if keep_session_files else None
    trade_lines: list[str] | None = [] if keep_session_files else None
    last_emitted_mid = {product: None for product in dataset.products}
    if initial_mid_prices is not None:
        for product, mid_price in initial_mid_prices.items():
            if product in last_emitted_mid and mid_price > 0:
                last_emitted_mid[product] = mid_price

    target_tick = 0
    while target_tick < TICKS_PER_DAY:
        source_day = rng.choice(dataset.source_days)
        jitter = max(25, block_size // 2)
        source_start = max(0, min(TICKS_PER_DAY - 1, target_tick + rng.randint(-jitter, jitter)))
        block_length = min(
            sample_block_length(rng, block_size),
            TICKS_PER_DAY - target_tick,
            TICKS_PER_DAY - source_start,
        )
        block_price_deltas: dict[str, int] = {}
        for product in dataset.products:
            if product not in dataset.reanchor_products:
                block_price_deltas[product] = 0
                continue
            anchor_mid = source_day.ticks[source_start].prices_by_product[product].mid_price
            previous_mid = last_emitted_mid[product]
            if previous_mid is None or anchor_mid <= 0:
                block_price_deltas[product] = 0
            else:
                block_price_deltas[product] = int(round(previous_mid - anchor_mid))

        for offset in range(block_length):
            source_tick = source_day.ticks[source_start + offset]
            timestamp = (target_tick + offset) * TIMESTAMP_STEP
            for product in dataset.products:
                price_row = source_tick.prices_by_product[product]
                price_delta = block_price_deltas[product]
                price_rows.append(price_row.to_shifted_backtest_row(target_day, timestamp, price_delta))
                if price_lines is not None:
                    price_lines.append(price_row.to_shifted_csv_line(target_day, timestamp, price_delta))
                last_emitted_mid[product] = price_row.mid_price + price_delta
            for trade_row in source_tick.trades:
                price_delta = block_price_deltas.get(trade_row.symbol, 0)
                trades.append(trade_row.to_shifted_trade(timestamp, price_delta))
                if trade_lines is not None:
                    trade_lines.append(trade_row.to_shifted_csv_line(timestamp, price_delta))

        target_tick += block_length

    return {
        "priceRows": price_rows,
        "trades": trades,
        "priceLines": price_lines,
        "tradeLines": trade_lines,
        "endingMidPrices": {product: float(mid_price) for product, mid_price in last_emitted_mid.items() if mid_price is not None},
    }


def run_backtest_with_data(
    trader,
    data: bt_data.BacktestData,
    trade_matching_mode: TradeMatchingMode,
    initial_trader_data: str = "",
    initial_positions: Optional[dict[str, int]] = None,
    initial_cash: Optional[dict[str, float]] = None,
) -> BacktestRunOutput:
    trader_data = initial_trader_data
    starting_positions = {product: int((initial_positions or {}).get(product, 0)) for product in data.products}
    starting_cash = {product: float((initial_cash or {}).get(product, 0.0)) for product in data.products}
    state = TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths={},
        own_trades={},
        market_trades={},
        position=starting_positions,
        observations=Observation({}, {}),
    )
    data.profit_loss.update(starting_cash)

    result = BacktestResult(
        round_num=data.round_num,
        day_num=data.day_num,
        sandbox_logs=[],
        activity_logs=[],
        trades=[],
    )

    for timestamp in sorted(data.prices.keys()):
        state.timestamp = timestamp
        state.traderData = trader_data

        prepare_state(state, data)
        orders, conversions, trader_data = trader.run(state)
        _ = conversions

        sandbox_row = SandboxLogRow(timestamp=timestamp, sandbox_log="", lambda_log="")
        result.sandbox_logs.append(sandbox_row)

        type_check_orders(orders)
        create_activity_logs(state, data, result)
        enforce_limits(state, data, orders, sandbox_row)
        match_orders(state, data, orders, result, trade_matching_mode)

    return BacktestRunOutput(
        result=result,
        trader_data=trader_data,
        final_positions={product: state.position.get(product, 0) for product in data.products},
        final_cash={product: data.profit_loss.get(product, 0.0) for product in data.products},
    )


def to_float_or_nan(value: Any) -> float:
    if value == "":
        return math.nan
    return float(value)


def linear_fit(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        return {"slopePerStep": 0.0, "r2": 0.0}

    x_values = list(range(len(values)))
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(values)
    sxx = sum((x - x_mean) ** 2 for x in x_values)
    syy = sum((y - y_mean) ** 2 for y in values)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    r2 = (sxy * sxy) / (sxx * syy) if sxx > 1e-12 and syy > 1e-12 else 0.0
    return {"slopePerStep": slope, "r2": r2}


def empty_product_path() -> dict[str, list[float]]:
    return {
        "timestamps": [],
        "fair": [],
        "mid": [],
        "bid1": [],
        "ask1": [],
        "position": [],
        "cash": [],
        "mtmPnl": [],
    }


def build_day_path(
    result: BacktestResult,
    products: tuple[str, ...],
    timestamp_offset: int,
    initial_positions: dict[str, int],
    initial_cash: dict[str, float],
    write_trace: bool,
) -> dict[str, Any]:
    paths = {product: empty_product_path() for product in products}
    total = {"timestamps": [], "mtmPnl": []}
    trace_lines: list[str] = []

    own_trades_by_key: dict[tuple[int, str], list[Trade]] = defaultdict(list)
    for trade_row in result.trades:
        trade = trade_row.trade
        if trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION":
            own_trades_by_key[(trade.timestamp, trade.symbol)].append(trade)

    activity_by_timestamp: dict[int, dict[str, ActivityLogRow]] = defaultdict(dict)
    for row in result.activity_logs:
        activity_by_timestamp[row.timestamp][row.columns[2]] = row

    positions = {product: int(initial_positions.get(product, 0)) for product in products}
    cash = {product: float(initial_cash.get(product, 0.0)) for product in products}
    last_market: dict[str, dict[str, float]] = {
        product: {"mid": 0.0, "bid1": math.nan, "ask1": math.nan} for product in products
    }

    for timestamp in sorted(activity_by_timestamp):
        timestamp_total = 0.0
        rows = activity_by_timestamp[timestamp]

        for product in products:
            row = rows[product]
            mid = float(row.columns[15])
            bid1 = to_float_or_nan(row.columns[3])
            ask1 = to_float_or_nan(row.columns[9])
            mtm = cash[product] + positions[product] * mid

            path = paths[product]
            path["timestamps"].append(timestamp + timestamp_offset)
            path["fair"].append(mid)
            path["mid"].append(mid)
            path["bid1"].append(bid1)
            path["ask1"].append(ask1)
            path["position"].append(float(positions[product]))
            path["cash"].append(cash[product])
            path["mtmPnl"].append(mtm)

            timestamp_total += mtm
            last_market[product] = {"mid": mid, "bid1": bid1, "ask1": ask1}

            if write_trace:
                trace_lines.append(
                    f"{result.day_num};{timestamp};{product};{mid};{positions[product]};{cash[product]};{mtm}"
                )

        total["timestamps"].append(timestamp + timestamp_offset)
        total["mtmPnl"].append(timestamp_total)

        for product in products:
            for trade in own_trades_by_key.get((timestamp, product), []):
                if trade.buyer == "SUBMISSION":
                    positions[product] += trade.quantity
                    cash[product] -= trade.price * trade.quantity
                elif trade.seller == "SUBMISSION":
                    positions[product] -= trade.quantity
                    cash[product] += trade.price * trade.quantity

    final_product_pnls: dict[str, float] = {}
    for product in products:
        market = last_market[product]
        final_mtm = cash[product] + positions[product] * market["mid"]
        final_product_pnls[product] = final_mtm

        path = paths[product]
        path["timestamps"].append(TERMINAL_TIMESTAMP + timestamp_offset)
        path["fair"].append(market["mid"])
        path["mid"].append(market["mid"])
        path["bid1"].append(market["bid1"])
        path["ask1"].append(market["ask1"])
        path["position"].append(float(positions[product]))
        path["cash"].append(cash[product])
        path["mtmPnl"].append(final_mtm)

        if write_trace:
            trace_lines.append(
                f"{result.day_num};{TERMINAL_TIMESTAMP};{product};{market['mid']};{positions[product]};{cash[product]};{final_mtm}"
            )

    total["timestamps"].append(TERMINAL_TIMESTAMP + timestamp_offset)
    total["mtmPnl"].append(sum(final_product_pnls[product] for product in products))

    return {
        "products": paths,
        "total": total,
        "traceLines": trace_lines,
        "finalProductPnls": final_product_pnls,
        "finalPositions": positions,
        "finalCash": cash,
    }


def first_two_products(product_order: list[str]) -> tuple[str, str]:
    if len(product_order) >= 2:
        return product_order[0], product_order[1]
    if len(product_order) == 1:
        return product_order[0], product_order[0]
    return "EMERALDS", "TOMATOES"


def compatibility_keys(product_order: list[str]) -> dict[str, str]:
    first, second = first_two_products(product_order)
    return {
        "emerald": first,
        "tomato": second,
    }


def session_dir(output_dir: Path, session_id: int) -> Path:
    return output_dir / "sessions" / f"session_{session_id:04d}"


def sample_path_file(output_dir: Path, session_id: int) -> Path:
    return output_dir / "sample_paths" / f"session_{session_id:04d}.json"


def plot_dir(output_dir: Path, session_id: int) -> Path:
    return output_dir / "plots" / f"session_{session_id:04d}"


def write_sample_session_files(
    output_dir: Path,
    round_num: int,
    session_id: int,
    day_outputs: list[dict[str, Any]],
    sample_path: dict[str, Any],
) -> None:
    sample_dir = session_dir(output_dir, session_id)
    round_dir = sample_dir / f"round{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_paths").mkdir(parents=True, exist_ok=True)

    price_header = (
        "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
        "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    )
    trade_header = "timestamp;buyer;seller;symbol;currency;price;quantity"
    trace_header = "day;timestamp;product;fair_value;position;cash;mtm_pnl"

    for day_output in day_outputs:
        day = day_output["day"]
        (round_dir / f"prices_round_{round_num}_day_{day}.csv").write_text(
            price_header + "\n" + "\n".join(day_output["priceLines"]) + "\n",
            encoding="utf-8",
        )
        (round_dir / f"trades_round_{round_num}_day_{day}.csv").write_text(
            trade_header + ("\n" + "\n".join(day_output["tradeLines"]) if day_output["tradeLines"] else "") + "\n",
            encoding="utf-8",
        )
        (round_dir / f"trace_round_{round_num}_day_{day}.csv").write_text(
            trace_header + "\n" + "\n".join(day_output["traceLines"]) + "\n",
            encoding="utf-8",
        )

    sample_path_file(output_dir, session_id).write_text(
        json.dumps(sample_path, separators=(",", ":")),
        encoding="utf-8",
    )


def initialize_worker(
    algorithm: str,
    data_root: str,
    output_dir: str,
    round_num: int,
    source_days: tuple[int, ...],
    session_day_labels: tuple[int, ...],
    block_size: int,
    trade_matching_mode: str,
    sample_sessions: int,
    workers: int,
) -> None:
    global WORKER_CONTEXT
    dataset = load_round_dataset(Path(data_root), round_num, source_days)
    bt_data.LIMITS.update(POSITION_LIMITS)
    WORKER_CONTEXT = WorkerContext(
        algorithm=Path(algorithm),
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        round_num=round_num,
        source_days=source_days,
        session_day_labels=session_day_labels,
        block_size=block_size,
        trade_matching_mode=TradeMatchingMode(trade_matching_mode),
        sample_sessions=sample_sessions,
        workers=workers,
        dataset=dataset,
    )


def combine_paths(destination: dict[str, Any], source: dict[str, Any], products: tuple[str, ...]) -> None:
    for product in products:
        for key, values in source["products"][product].items():
            destination["products"][product][key].extend(values)
    destination["total"]["timestamps"].extend(source["total"]["timestamps"])
    destination["total"]["mtmPnl"].extend(source["total"]["mtmPnl"])


def session_summary_row(
    session_id: int,
    product_order: list[str],
    product_pnls: dict[str, float],
    positions: dict[str, int],
    cash: dict[str, float],
    combined_path: dict[str, Any],
) -> dict[str, Any]:
    compat = compatibility_keys(product_order)
    total_pnl = sum(product_pnls.values())
    total_fit = linear_fit(combined_path["total"]["mtmPnl"])
    row: dict[str, Any] = {
        "sessionId": session_id,
        "totalPnl": total_pnl,
        "productPnls": dict(product_pnls),
        "productPositions": dict(positions),
        "productCash": dict(cash),
        "totalSlopePerStep": total_fit["slopePerStep"],
        "totalR2": total_fit["r2"],
    }

    for product in product_order:
        fit = linear_fit(combined_path["products"][product]["mtmPnl"])
        row.setdefault("productSlopePerStep", {})[product] = fit["slopePerStep"]
        row.setdefault("productR2", {})[product] = fit["r2"]

    row["emeraldPnl"] = product_pnls[compat["emerald"]]
    row["tomatoPnl"] = product_pnls[compat["tomato"]]
    row["emeraldPosition"] = positions[compat["emerald"]]
    row["tomatoPosition"] = positions[compat["tomato"]]
    row["emeraldCash"] = cash[compat["emerald"]]
    row["tomatoCash"] = cash[compat["tomato"]]
    row["emeraldSlopePerStep"] = row["productSlopePerStep"][compat["emerald"]]
    row["tomatoSlopePerStep"] = row["productSlopePerStep"][compat["tomato"]]
    row["emeraldR2"] = row["productR2"][compat["emerald"]]
    row["tomatoR2"] = row["productR2"][compat["tomato"]]

    return row


def run_summary_row(
    session_id: int,
    day_num: int,
    product_order: list[str],
    day_path: dict[str, Any],
) -> dict[str, Any]:
    compat = compatibility_keys(product_order)
    product_pnls = day_path["finalProductPnls"]
    total_pnl = sum(product_pnls.values())
    total_fit = linear_fit(day_path["total"]["mtmPnl"])
    row: dict[str, Any] = {
        "sessionId": session_id,
        "day": day_num,
        "totalPnl": total_pnl,
        "productPnls": dict(product_pnls),
        "totalSlopePerStep": total_fit["slopePerStep"],
        "totalR2": total_fit["r2"],
    }

    for product in product_order:
        fit = linear_fit(day_path["products"][product]["mtmPnl"])
        row.setdefault("productSlopePerStep", {})[product] = fit["slopePerStep"]
        row.setdefault("productR2", {})[product] = fit["r2"]

    row["emeraldPnl"] = product_pnls[compat["emerald"]]
    row["tomatoPnl"] = product_pnls[compat["tomato"]]
    row["emeraldSlopePerStep"] = row["productSlopePerStep"][compat["emerald"]]
    row["tomatoSlopePerStep"] = row["productSlopePerStep"][compat["tomato"]]
    row["emeraldR2"] = row["productR2"][compat["emerald"]]
    row["tomatoR2"] = row["productR2"][compat["tomato"]]
    return row


def run_session_task(task: tuple[int, int]) -> dict[str, Any]:
    if WORKER_CONTEXT is None:
        raise RuntimeError("Monte Carlo worker context is not initialized")

    session_id, seed = task
    ctx = WORKER_CONTEXT
    rng = random.Random(seed)
    keep_session_files = session_id <= ctx.sample_sessions
    product_order = list(ctx.dataset.products)
    trader_module = load_trader_module(ctx.algorithm)
    trader = trader_module.Trader()
    combined_path = {
        "products": {product: empty_product_path() for product in ctx.dataset.products},
        "total": {"timestamps": [], "mtmPnl": []},
    }
    session_product_pnls = {product: 0.0 for product in ctx.dataset.products}
    session_positions = {product: 0 for product in ctx.dataset.products}
    session_cash = {product: 0.0 for product in ctx.dataset.products}
    trader_data = ""
    day_outputs: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    synthetic_mid_anchor: dict[str, float] = {}

    for day_index, day_num in enumerate(ctx.session_day_labels):
        synthetic = bootstrap_day(
            ctx.dataset,
            day_num,
            rng,
            ctx.block_size,
            keep_session_files,
            initial_mid_prices=synthetic_mid_anchor,
        )
        backtest_data = bt_data.create_backtest_data(
            round_num=ctx.round_num,
            day_num=day_num,
            prices=synthetic["priceRows"],
            trades=synthetic["trades"],
            observations=[],
        )
        backtest_output = run_backtest_with_data(
            trader,
            backtest_data,
            ctx.trade_matching_mode,
            initial_trader_data=trader_data,
            initial_positions=session_positions,
            initial_cash=session_cash,
        )
        result = backtest_output.result
        day_path = build_day_path(
            result=result,
            products=ctx.dataset.products,
            timestamp_offset=day_index * DAY_TIMESTAMP_OFFSET,
            initial_positions=session_positions,
            initial_cash=session_cash,
            write_trace=keep_session_files,
        )
        combine_paths(combined_path, day_path, ctx.dataset.products)
        run_rows.append(run_summary_row(session_id, day_num, product_order, day_path))
        synthetic_mid_anchor = synthetic["endingMidPrices"]
        session_product_pnls = dict(day_path["finalProductPnls"])
        session_positions = dict(backtest_output.final_positions)
        session_cash = dict(backtest_output.final_cash)
        trader_data = backtest_output.trader_data

        if keep_session_files:
            day_outputs.append(
                {
                    "day": day_num,
                    "priceLines": synthetic["priceLines"] or [],
                    "tradeLines": synthetic["tradeLines"] or [],
                    "traceLines": day_path["traceLines"],
                }
            )

    session_row = session_summary_row(
        session_id=session_id,
        product_order=product_order,
        product_pnls=session_product_pnls,
        positions=session_positions,
        cash=session_cash,
        combined_path=combined_path,
    )

    if keep_session_files:
        sample_path = {
            "sessionId": session_id,
            "products": combined_path["products"],
            "total": combined_path["total"],
        }
        write_sample_session_files(ctx.output_dir, ctx.round_num, session_id, day_outputs, sample_path)

    return {
        "session": session_row,
        "runs": run_rows,
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return sorted_values[lo]

    weight = index - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def downside_deviation(values: list[float]) -> float:
    downside = [min(value, 0.0) ** 2 for value in values]
    if not downside:
        return 0.0
    return math.sqrt(sum(downside) / len(downside))


def skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = statistics.fmean(values)
    std = sample_std(values)
    if std == 0:
        return 0.0
    return sum(((value - mean) / std) ** 3 for value in values) / len(values)


def correlation(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    std_a = sample_std(a)
    std_b = sample_std(b)
    if std_a == 0 or std_b == 0:
        return 0.0
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (len(a) - 1)
    return cov / (std_a * std_b)


def summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    mean = statistics.fmean(values)
    std = sample_std(values)
    downside = downside_deviation(values)
    q05 = quantile(values, 0.05)
    q01 = quantile(values, 0.01)
    tail_5 = [value for value in values if value <= q05] or [min(values)]
    tail_1 = [value for value in values if value <= q01] or [min(values)]
    ci_half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0

    return {
        "count": float(len(values)),
        "mean": mean,
        "std": std,
        "min": min(values),
        "p01": q01,
        "p05": q05,
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
        "positiveRate": sum(value > 0 for value in values) / len(values),
        "negativeRate": sum(value < 0 for value in values) / len(values),
        "zeroRate": sum(value == 0 for value in values) / len(values),
        "var95": q05,
        "cvar95": statistics.fmean(tail_5),
        "var99": q01,
        "cvar99": statistics.fmean(tail_1),
        "meanConfidenceLow95": mean - ci_half_width,
        "meanConfidenceHigh95": mean + ci_half_width,
        "sharpeLike": mean / std if std > 0 else 0.0,
        "sortinoLike": mean / downside if downside > 0 else 0.0,
        "skewness": skewness(values),
    }


def histogram(values: list[float], bins: int = 40) -> dict[str, list[float] | list[int]]:
    if not values:
        return {"binEdges": [], "counts": []}

    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(int((value - lo) / width), bins - 1)
        counts[idx] += 1

    return {"binEdges": edges, "counts": counts}


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def fit_r_squared(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        return 0.0
    actual_mean = statistics.fmean(actual)
    sst = sum((value - actual_mean) ** 2 for value in actual)
    if sst <= 1e-12:
        return 0.0
    sse = sum((a - b) ** 2 for a, b in zip(actual, predicted))
    return max(0.0, 1.0 - sse / sst)


def normal_fit(values: list[float], bins: int = 40, points: int = 200) -> dict[str, Any]:
    hist = histogram(values, bins)
    bin_edges = hist["binEdges"]
    counts = hist["counts"]
    mu = statistics.fmean(values) if values else 0.0
    sigma = sample_std(values)

    if len(bin_edges) < 2:
        return {"mean": mu, "std": sigma, "r2": 0.0, "line": []}

    bin_width = float(bin_edges[1] - bin_edges[0])
    centers = [(bin_edges[index] + bin_edges[index + 1]) / 2.0 for index in range(len(counts))]
    expected_counts = [normal_pdf(center, mu, sigma) * len(values) * bin_width for center in centers]
    lo = float(bin_edges[0])
    hi = float(bin_edges[-1])
    line = []
    for index in range(max(points, 2)):
        x = lo + (hi - lo) * index / (max(points, 2) - 1)
        y = normal_pdf(x, mu, sigma) * len(values) * bin_width
        line.append([x, y])

    return {
        "mean": mu,
        "std": sigma,
        "r2": fit_r_squared([float(count) for count in counts], expected_counts),
        "line": line,
    }


def linear_regression(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "correlation": 0.0,
            "line": [],
            "diagnosis": "insufficient data",
        }

    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    sxx = sum((x - x_mean) ** 2 for x in x_values)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    corr = correlation(x_values, y_values)
    r2 = corr * corr
    x_min = min(x_values)
    x_max = max(x_values)
    line = [[x_min, intercept + slope * x_min], [x_max, intercept + slope * x_max]]
    strength = abs(corr)
    if strength < 0.1:
        diagnosis = "no meaningful correlation"
    elif strength < 0.3:
        diagnosis = "weak correlation"
    elif strength < 0.6:
        diagnosis = "moderate correlation"
    else:
        diagnosis = "strong correlation"

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "correlation": corr,
        "line": line,
        "diagnosis": diagnosis,
    }


def downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= max_points:
        return list(range(length))

    if max_points <= 1:
        return [length - 1]

    indices = [min(round(i * (length - 1) / (max_points - 1)), length - 1) for i in range(max_points)]
    deduped: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            deduped.append(index)
            seen.add(index)
    if deduped[-1] != length - 1:
        deduped[-1] = length - 1
    return deduped


def downsample_path_node(node: dict[str, list[float] | list[int]], max_points: int) -> dict[str, list[float] | list[int]]:
    indices = downsample_indices(len(node["timestamps"]), max_points)
    return {key: [values[index] for index in indices] for key, values in node.items()}


def load_sample_paths(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs: list[dict[str, Any]] = []
    sample_paths: list[dict[str, Any]] = []
    sample_dir = output_dir / "sample_paths"
    if not sample_dir.exists():
        return refs, sample_paths

    for path in sorted(sample_dir.glob("session_*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        refs.append(
            {
                "sessionId": sample["sessionId"],
                "url": Path("sample_paths", path.name).as_posix(),
            }
        )
        sample_paths.append(sample)

    return refs, sample_paths


def valid_series_bounds(*series_groups: list[float]) -> tuple[float, float] | None:
    values = [value for series in series_groups for value in series if isinstance(value, (int, float)) and math.isfinite(value)]
    if not values:
        return None
    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0)
        lower -= padding
        upper += padding
    return lower, upper


def session_plot_title(sample: dict[str, Any], suffix: str) -> str:
    return f"Session {sample['sessionId']:04d} - {suffix}"


def render_sample_path_plots(
    output_dir: Path,
    sample: dict[str, Any],
    product_order: list[str],
    product_labels: dict[str, str],
) -> list[Path]:
    target_dir = plot_dir(output_dir, int(sample["sessionId"]))
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    timestamps = sample["total"]["timestamps"]
    total_pnl = sample["total"]["mtmPnl"]

    plt.style.use("default")

    total_path = target_dir / "total_pnl.png"
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, total_pnl, color="#1d4ed8", linewidth=1.8)
    ax.set_title(session_plot_title(sample, "Total MTM PnL"))
    ax.set_xlabel("Synthetic timestamp")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(total_path, dpi=160)
    plt.close(fig)
    saved_paths.append(total_path)

    for product in product_order:
        label = product_labels.get(product, product)
        node = sample["products"][product]

        product_path = target_dir / f"{product.lower()}_overview.png"
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

        axes[0].plot(node["timestamps"], node["mid"], label="mid", color="#111827", linewidth=1.5)
        if any(math.isfinite(value) for value in node["bid1"]):
            axes[0].plot(node["timestamps"], node["bid1"], label="bid1", color="#059669", linewidth=1.0, alpha=0.85)
        if any(math.isfinite(value) for value in node["ask1"]):
            axes[0].plot(node["timestamps"], node["ask1"], label="ask1", color="#dc2626", linewidth=1.0, alpha=0.85)
        if any(math.isfinite(value) for value in node["fair"]):
            axes[0].plot(node["timestamps"], node["fair"], label="fair proxy", color="#2563eb", linewidth=1.0, alpha=0.65)
        axes[0].set_title(session_plot_title(sample, f"{label} price path"))
        axes[0].set_ylabel("Price")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(node["timestamps"], node["position"], color="#7c3aed", linewidth=1.4)
        axes[1].axhline(0, color="#9ca3af", linewidth=1.0, linestyle="--")
        axes[1].set_title(f"{label} position")
        axes[1].set_ylabel("Position")
        axes[1].grid(alpha=0.25)

        axes[2].plot(node["timestamps"], node["mtmPnl"], color="#ea580c", linewidth=1.5)
        axes[2].axhline(0, color="#9ca3af", linewidth=1.0, linestyle="--")
        axes[2].set_title(f"{label} MTM PnL")
        axes[2].set_xlabel("Synthetic timestamp")
        axes[2].set_ylabel("PnL")
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(product_path, dpi=160)
        plt.close(fig)
        saved_paths.append(product_path)

    return saved_paths


def write_sample_path_plots(output_dir: Path, product_order: list[str]) -> list[Path]:
    _, sample_paths = load_sample_paths(output_dir)
    if not sample_paths:
        return []

    product_labels = {product: product for product in product_order}
    saved_paths: list[Path] = []
    for sample in sample_paths:
        saved_paths.extend(render_sample_path_plots(output_dir, sample, product_order, product_labels))
    return saved_paths


def mean_std_band_series(sample_paths: list[dict[str, Any]], value_getter) -> dict[str, list[float]]:
    if not sample_paths:
        return {}

    base_values = value_getter(sample_paths[0])
    indices = downsample_indices(len(base_values), STATIC_CHART_POINTS)
    timestamps = [sample_paths[0]["total"]["timestamps"][index] for index in indices]

    mean_values: list[float] = []
    std1_low: list[float] = []
    std1_high: list[float] = []
    std3_low: list[float] = []
    std3_high: list[float] = []

    for index in indices:
        values = [value_getter(path)[index] for path in sample_paths]
        mu = statistics.fmean(values)
        sigma = sample_std(values)
        mean_values.append(mu)
        std1_low.append(mu - sigma)
        std1_high.append(mu + sigma)
        std3_low.append(mu - 3.0 * sigma)
        std3_high.append(mu + 3.0 * sigma)

    return {
        "timestamps": timestamps,
        "mean": mean_values,
        "std1Low": std1_low,
        "std1High": std1_high,
        "std3Low": std3_low,
        "std3High": std3_high,
    }


def build_band_series(sample_paths: list[dict[str, Any]], product_order: list[str]) -> dict[str, dict[str, dict[str, list[float]]]]:
    if not sample_paths:
        return {}

    series: dict[str, dict[str, dict[str, list[float]]]] = {}
    for product in product_order:
        series[product] = {
            "fair": mean_std_band_series(sample_paths, lambda path, product=product: path["products"][product]["fair"]),
            "mtmPnl": mean_std_band_series(sample_paths, lambda path, product=product: path["products"][product]["mtmPnl"]),
            "position": mean_std_band_series(sample_paths, lambda path, product=product: path["products"][product]["position"]),
        }
    return series


def write_csv_summaries(output_dir: Path, session_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], product_order: list[str]) -> None:
    compat = compatibility_keys(product_order)
    session_fields = [
        "session_id",
        "total_pnl",
        "primary_pnl",
        "secondary_pnl",
        "primary_position",
        "secondary_position",
        "primary_cash",
        "secondary_cash",
        "total_slope_per_step",
        "total_r2",
        "primary_slope_per_step",
        "primary_r2",
        "secondary_slope_per_step",
        "secondary_r2",
        "product_order",
        "product_pnls_json",
        "product_positions_json",
        "product_cash_json",
        "product_slope_per_step_json",
        "product_r2_json",
    ]
    with (output_dir / "session_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=session_fields)
        writer.writeheader()
        for row in session_rows:
            writer.writerow(
                {
                    "session_id": row["sessionId"],
                    "total_pnl": row["totalPnl"],
                    "primary_pnl": row["productPnls"][compat["emerald"]],
                    "secondary_pnl": row["productPnls"][compat["tomato"]],
                    "primary_position": row["productPositions"][compat["emerald"]],
                    "secondary_position": row["productPositions"][compat["tomato"]],
                    "primary_cash": row["productCash"][compat["emerald"]],
                    "secondary_cash": row["productCash"][compat["tomato"]],
                    "total_slope_per_step": row["totalSlopePerStep"],
                    "total_r2": row["totalR2"],
                    "primary_slope_per_step": row["productSlopePerStep"][compat["emerald"]],
                    "primary_r2": row["productR2"][compat["emerald"]],
                    "secondary_slope_per_step": row["productSlopePerStep"][compat["tomato"]],
                    "secondary_r2": row["productR2"][compat["tomato"]],
                    "product_order": json.dumps(product_order),
                    "product_pnls_json": json.dumps(row["productPnls"], separators=(",", ":")),
                    "product_positions_json": json.dumps(row["productPositions"], separators=(",", ":")),
                    "product_cash_json": json.dumps(row["productCash"], separators=(",", ":")),
                    "product_slope_per_step_json": json.dumps(row["productSlopePerStep"], separators=(",", ":")),
                    "product_r2_json": json.dumps(row["productR2"], separators=(",", ":")),
                }
            )

    run_fields = [
        "session_id",
        "day",
        "total_pnl",
        "primary_pnl",
        "secondary_pnl",
        "total_slope_per_step",
        "total_r2",
        "primary_slope_per_step",
        "primary_r2",
        "secondary_slope_per_step",
        "secondary_r2",
        "product_pnls_json",
        "product_slope_per_step_json",
        "product_r2_json",
    ]
    with (output_dir / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(
                {
                    "session_id": row["sessionId"],
                    "day": row["day"],
                    "total_pnl": row["totalPnl"],
                    "primary_pnl": row["productPnls"][compat["emerald"]],
                    "secondary_pnl": row["productPnls"][compat["tomato"]],
                    "total_slope_per_step": row["totalSlopePerStep"],
                    "total_r2": row["totalR2"],
                    "primary_slope_per_step": row["productSlopePerStep"][compat["emerald"]],
                    "primary_r2": row["productR2"][compat["emerald"]],
                    "secondary_slope_per_step": row["productSlopePerStep"][compat["tomato"]],
                    "secondary_r2": row["productR2"][compat["tomato"]],
                    "product_pnls_json": json.dumps(row["productPnls"], separators=(",", ":")),
                    "product_slope_per_step_json": json.dumps(row["productSlopePerStep"], separators=(",", ":")),
                    "product_r2_json": json.dumps(row["productR2"], separators=(",", ":")),
                }
            )


def write_run_log(
    output_dir: Path,
    algorithm: Path,
    round_num: int,
    sessions: int,
    sample_sessions: int,
    block_size: int,
    seed: int,
    workers: int,
    trade_matching_mode: str,
    day_rows: list[dict[str, Any]],
) -> None:
    total_values = [row["totalPnl"] for row in day_rows]
    summary = summarize_distribution(total_values)
    lines = [
        f"algorithm={algorithm}",
        f"round={round_num}",
        f"sessions={sessions}",
        f"sample_sessions={sample_sessions}",
        f"block_size={block_size}",
        f"seed={seed}",
        f"workers={workers}",
        f"trade_matching_mode={trade_matching_mode}",
        f"mean_total_pnl={summary.get('mean', 0.0):.2f}",
        f"p05_total_pnl={summary.get('p05', 0.0):.2f}",
        f"p95_total_pnl={summary.get('p95', 0.0):.2f}",
    ]
    (output_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dashboard(
    output_dir: Path,
    algorithm: Path,
    round_num: int,
    sessions: int,
    sample_sessions: int,
    source_days: list[int],
    session_day_count: int,
    block_size: int,
    seed: int,
    workers: int,
    trade_matching_mode: str,
    product_order: list[str],
    session_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    compat = compatibility_keys(product_order)
    product_display_names = {product: product for product in product_order}
    product_colors = {product: PRODUCT_COLORS.get(product, "#4c6ef5") for product in product_order}

    total = [row["totalPnl"] for row in session_rows]
    product_pnls = {
        product: [row["productPnls"][product] for row in session_rows]
        for product in product_order
    }
    product_positions = {
        product: [float(row["productPositions"][product]) for row in session_rows]
        for product in product_order
    }
    product_cash = {
        product: [row["productCash"][product] for row in session_rows]
        for product in product_order
    }
    total_profitability = [row["totalSlopePerStep"] for row in run_rows]
    total_stability = [row["totalR2"] for row in run_rows]
    product_profitability = {
        product: [row["productSlopePerStep"][product] for row in run_rows]
        for product in product_order
    }
    product_stability = {
        product: [row["productR2"][product] for row in run_rows]
        for product in product_order
    }
    session_total_profitability = [row["totalSlopePerStep"] for row in session_rows]
    session_total_stability = [row["totalR2"] for row in session_rows]
    session_product_profitability = {
        product: [row["productSlopePerStep"][product] for row in session_rows]
        for product in product_order
    }
    session_product_stability = {
        product: [row["productR2"][product] for row in session_rows]
        for product in product_order
    }

    sample_path_refs, sample_paths = load_sample_paths(output_dir)
    band_series = build_band_series(sample_paths, product_order) if sample_paths else {}

    runs_by_session: dict[int, list[dict[str, Any]]] = {}
    for run in run_rows:
        runs_by_session.setdefault(run["sessionId"], []).append(run)

    for row in session_rows:
        session_runs = runs_by_session.get(row["sessionId"], [])
        if session_runs:
            row["runMeanTotalSlopePerStep"] = statistics.fmean(run["totalSlopePerStep"] for run in session_runs)
            row["runMeanTotalR2"] = statistics.fmean(run["totalR2"] for run in session_runs)
        else:
            row["runMeanTotalSlopePerStep"] = row["totalSlopePerStep"]
            row["runMeanTotalR2"] = row["totalR2"]

    top_sessions = sorted(session_rows, key=lambda row: row["totalPnl"], reverse=True)[:10]
    bottom_sessions = sorted(session_rows, key=lambda row: row["totalPnl"])[:10]
    x_product, y_product = first_two_products(product_order)
    scatter_fit = linear_regression(product_pnls[x_product], product_pnls[y_product])
    scatter_fit["xProduct"] = x_product
    scatter_fit["yProduct"] = y_product

    overall = {
        "totalPnl": summarize_distribution(total),
        "productPnl": {product: summarize_distribution(values) for product, values in product_pnls.items()},
        "productCorrelations": {
            f"{x_product}|{y_product}": correlation(product_pnls[x_product], product_pnls[y_product])
        },
        "emeraldPnl": summarize_distribution(product_pnls[compat["emerald"]]),
        "tomatoPnl": summarize_distribution(product_pnls[compat["tomato"]]),
        "emeraldTomatoCorrelation": correlation(product_pnls[compat["emerald"]], product_pnls[compat["tomato"]]),
    }

    trend_fits = {
        "TOTAL": {
            "profitability": summarize_distribution(total_profitability),
            "stability": summarize_distribution(total_stability),
        }
    }
    aggregate_trend_fits = {
        "TOTAL": {
            "profitability": summarize_distribution(session_total_profitability),
            "stability": summarize_distribution(session_total_stability),
        }
    }
    for product in product_order:
        trend_fits[product] = {
            "profitability": summarize_distribution(product_profitability[product]),
            "stability": summarize_distribution(product_stability[product]),
        }
        aggregate_trend_fits[product] = {
            "profitability": summarize_distribution(session_product_profitability[product]),
            "stability": summarize_distribution(session_product_stability[product]),
        }

    normal_fits = {
        "totalPnl": normal_fit(total),
        "productPnl": {product: normal_fit(values) for product, values in product_pnls.items()},
        "emeraldPnl": normal_fit(product_pnls[compat["emerald"]]),
        "tomatoPnl": normal_fit(product_pnls[compat["tomato"]]),
    }

    products = {
        product: {
            "pnl": summarize_distribution(product_pnls[product]),
            "finalPosition": summarize_distribution(product_positions[product]),
            "cash": summarize_distribution(product_cash[product]),
        }
        for product in product_order
    }

    histograms = {
        "totalPnl": histogram(total),
        "productPnl": {product: histogram(values) for product, values in product_pnls.items()},
        "totalProfitability": histogram(total_profitability),
        "totalStability": histogram(total_stability),
        "productProfitability": {product: histogram(values) for product, values in product_profitability.items()},
        "productStability": {product: histogram(values) for product, values in product_stability.items()},
        "emeraldPnl": histogram(product_pnls[compat["emerald"]]),
        "tomatoPnl": histogram(product_pnls[compat["tomato"]]),
        "emeraldProfitability": histogram(product_profitability[compat["emerald"]]),
        "tomatoProfitability": histogram(product_profitability[compat["tomato"]]),
        "emeraldStability": histogram(product_stability[compat["emerald"]]),
        "tomatoStability": histogram(product_stability[compat["tomato"]]),
    }

    generator_model = {
        product: {
            "name": "Empirical Block Bootstrap",
            "formula": f"Sample time-aligned {block_size}-tick blocks from round {round_num} history",
            "notes": [
                f"Historical {product} order books and market trades are resampled in contiguous, time-aligned blocks",
                "Cross-product timing is preserved because both products are copied from the same historical ticks",
            ],
        }
        for product in product_order
    }

    return {
        "kind": "monte_carlo_dashboard",
        "meta": {
            "algorithmPath": str(algorithm),
            "sessionCount": sessions,
            "sampleSessions": sample_sessions,
            "bandSessionCount": len(sample_paths),
            "round": round_num,
            "sourceDays": source_days,
            "sessionDayCount": session_day_count,
            "simulationModel": "time_aligned_block_bootstrap",
            "blockSize": block_size,
            "seed": seed,
            "workers": workers,
            "tradeMatchingMode": trade_matching_mode,
            "productOrder": product_order,
            "productDisplayNames": product_display_names,
            "productColors": product_colors,
        },
        "overall": overall,
        "trendFits": trend_fits,
        "aggregateTrendFits": aggregate_trend_fits,
        "normalFits": normal_fits,
        "scatterFit": scatter_fit,
        "generatorModel": generator_model,
        "products": products,
        "histograms": histograms,
        "sessions": session_rows,
        "runs": run_rows,
        "topSessions": top_sessions,
        "bottomSessions": bottom_sessions,
        "samplePaths": [],
        "samplePathRefs": sample_path_refs,
        "bandSeries": band_series,
    }


def run_monte_carlo_mode(
    algorithm: Path,
    dashboard_path: Path,
    data_root: Optional[Path],
    round_num: int,
    sessions: int,
    sample_sessions: int,
    session_days: Optional[int],
    block_size: int,
    seed: int,
    workers: int,
    trade_matching_mode: str,
    plot_samples: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    algorithm = algorithm.resolve()
    output_dir = dashboard_path.parent.resolve()
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
    dataset = load_round_dataset(data_root, round_num, source_days)
    if session_days is None:
        session_day_labels = source_days
    else:
        if session_days <= 0:
            raise ValueError("--session-days must be positive")
        session_day_labels = tuple(range(session_days))

    session_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    seeds = [(session_id, seed + session_id * 9973) for session_id in range(1, sessions + 1)]

    if workers <= 1:
        initialize_worker(
            algorithm=str(algorithm),
            data_root=str(data_root),
            output_dir=str(output_dir),
            round_num=round_num,
            source_days=source_days,
            session_day_labels=session_day_labels,
            block_size=block_size,
            trade_matching_mode=trade_matching_mode,
            sample_sessions=sample_sessions,
            workers=1,
        )
        iterator = tqdm(seeds, disable=not show_progress, desc="Monte Carlo sessions", ascii=True)
        for task in iterator:
            result = run_session_task(task)
            session_rows.append(result["session"])
            run_rows.extend(result["runs"])
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(
                str(algorithm),
                str(data_root),
                str(output_dir),
                round_num,
                source_days,
                session_day_labels,
                block_size,
                trade_matching_mode,
                sample_sessions,
                workers,
            ),
        ) as executor:
            futures = {executor.submit(run_session_task, task): task[0] for task in seeds}
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Monte Carlo sessions", ascii=True)
            for future in iterator:
                result = future.result()
                session_rows.append(result["session"])
                run_rows.extend(result["runs"])

    session_rows.sort(key=lambda row: row["sessionId"])
    run_rows.sort(key=lambda row: (row["sessionId"], row["day"]))

    write_csv_summaries(output_dir, session_rows, run_rows, list(dataset.products))
    write_run_log(
        output_dir=output_dir,
        algorithm=algorithm,
        round_num=round_num,
        sessions=sessions,
        sample_sessions=sample_sessions,
        block_size=block_size,
        seed=seed,
        workers=workers,
        trade_matching_mode=trade_matching_mode,
        day_rows=session_rows,
    )

    plot_files: list[Path] = []
    if plot_samples:
        plot_files = write_sample_path_plots(output_dir, list(dataset.products))

    dashboard = build_dashboard(
        output_dir=output_dir,
        algorithm=algorithm,
        round_num=round_num,
        sessions=sessions,
        sample_sessions=sample_sessions,
        source_days=list(source_days),
        session_day_count=len(session_day_labels),
        block_size=block_size,
        seed=seed,
        workers=workers,
        trade_matching_mode=trade_matching_mode,
        product_order=list(dataset.products),
        session_rows=session_rows,
        run_rows=run_rows,
    )
    with dashboard_path.open("w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, indent=2)

    if plot_files:
        dashboard.setdefault("meta", {})["plotFiles"] = [str(path) for path in plot_files]
        with dashboard_path.open("w", encoding="utf-8") as handle:
            json.dump(dashboard, handle, indent=2)

    return dashboard


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run empirical block-bootstrap Monte Carlo sessions for round 1 or round 2."
    )
    parser.add_argument("round", type=int, help="Round number to simulate, for example 1 or 2.")
    parser.add_argument("--algorithm", default=str(REPO_ROOT / "src" / "trader.py"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--sample-sessions", type=int, default=DEFAULT_SAMPLE_SESSIONS)
    parser.add_argument(
        "--session-days",
        type=int,
        help="Number of synthetic day-equivalents inside each session. Defaults to the count of real source days for the round.",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=worker_default_count())
    parser.add_argument(
        "--match-trades",
        choices=[mode.value for mode in TradeMatchingMode],
        default=TradeMatchingMode.worse.value,
    )
    parser.add_argument(
        "--plot-samples",
        action="store_true",
        help="Render PNG plots for each persisted sample session under output_dir/plots/.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick and args.heavy:
        raise SystemExit("--quick and --heavy are mutually exclusive")

    sessions = args.sessions
    sample_sessions = args.sample_sessions
    if args.quick:
        sessions = QUICK_SESSIONS
        sample_sessions = QUICK_SAMPLE_SESSIONS
    elif args.heavy:
        sessions = HEAVY_SESSIONS
        sample_sessions = HEAVY_SAMPLE_SESSIONS

    dashboard_path = normalize_dashboard_path(args.out, False)
    if dashboard_path is None:
        raise SystemExit("dashboard output path could not be resolved")

    dashboard = run_monte_carlo_mode(
        algorithm=Path(args.algorithm),
        dashboard_path=dashboard_path,
        data_root=Path(args.data_root),
        round_num=args.round,
        sessions=sessions,
        sample_sessions=sample_sessions,
        session_days=args.session_days,
        block_size=args.block_size,
        seed=args.seed,
        workers=max(1, args.workers),
        trade_matching_mode=args.match_trades,
        plot_samples=args.plot_samples,
        show_progress=not args.no_progress,
    )

    total_stats = dashboard["overall"]["totalPnl"]
    print(f"Round: {args.round}")
    print(f"Sessions: {int(total_stats['count'])}")
    print(f"Mean total PnL: {total_stats['mean']:,.2f}")
    print(f"Std total PnL: {total_stats['std']:,.2f}")
    print(f"Median total PnL: {total_stats['p50']:,.2f}")
    print(f"5%-95% range: {total_stats['p05']:,.2f} to {total_stats['p95']:,.2f}")
    print(f"Saved Monte Carlo dashboard to {format_path(dashboard_path)}")
    if dashboard.get("meta", {}).get("plotFiles"):
        print("Saved sample plots:")
        for plot_file in dashboard["meta"]["plotFiles"]:
            print(f"  {format_path(Path(plot_file))}")

    if args.vis:
        from prosperity4mcbt.open import open_dashboard

        open_dashboard(dashboard_path)


if __name__ == "__main__":
    main()
