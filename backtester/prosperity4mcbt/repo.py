from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_REPLAY_RESULTS_DIR = REPO_ROOT / "backtests" / "results" / "mcbt_replay"
DEFAULT_MONTE_CARLO_RESULTS_DIR = REPO_ROOT / "backtests" / "results" / "mcbt_monte_carlo"
DEFAULT_PBO_RESULTS_DIR = REPO_ROOT / "backtests" / "results" / "mcbt_pbo"
POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import prosperity3bt.data as bt_data
from prosperity3bt.file_reader import FileReader

bt_data.LIMITS.update(POSITION_LIMITS)


@contextmanager
def wrap_in_context_manager(value) -> Iterator[Path | None]:
    yield value


class RepoFileReader(FileReader):
    def __init__(self, root: Path) -> None:
        self._root = root

    def _candidate_roots(self) -> list[Path]:
        roots = [self._root]

        raw_root = self._root / "raw"
        if raw_root.is_dir():
            roots.append(raw_root)

        return roots

    def _candidate_first_parts(self, first_part: str) -> list[str]:
        candidates = [first_part]

        round_match = re.fullmatch(r"round(\d+)", first_part, flags=re.IGNORECASE)
        if round_match:
            candidates.append(f"ROUND_{int(round_match.group(1))}")

        return candidates

    def file(self, path_parts: list[str]):
        if len(path_parts) == 0:
            return wrap_in_context_manager(None)

        first_part, *remaining_parts = path_parts
        for root in self._candidate_roots():
            for candidate_first_part in self._candidate_first_parts(first_part):
                file = root / candidate_first_part
                for part in remaining_parts:
                    file = file / part

                if file.is_file():
                    return wrap_in_context_manager(file)

        return wrap_in_context_manager(None)


def trader_module_name(algorithm: Path) -> str:
    digest = hashlib.sha1(str(algorithm.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"prosperity4mcbt_algo_{algorithm.stem}_{digest}"


def load_trader_module(algorithm: Path):
    algorithm = algorithm.resolve()
    algorithm_dir = str(algorithm.parent)
    if algorithm_dir not in sys.path:
        sys.path.insert(0, algorithm_dir)

    module_name = trader_module_name(algorithm)
    spec = importlib.util.spec_from_file_location(module_name, algorithm)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {algorithm}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_days(file_reader: RepoFileReader, round_num: int, requested_days: list[int] | None) -> list[int]:
    if requested_days:
        available_days = []
        for day_num in requested_days:
            if not bt_data.has_day_data(file_reader, round_num, day_num):
                raise ValueError(f"No data found for round {round_num} day {day_num}")
            available_days.append(day_num)
        return available_days

    available_days = [day_num for day_num in range(-5, 100) if bt_data.has_day_data(file_reader, round_num, day_num)]
    if not available_days:
        raise ValueError(f"No data found for round {round_num}")
    return available_days


def format_path(path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        return str(path.relative_to(REPO_ROOT))
    return str(path)
