"""Shared fixtures: deterministic synthetic markets, so tests never touch the network."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from kea.config import (
    BrokerConfig,
    Config,
    DataConfig,
    ExecutionConfig,
    UniverseConfig,
)
from kea.data.history import PriceHistory


def make_bars(
    drift: float,
    vol: float,
    dates: pd.DatetimeIndex,
    seed: int,
    start_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
    gap = np.exp(rng.normal(0, vol / 4, len(dates)))
    open_ = np.r_[start_price, close[:-1]] * gap
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1e6,
        },
        index=dates,
    )


def make_history(
    drifts: Mapping[str, float],
    days: int = 800,
    vol: float | Mapping[str, float] = 0.01,
    start: str = "2015-01-01",
    seed: int = 1,
) -> PriceHistory:
    """Random walks with a chosen daily drift (and optionally vol) per symbol.

    "BIL" defaults to near-zero volatility, like the T-bill fund it stands in for.
    """
    dates = pd.bdate_range(start, periods=days)

    def sigma(symbol: str) -> float:
        if isinstance(vol, Mapping):
            return vol.get(symbol, 0.01)
        return 0.0003 if symbol == "BIL" else vol

    return PriceHistory.from_bars(
        {s: make_bars(d, sigma(s), dates, seed + i) for i, (s, d) in enumerate(drifts.items())}
    )


DRIFTS = {"UP": 0.0015, "FLAT": 0.0, "DOWN": -0.0015, "SPY": 0.0004, "BIL": 0.0001}
VOLS = {"UP": 0.006, "FLAT": 0.012, "DOWN": 0.006, "SPY": 0.01, "BIL": 0.0003}


@pytest.fixture
def history() -> PriceHistory:
    return make_history(DRIFTS, vol=VOLS)


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        universe=UniverseConfig(
            symbols=("UP", "FLAT", "DOWN", "SPY"), cash_symbol="BIL", benchmark="SPY"
        ),
        data=DataConfig(cache_dir=tmp_path / "cache"),
        execution=ExecutionConfig(slippage_bps=5.0),
        broker=BrokerConfig(initial_cash=100_000.0, state_dir=tmp_path / "state"),
    )
