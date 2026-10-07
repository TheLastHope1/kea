"""The strategy contract and the signal helpers strategies share."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from kea.data.history import PriceHistory


class Strategy(ABC):
    """Maps the past to long-only target weights.

    A strategy is called with `history.until(day)` and must return weights for its
    risky symbols: non-negative and summing to at most 1. Whatever is left over is
    held in cash (or the cash proxy). Strategies never see the account, prices
    after `day`, or each other.
    """

    name: str = "strategy"

    def __init__(self, symbols: Sequence[str], cash_symbol: str | None, periods_per_year: int):
        self.symbols = list(symbols)
        self.cash_symbol = cash_symbol
        self.periods_per_year = periods_per_year

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars of history needed before the first meaningful decision."""

    @abstractmethod
    def target_weights(self, history: PriceHistory) -> pd.Series:
        """Target weights indexed by `self.symbols`."""

    def _empty(self) -> pd.Series:
        return pd.Series(0.0, index=self.symbols)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def trailing_return(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.Series:
    """Return from `lookback` bars ago to `skip` bars ago (NaN without enough data)."""
    if len(close) <= lookback:
        return pd.Series(np.nan, index=close.columns)
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - lookback]
    return end / start - 1.0


def trailing_vol(close: pd.DataFrame, lookback: int, periods_per_year: int) -> pd.Series:
    """Annualised volatility of daily returns over the last `lookback` bars."""
    returns = close.iloc[-(lookback + 1) :].pct_change(fill_method=None).iloc[1:]
    vol = returns.std(ddof=1) * math.sqrt(periods_per_year)
    return vol.where(returns.count() >= max(lookback * 0.8, 2))


def inverse_vol_weights(vol: pd.Series, floor: float = 0.02) -> pd.Series:
    """Risk-parity-lite: weight each asset by 1/vol, normalised to sum to 1."""
    inv = 1.0 / vol.dropna().clip(lower=floor)
    if inv.empty:
        return inv
    return inv / inv.sum()


def cash_return(
    close: pd.DataFrame, cash_symbol: str | None, lookback: int, skip: int = 0
) -> float:
    """What the cash proxy earned over the same window (0 without a proxy)."""
    if cash_symbol is None or cash_symbol not in close.columns:
        return 0.0
    value = trailing_return(close[[cash_symbol]], lookback, skip).iloc[0]
    return 0.0 if pd.isna(value) else float(value)


def eligible(history: PriceHistory, symbols: Sequence[str], min_bars: int) -> list[str]:
    available = set(history.available(min_bars))
    return [s for s in symbols if s in available]
