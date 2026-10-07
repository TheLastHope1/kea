"""Rule-based strategies with decades of out-of-sample evidence behind them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from kea.config import FilterConfig, MomentumConfig, TrendConfig
from kea.data.history import PriceHistory
from kea.strategies.base import (
    Strategy,
    cash_return,
    eligible,
    inverse_vol_weights,
    trailing_return,
    trailing_vol,
)


class TrendFollowing(Strategy):
    """Multi-horizon time-series momentum, risk-weighted.

    Each asset gets a conviction in [0, 1]: the fraction of lookbacks (1, 3, 6 and
    12 months by default) over which it beat cash. Conviction scales an
    inverse-volatility allocation, so an asset in a clear uptrend gets its full
    risk budget and one in a downtrend gets nothing. Averaging over horizons
    avoids betting everything on a single, possibly lucky, parameter.
    """

    name = "trend"

    def __init__(
        self,
        symbols: Sequence[str],
        cash_symbol: str | None,
        periods_per_year: int,
        config: TrendConfig,
    ) -> None:
        super().__init__(symbols, cash_symbol, periods_per_year)
        self.config = config

    @property
    def warmup(self) -> int:
        return max(*self.config.lookbacks, self.config.vol_lookback) + 1

    def convictions(self, history: PriceHistory) -> pd.Series:
        assets = eligible(history, self.symbols, self.warmup)
        close = history.close
        votes = [
            trailing_return(close[assets], n) > cash_return(close, self.cash_symbol, n)
            for n in self.config.lookbacks
        ]
        return pd.concat(votes, axis=1).mean(axis=1) if assets else pd.Series(dtype=float)

    def target_weights(self, history: PriceHistory) -> pd.Series:
        conviction = self.convictions(history)
        if conviction.empty:
            return self._empty()
        vol = trailing_vol(
            history.close[conviction.index], self.config.vol_lookback, self.periods_per_year
        )
        weights = inverse_vol_weights(vol) * conviction
        return weights.reindex(self.symbols).fillna(0.0)


class DualMomentum(Strategy):
    """Hold the top-k assets by 12-1 month momentum, but only if they beat cash.

    Relative momentum picks the leaders; the absolute filter sends their slot to
    cash when even the leaders are falling, which is what kept classic dual
    momentum out of most of 2008.
    """

    name = "momentum"

    def __init__(
        self,
        symbols: Sequence[str],
        cash_symbol: str | None,
        periods_per_year: int,
        config: MomentumConfig,
    ) -> None:
        super().__init__(symbols, cash_symbol, periods_per_year)
        self.config = config

    @property
    def warmup(self) -> int:
        return self.config.lookback + 1

    def scores(self, history: PriceHistory) -> pd.Series:
        assets = eligible(history, self.symbols, self.warmup)
        cfg = self.config
        return trailing_return(history.close[assets], cfg.lookback, cfg.skip).dropna()

    def target_weights(self, history: PriceHistory) -> pd.Series:
        cfg = self.config
        scores = self.scores(history)
        hurdle = cash_return(history.close, self.cash_symbol, cfg.lookback, cfg.skip)
        leaders = scores[scores > hurdle].nlargest(cfg.top_k)
        weights = self._empty()
        weights[leaders.index] = 1.0 / cfg.top_k
        return weights


class TrendFilter(Strategy):
    """Hold one market while it trades above its moving average; otherwise hold cash.

    The classic retail trend rule (Faber 2007, "A Quantitative Approach to Tactical
    Asset Allocation"). It gives up a little return in bull markets in exchange for
    stepping aside during long bear markets, and it trades only a few times a year,
    which matters when every order costs a flat fee.
    """

    name = "trend_filter"

    def __init__(
        self,
        symbols: Sequence[str],
        cash_symbol: str | None,
        periods_per_year: int,
        config: FilterConfig,
        asset: str,
    ) -> None:
        super().__init__(symbols, cash_symbol, periods_per_year)
        if asset not in self.symbols:
            raise ValueError(f"trend_filter asset {asset} must be one of the universe symbols")
        self.config = config
        self.asset = asset

    @property
    def warmup(self) -> int:
        return self.config.sma_days + 1

    def target_weights(self, history: PriceHistory) -> pd.Series:
        weights = self._empty()
        close = history.close[self.asset].dropna()
        if len(close) < self.config.sma_days or pd.isna(history.close[self.asset].iloc[-1]):
            return weights
        average = close.iloc[-self.config.sma_days :].mean()
        if close.iloc[-1] > average:
            weights[self.asset] = 1.0
        return weights


class StaticWeights(Strategy):
    """Fixed weights, e.g. buy-and-hold SPY or a 60/40 portfolio. Used as benchmarks."""

    def __init__(self, name: str, weights: Mapping[str, float], periods_per_year: int) -> None:
        super().__init__(list(weights), None, periods_per_year)
        self.name = name
        self.weights = pd.Series(weights, dtype=float)
        if (self.weights < 0).any() or self.weights.sum() > 1 + 1e-9:
            raise ValueError("static weights must be non-negative and sum to at most 1")

    @property
    def warmup(self) -> int:
        return 1

    def target_weights(self, history: PriceHistory) -> pd.Series:
        live = set(history.available())
        return self.weights.where([s in live for s in self.weights.index], 0.0)
