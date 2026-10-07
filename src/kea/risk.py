"""Risk: what Kea is willing to hold, and when it refuses to trade at all.

Two layers:

* `RiskManager` shapes every target portfolio: long-only, no leverage, a cap per
  asset, and a volatility cap that cuts exposure when markets get violent.
* Circuit breakers (`breaker_reason`, `data_problems`) stop the autonomous agent
  outright on a deep drawdown, a large one-day loss, or suspicious data, and
  wait for a human to look before trading again.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kea.config import RiskConfig
from kea.data.history import PriceHistory


@dataclass(frozen=True)
class RiskDecision:
    """Final weights over the risky symbols and the cash proxy, plus how they were shaped."""

    weights: pd.Series
    raw_vol: float
    predicted_vol: float
    scale: float
    capped: tuple[str, ...]
    cash_symbol: str | None = None

    @property
    def exposure(self) -> float:
        """Fraction of equity in risky assets (the cash proxy does not count)."""
        return float(self.weights.drop(labels=[self.cash_symbol], errors="ignore").sum())


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        symbols: Sequence[str],
        cash_symbol: str | None,
        periods_per_year: int,
    ) -> None:
        self.config = config
        self.symbols = list(symbols)
        self.cash_symbol = cash_symbol
        self.periods_per_year = periods_per_year

    def apply(self, raw: pd.Series, history: PriceHistory) -> RiskDecision:
        """Final weights over the risky symbols plus the cash proxy (summing to 1 with one)."""
        cfg = self.config
        weights = raw.reindex(self.symbols).fillna(0.0).clip(lower=0.0)
        tradable = set(history.available())
        weights[[s not in tradable for s in weights.index]] = 0.0

        total = weights.sum()
        if total > cfg.max_gross:
            weights *= cfg.max_gross / total
        capped = tuple(weights.index[weights > cfg.max_weight])
        weights = weights.clip(upper=cfg.max_weight)

        raw_vol = self.predicted_vol(weights, history)
        scale = min(1.0, cfg.target_vol / raw_vol) if raw_vol > 0 else 1.0
        weights *= scale

        if self.cash_symbol is not None:
            cash_weight = 0.0 if self.cash_symbol not in tradable else max(0.0, 1.0 - weights.sum())
            weights[self.cash_symbol] = cash_weight
        return RiskDecision(
            weights=weights,
            raw_vol=raw_vol,
            predicted_vol=raw_vol * scale,
            scale=scale,
            capped=capped,
            cash_symbol=self.cash_symbol,
        )

    def predicted_vol(self, weights: pd.Series, history: PriceHistory) -> float:
        """Annualised volatility of `weights`, from a shrunk recent covariance matrix."""
        held = weights[weights > 0]
        if held.empty:
            return 0.0
        returns = history.returns()[held.index].iloc[-self.config.cov_lookback :]
        cov = shrunk_covariance(returns, self.config.shrinkage) * self.periods_per_year
        w = held.to_numpy()
        return math.sqrt(max(float(w @ cov.to_numpy() @ w), 0.0))


def shrunk_covariance(returns: pd.DataFrame, shrinkage: float) -> pd.DataFrame:
    """Sample covariance pulled toward its diagonal: steadier with few observations."""
    sample = returns.cov(min_periods=20).fillna(0.0)
    diagonal = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    return (1 - shrinkage) * sample + shrinkage * diagonal


def breaker_reason(equity: pd.Series, config: RiskConfig) -> str | None:
    """Why the agent must stop trading, judged from its equity history (None if fine)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return None
    drawdown = 1 - equity.iloc[-1] / equity.cummax().iloc[-1]
    return breach_reason(drawdown, equity.iloc[-1] / equity.iloc[-2] - 1, config)


def breach_reason(drawdown: float, day_change: float, config: RiskConfig) -> str | None:
    if drawdown >= config.halt_drawdown:
        return f"drawdown of {drawdown:.1%} breached the {config.halt_drawdown:.0%} limit"
    if day_change <= -config.halt_daily_loss:
        return f"one-day loss of {-day_change:.1%} breached the {config.halt_daily_loss:.0%} limit"
    return None


def data_problems(history: PriceHistory, symbols: Sequence[str], config: RiskConfig) -> list[str]:
    """Reasons not to trust the latest bar (an empty list means the data looks sane)."""
    problems = []
    last = history.close.iloc[-1]
    for symbol in symbols:
        if symbol not in history.close.columns or history.close[symbol].notna().sum() == 0:
            problems.append(f"{symbol}: no data")
        elif pd.isna(last[symbol]):
            final = history.close[symbol].last_valid_index()
            problems.append(f"{symbol}: no bar for {history.last_date} (last seen {final.date()})")
    if len(history) >= 2:
        moves = (history.close.iloc[-1] / history.close.iloc[-2] - 1).abs()
        problems += [
            f"{symbol}: {moves[symbol]:.0%} one-day move looks like bad data"
            for symbol in symbols
            if symbol in moves.index and moves[symbol] > config.max_price_jump
        ]
    return problems
