"""Strategies turn price history into target weights; `build_strategy` wires them from config."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from kea.config import Config
from kea.data.history import PriceHistory
from kea.strategies.base import Strategy
from kea.strategies.ml import MLForecaster
from kea.strategies.trend import DualMomentum, StaticWeights, TrendFollowing


class Ensemble(Strategy):
    """Equal-weight blend of member strategies.

    Blending signals that fail at different times (slow trend, fast momentum, a
    statistical model) is the cheapest robustness there is.
    """

    name = "ensemble"

    def __init__(self, members: Sequence[Strategy]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        first = members[0]
        super().__init__(first.symbols, first.cash_symbol, first.periods_per_year)
        self.members = list(members)

    @property
    def warmup(self) -> int:
        return max(m.warmup for m in self.members)

    def target_weights(self, history: PriceHistory) -> pd.Series:
        blend = sum(
            m.target_weights(history).reindex(self.symbols).fillna(0.0) for m in self.members
        )
        return blend / len(self.members)

    def member_weights(self, history: PriceHistory) -> dict[str, pd.Series]:
        return {m.name: m.target_weights(history) for m in self.members}


def build_member(name: str, config: Config) -> Strategy:
    u, s = config.universe, config.strategy
    args = (u.symbols, u.cash_symbol, u.periods_per_year)
    if name == "trend":
        return TrendFollowing(*args, s.trend)
    if name == "momentum":
        return DualMomentum(*args, s.momentum)
    if name == "ml":
        return MLForecaster(*args, s.ml)
    raise ValueError(f"unknown strategy '{name}' (choose from {', '.join(STRATEGY_NAMES)})")


def build_strategy(config: Config, name: str | None = None) -> Strategy:
    """The configured strategy, or a named one (any member, 'ensemble' or a benchmark)."""
    name = name or config.strategy.name
    if name == "ensemble":
        return Ensemble([build_member(m, config) for m in config.strategy.members])
    if name in BENCHMARK_NAMES:
        return build_benchmark(name, config)
    return build_member(name, config)


def build_benchmark(name: str, config: Config) -> Strategy:
    ppy = config.universe.periods_per_year
    benchmark = config.universe.benchmark
    if name == "buy_and_hold":
        return StaticWeights("buy_and_hold", {benchmark: 1.0}, ppy)
    if name == "sixty_forty":
        bonds = "IEF" if "IEF" in config.universe.symbols else config.universe.cash_symbol
        if bonds is None:
            raise ValueError("sixty_forty needs IEF or a cash symbol in the universe")
        return StaticWeights("sixty_forty", {benchmark: 0.6, bonds: 0.4}, ppy)
    raise ValueError(f"unknown benchmark '{name}'")


STRATEGY_NAMES = ("trend", "momentum", "ml", "ensemble")
BENCHMARK_NAMES = ("buy_and_hold", "sixty_forty")

__all__ = [
    "BENCHMARK_NAMES",
    "STRATEGY_NAMES",
    "DualMomentum",
    "Ensemble",
    "MLForecaster",
    "StaticWeights",
    "Strategy",
    "TrendFollowing",
    "build_benchmark",
    "build_member",
    "build_strategy",
]
