"""An event-driven daily backtester that refuses to flatter the strategy.

On each session:

1. Orders decided yesterday fill at today's **open**, after slippage and the
   broker's real fee schedule. Sells go first; buys are cut back rather than
   ever borrowing.
2. The account is marked to market at today's close.
3. On a rebalance day the strategy sees history up to today's close (and no
   further), the risk manager shapes its wishes, and the shared order planner
   decides tomorrow's trades.

Circuit-breaker trips are recorded rather than acted on, so the report can show
when the live agent would have stopped to wait for a human.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from kea.config import Config
from kea.costs import build_fee_model
from kea.data.history import PriceHistory
from kea.execution import simulate_fill
from kea.orders import Fill, Order, plan_orders
from kea.risk import RiskManager, breach_reason
from kea.schedule import is_rebalance_day
from kea.strategies import Ensemble, Strategy
from kea.strategies.ml import MLForecaster


@dataclass(frozen=True)
class Decision:
    day: date
    raw: pd.Series
    target: pd.Series
    predicted_vol: float
    scale: float
    orders: tuple[Order, ...]


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    cash: pd.Series
    weights: pd.DataFrame
    fills: list[Fill]
    decisions: list[Decision]
    breaker_trips: list[tuple[date, str]]
    initial_cash: float
    periods_per_year: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def total_fees(self) -> float:
        return sum(f.fees for f in self.fills)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.fills])

    def targets_frame(self) -> pd.DataFrame:
        return pd.DataFrame({d.day: d.target for d in self.decisions}).T.fillna(0.0)


class Backtester:
    """Replays a strategy through history.

    `risk_overlay=False` holds the strategy's weights exactly as given, which is
    what a benchmark such as buy-and-hold should do.
    """

    def __init__(
        self,
        config: Config,
        strategy: Strategy,
        initial_cash: float | None = None,
        risk_overlay: bool = True,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.initial_cash = initial_cash or config.broker.initial_cash
        self.risk_overlay = risk_overlay
        u = config.universe
        self.risk = RiskManager(config.risk, u.symbols, u.cash_symbol, u.periods_per_year)
        self.fees = build_fee_model(config.execution)

    def first_decision_index(self, history: PriceHistory, start: date | None = None) -> int:
        first = max(self.strategy.warmup, self.config.risk.cov_lookback + 1)
        if start is not None:
            first = max(first, int(history.dates.searchsorted(pd.Timestamp(start))))
        return first

    def run(
        self, history: PriceHistory, start: date | None = None, end: date | None = None
    ) -> BacktestResult:
        cfg = self.config
        tradables = list(cfg.universe.tradable)
        missing = [s for s in tradables if s not in history.symbols]
        if missing:
            raise ValueError(f"history is missing {', '.join(missing)}")
        dates = history.dates
        first = self.first_decision_index(history, start)
        last = len(dates) - 1
        if end is not None:
            last = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        if first >= last:
            raise ValueError(
                f"not enough history: the strategy needs {first} sessions of warm-up "
                f"but only {last + 1} are available"
            )

        column = {s: j for j, s in enumerate(tradables)}
        opens = history.open[tradables].to_numpy()
        marks = history.close[tradables].ffill().to_numpy()
        quantities = np.zeros(len(tradables))
        cash = float(self.initial_cash)
        orders_per_month: Counter[tuple[int, int]] = Counter()

        pending: list[Order] = []
        fills: list[Fill] = []
        decisions: list[Decision] = []
        trips: list[tuple[date, str]] = []
        equity_path, cash_path, weight_rows = [], [], []
        last_rebalance: date | None = None
        tripped = False
        peak = cash

        for i in range(first, last + 1):
            day = dates[i].date()
            if pending:
                for order in pending:
                    fill = self._fill(
                        order, opens[i], marks[i], column, quantities, cash, day, orders_per_month
                    )
                    if fill is not None:
                        quantities[column[fill.symbol]] += fill.signed_quantity
                        cash += fill.cash_flow
                        orders_per_month[(day.year, day.month)] += 1
                        fills.append(fill)
                pending = []

            values = quantities * np.nan_to_num(marks[i])
            equity = cash + values.sum()
            equity_path.append(equity)
            cash_path.append(cash)
            weight_rows.append(values / equity if equity > 0 else values * 0)

            peak = max(peak, equity)
            previous = equity_path[-2] if len(equity_path) > 1 else equity
            reason = breach_reason(1 - equity / peak, equity / previous - 1, cfg.risk)
            if reason and not tripped:
                trips.append((day, reason))
            tripped = reason is not None

            if i < last and is_rebalance_day(day, last_rebalance, cfg.strategy.rebalance):
                view = history.head(i + 1)
                raw = self.strategy.target_weights(view)
                if self.risk_overlay:
                    shaped = self.risk.apply(raw, view)
                    target, vol, scale = shaped.weights, shaped.predicted_vol, shaped.scale
                else:
                    target, vol, scale = raw.clip(lower=0.0), float("nan"), 1.0
                positions = {s: quantities[j] for s, j in column.items() if quantities[j]}
                prices = {s: marks[i][j] for s, j in column.items()}
                plan = plan_orders(target.to_dict(), positions, cash, prices, cfg.execution)
                pending = plan.orders
                last_rebalance = day
                decisions.append(Decision(day, raw, target, vol, scale, tuple(plan.orders)))

        index = dates[first : last + 1]
        return BacktestResult(
            strategy=self.strategy.name,
            equity=pd.Series(equity_path, index=index, name="equity"),
            cash=pd.Series(cash_path, index=index, name="cash"),
            weights=pd.DataFrame(weight_rows, index=index, columns=tradables),
            fills=fills,
            decisions=decisions,
            breaker_trips=trips,
            initial_cash=float(self.initial_cash),
            periods_per_year=cfg.universe.periods_per_year,
            diagnostics=collect_diagnostics(self.strategy),
        )

    def _fill(
        self,
        order: Order,
        opens: np.ndarray,
        marks: np.ndarray,
        column: dict[str, int],
        quantities: np.ndarray,
        cash: float,
        day: date,
        orders_per_month: Counter[tuple[int, int]],
    ) -> Fill | None:
        execution = self.config.execution
        j = column[order.symbol]
        base = opens[j] if math.isfinite(opens[j]) and opens[j] > 0 else marks[j]
        waive = orders_per_month[(day.year, day.month)] < execution.free_orders_per_month
        return simulate_fill(
            order, float(base), float(quantities[j]), cash, day, self.fees, execution, waive
        )


def collect_diagnostics(strategy: Strategy) -> dict[str, Any]:
    members = strategy.members if isinstance(strategy, Ensemble) else [strategy]
    diagnostics: dict[str, Any] = {}
    for member in members:
        if isinstance(member, MLForecaster):
            diagnostics["ml_predictions"] = member.prediction_frame()
    return diagnostics
