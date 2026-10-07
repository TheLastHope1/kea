"""Orders, fills, and the planner that turns target weights into a short list of trades.

The planner is shared by the backtester and the live agent, so the trades a
backtest assumes are exactly the trades the agent would place.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from kea.config import ExecutionConfig


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    reference_price: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")

    @property
    def notional(self) -> float:
        return self.quantity * self.reference_price

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Order:
        return cls(**{**data, "side": Side(data["side"])})


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    fees: float
    day: date
    order_id: str | None = None

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side is Side.BUY else -self.quantity

    @property
    def cash_flow(self) -> float:
        """Change in cash: negative for buys, positive for sells, always net of fees."""
        gross = self.quantity * self.price
        return (-gross if self.side is Side.BUY else gross) - self.fees

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "day": self.day.isoformat()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fill:
        return cls(**{**data, "side": Side(data["side"]), "day": date.fromisoformat(data["day"])})


@dataclass(frozen=True)
class Plan:
    orders: list[Order]
    equity: float
    target_values: dict[str, float]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def turnover(self) -> float:
        """One-way turnover as a fraction of equity (a full rotation is 1.0)."""
        return sum(o.notional for o in self.orders) / (2 * self.equity) if self.equity else 0.0


def plan_orders(
    targets: Mapping[str, float],
    positions: Mapping[str, float],
    cash: float,
    prices: Mapping[str, float],
    config: ExecutionConfig,
) -> Plan:
    """Orders that move the account toward `targets` (weights of equity).

    * A `cash_buffer` slice of equity stays uninvested to absorb overnight gaps.
    * Trades smaller than the drift band (or `min_trade_value`) are skipped: with a
      flat USD 2 commission, chasing small deviations is a guaranteed loss.
    * Assets the strategy no longer wants are always exited in full.
    * Sells come first so their proceeds fund the buys.
    """
    held_value = sum(q * prices[s] for s, q in positions.items() if q and _valid(prices.get(s)))
    equity = cash + held_value
    investable = equity * (1 - config.cash_buffer)
    threshold = max(config.drift_band * equity, config.min_trade_value)
    symbols = sorted(
        {s for s, w in targets.items() if w > 0} | {s for s, q in positions.items() if q}
    )

    orders: list[Order] = []
    target_values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for symbol in symbols:
        price = prices.get(symbol)
        if not _valid(price):
            skipped[symbol] = "no price"
            continue
        held = positions.get(symbol, 0.0)
        weight = targets.get(symbol, 0.0)
        target_values[symbol] = weight * investable
        delta = target_values[symbol] - held * price
        if weight <= 0 and held > 0:
            orders.append(Order(symbol, Side.SELL, held, price, "exit"))
            continue
        if abs(delta) < threshold:
            skipped[symbol] = "inside drift band"
            continue
        quantity = abs(delta) / price
        if config.whole_shares:
            quantity = math.floor(quantity) if delta > 0 else min(math.ceil(quantity), held)
        if quantity <= 0:
            skipped[symbol] = "rounds to zero shares"
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        reason = "increase" if side is Side.BUY else "reduce"
        orders.append(Order(symbol, side, quantity, price, reason))

    orders.sort(key=lambda o: (o.side is Side.BUY, o.symbol))
    return Plan(orders=orders, equity=equity, target_values=target_values, skipped=skipped)


def _valid(price: float | None) -> bool:
    return price is not None and math.isfinite(price) and price > 0
