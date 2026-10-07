"""A simulated account that fills orders exactly the way the backtest does."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from kea.brokers.base import Account, Broker, OrderTicket
from kea.config import ExecutionConfig
from kea.costs import FeeModel, build_fee_model
from kea.data.history import PriceHistory
from kea.execution import simulate_fill
from kea.orders import Fill, Order


class PaperBroker(Broker):
    """Paper trading against real market data.

    Orders decided after day d's close fill at the open of the next session,
    with the backtest's slippage and fee schedule. State is a small JSON file,
    so every run of the agent continues the same account, and the file's git
    history doubles as a tamper-evident track record.
    """

    name = "paper"
    live_money = False

    def __init__(
        self,
        path: Path,
        initial_cash: float,
        execution: ExecutionConfig,
        fee_model: FeeModel | None = None,
    ) -> None:
        self.path = path
        self.execution = execution
        self.fee_model = fee_model or build_fee_model(execution)
        self.state: dict[str, Any] = self._load(initial_cash)
        self.cancelled: list[Order] = []
        self.persist = True

    def _load(self, initial_cash: float) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {
            "initial_cash": initial_cash,
            "cash": initial_cash,
            "positions": {},
            "pending": [],
            "fills": [],
        }

    def describe(self) -> str:
        return f"local paper account ({self.path})"

    def detached(self) -> PaperBroker:
        clone = copy.copy(self)
        clone.state = copy.deepcopy(self.state)
        clone.persist = False
        return clone

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)

    @property
    def fills(self) -> list[Fill]:
        return [Fill.from_dict(f) for f in self.state["fills"]]

    def sync(self, history: PriceHistory) -> list[Fill]:
        settled: list[Fill] = []
        waiting: list[dict[str, Any]] = []
        self.cancelled = []
        for entry in self.state["pending"]:
            order = Order.from_dict(entry["order"])
            later = history.dates[history.dates > pd.Timestamp(entry["day"])]
            if len(later) == 0:
                waiting.append(entry)
                continue
            fill_day = later[0]
            fill = simulate_fill(
                order,
                self._open_price(history, order.symbol, fill_day),
                self.state["positions"].get(order.symbol, 0.0),
                self.state["cash"],
                fill_day.date(),
                self.fee_model,
                self.execution,
                waive_commission=self._orders_in_month(fill_day.date())
                < self.execution.free_orders_per_month,
            )
            if fill is None:
                self.cancelled.append(order)
            else:
                self._apply(fill)
                settled.append(fill)
        self.state["pending"] = waiting
        self.save()
        return settled

    def account(self, prices: Mapping[str, float]) -> Account:
        positions = {s: q for s, q in self.state["positions"].items() if q}
        missing = sorted(s for s in positions if s not in prices)
        value = sum(q * prices[s] for s, q in positions.items() if s in prices)
        notes = (f"no price to value {', '.join(missing)}",) if missing else ()
        return Account(self.state["cash"], positions, self.state["cash"] + value, notes)

    def open_orders(self) -> list[Order]:
        return [Order.from_dict(e["order"]) for e in self.state["pending"]]

    def submit(self, orders: Sequence[Order], day: date) -> list[OrderTicket]:
        self.state["pending"].extend({"order": o.to_dict(), "day": day.isoformat()} for o in orders)
        self.save()
        return [
            OrderTicket(o, "queued", message="fills at the next session's open") for o in orders
        ]

    def _apply(self, fill: Fill) -> None:
        positions = self.state["positions"]
        quantity = positions.get(fill.symbol, 0.0) + fill.signed_quantity
        if abs(quantity) < 1e-9:
            positions.pop(fill.symbol, None)
        else:
            positions[fill.symbol] = quantity
        self.state["cash"] += fill.cash_flow
        self.state["fills"].append(fill.to_dict())

    def _orders_in_month(self, day: date) -> int:
        return sum(1 for f in self.state["fills"] if f["day"][:7] == day.isoformat()[:7])

    @staticmethod
    def _open_price(history: PriceHistory, symbol: str, day: pd.Timestamp) -> float:
        if symbol not in history.symbols:
            return float("nan")
        price = history.open.at[day, symbol]
        return float(price if pd.notna(price) else history.close.at[day, symbol])
