"""The broker interface: where orders go, and where the account's truth lives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from kea.data.history import PriceHistory
from kea.orders import Fill, Order


@dataclass(frozen=True)
class Account:
    cash: float
    positions: dict[str, float]
    equity: float
    notes: tuple[str, ...] = field(default=())

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        if self.equity <= 0:
            return {}
        return {s: q * prices[s] / self.equity for s, q in self.positions.items() if s in prices}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderTicket:
    """What happened when an order was handed to the broker."""

    order: Order
    status: str  # "queued" (paper), "submitted" (live broker) or "rejected"
    broker_id: str | None = None
    message: str = ""

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "order": self.order.to_dict()}


class Broker(ABC):
    name: str = "broker"
    live_money: bool = False

    @abstractmethod
    def sync(self, history: PriceHistory) -> list[Fill]:
        """Bring the account up to date; return fills that happened since the last sync."""

    @abstractmethod
    def account(self, prices: Mapping[str, float]) -> Account:
        """Cash, positions and equity, valuing positions at `prices` where needed."""

    @abstractmethod
    def open_orders(self) -> list[Order]:
        """Orders accepted but not yet filled or cancelled."""

    @abstractmethod
    def submit(self, orders: Sequence[Order], day: date) -> list[OrderTicket]:
        """Send orders decided on `day` (a completed session)."""

    def detached(self) -> Broker:
        """A copy safe for dry runs: it may simulate, but must not persist or send anything."""
        raise NotImplementedError(f"{type(self).__name__} does not support dry runs")

    def not_ready_reason(self) -> str | None:
        """Why orders cannot be sent right now (None when they can)."""
        return None

    def describe(self) -> str:
        return f"{self.name} ({'REAL MONEY' if self.live_money else 'paper'})"
