"""What trading actually costs: broker fee schedules and slippage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kea.config import ExecutionConfig
from kea.orders import Side


class FeeModel(Protocol):
    def commission(self, quantity: float, price: float, side: Side) -> float:
        """The broker's own charge (promotions can waive this)."""

    def pass_through(self, quantity: float, price: float, side: Side) -> float:
        """Regulatory and settlement charges (never waived)."""


@dataclass(frozen=True)
class TigerNZFees:
    """Tiger Brokers (NZ) fee schedule for US stocks and ETFs.

    Source: https://www.itiger.com/nz/commissions, checked 2026-10-07.

    * 1-200 shares: USD 2 flat.
    * Over 200 shares: USD 0.01/share or 1% of value (min USD 2), whichever is less.
    * Fractional orders (< 1 share): 1% of value, max USD 1.
    * Pass-through: settlement USD 0.003/share (max 7% of value); on sells only,
      the SEC fee (0.00206% of value, min USD 0.01) and FINRA TAF
      (USD 0.000195/share, min USD 0.01, max USD 9.79).
    """

    flat_fee: float = 2.0
    flat_fee_max_shares: float = 200
    per_share: float = 0.01
    value_cap_rate: float = 0.01
    fractional_rate: float = 0.01
    fractional_cap: float = 1.0
    settlement_per_share: float = 0.003
    settlement_cap_rate: float = 0.07
    sec_fee_rate: float = 0.0000206
    sec_fee_min: float = 0.01
    taf_per_share: float = 0.000195
    taf_min: float = 0.01
    taf_max: float = 9.79

    def commission(self, quantity: float, price: float, side: Side) -> float:
        shares = abs(quantity)
        value = shares * price
        if shares < 1:
            return min(self.fractional_rate * value, self.fractional_cap)
        if shares <= self.flat_fee_max_shares:
            return self.flat_fee
        return min(self.per_share * shares, max(self.flat_fee, self.value_cap_rate * value))

    def pass_through(self, quantity: float, price: float, side: Side) -> float:
        shares = abs(quantity)
        value = shares * price
        fee = min(self.settlement_per_share * shares, self.settlement_cap_rate * value)
        if side is Side.SELL:
            fee += max(self.sec_fee_rate * value, self.sec_fee_min)
            fee += min(max(self.taf_per_share * shares, self.taf_min), self.taf_max)
        return fee


@dataclass(frozen=True)
class BasisPointFees:
    """A flat percentage of traded value, e.g. a crypto exchange's taker fee."""

    bps: float

    def commission(self, quantity: float, price: float, side: Side) -> float:
        return abs(quantity) * price * self.bps / 10_000

    def pass_through(self, quantity: float, price: float, side: Side) -> float:
        return 0.0


def build_fee_model(config: ExecutionConfig) -> FeeModel:
    if config.fees == "tiger_nz":
        return TigerNZFees()
    if config.fees == "bps":
        return BasisPointFees(config.fee_bps)
    return BasisPointFees(0.0)


def order_fees(
    model: FeeModel, quantity: float, price: float, side: Side, *, waive_commission: bool = False
) -> float:
    commission = 0.0 if waive_commission else model.commission(quantity, price, side)
    return commission + model.pass_through(quantity, price, side)


def slipped_price(price: float, side: Side, slippage_bps: float) -> float:
    """Fill price after paying half the spread plus market impact."""
    adjustment = slippage_bps / 10_000
    return price * (1 + adjustment) if side is Side.BUY else price * (1 - adjustment)
