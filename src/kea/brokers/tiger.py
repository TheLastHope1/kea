"""Tiger Brokers execution through the official OpenAPI SDK.

Safety model: a Tiger *paper* account (17+ digit account number) can be traded
freely. A real-money account is refused unless both `broker.allow_live = true`
in the config file and `KEA_ALLOW_LIVE=yes` in the environment, so going live
always takes two deliberate human steps.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from kea.brokers.base import Account, Broker, OrderTicket
from kea.data.history import PriceHistory
from kea.orders import Fill, Order, Side
from kea.tiger import is_paper_account

LIVE_ENV_FLAG = "KEA_ALLOW_LIVE"
ORDER_MARK = "kea"


class LiveTradingBlocked(RuntimeError):
    """A real-money account was configured without both live-trading switches."""


class TigerBroker(Broker):
    name = "tiger"

    def __init__(
        self,
        allow_live: bool = False,
        trade_client: Any = None,
        quote_client: Any = None,
        account: str | None = None,
    ) -> None:
        if trade_client is None or account is None:
            from kea.tiger import client_config

            config = client_config()
            account = account or config.account
            if trade_client is None:
                from tigeropen.quote.quote_client import QuoteClient
                from tigeropen.trade.trade_client import TradeClient

                trade_client = TradeClient(config)
                quote_client = quote_client or QuoteClient(config)
        self.client = trade_client
        self.quotes = quote_client
        self.account_id = str(account)
        self.live_money = not is_paper_account(self.account_id)
        if self.live_money and not (allow_live and os.environ.get(LIVE_ENV_FLAG) == "yes"):
            raise LiveTradingBlocked(
                f"Tiger account ...{self.account_id[-4:]} is a REAL-MONEY account. Kea will only "
                f"trade it with broker.allow_live = true in the config AND {LIVE_ENV_FLAG}=yes "
                "in the environment. Use a Tiger paper account until you are sure."
            )

    def detached(self) -> TigerBroker:
        clone = copy.copy(self)
        clone.submit = clone._refuse  # type: ignore[method-assign]
        return clone

    def _refuse(self, orders: Sequence[Order], day: date) -> list[OrderTicket]:
        raise RuntimeError("a detached (dry-run) broker must never submit orders")

    def describe(self) -> str:
        kind = "REAL MONEY" if self.live_money else "paper"
        return f"tiger account ...{self.account_id[-4:]} ({kind})"

    def sync(self, history: PriceHistory) -> list[Fill]:
        """Every order that filled in the last week; the agent de-duplicates by order id."""
        now_ms = int(time.time() * 1000)
        orders = self.client.get_filled_orders(
            account=self.account_id,
            sec_type="STK",
            market="US",
            start_time=now_ms - 7 * 86_400_000,
            end_time=now_ms,
        )
        fills = []
        for o in orders or []:
            if not o.filled:
                continue
            traded = datetime.fromtimestamp((o.trade_time or o.order_time or now_ms) / 1000, UTC)
            fills.append(
                Fill(
                    symbol=o.contract.symbol,
                    side=Side(str(o.action).upper()),
                    quantity=float(o.filled),
                    price=float(o.avg_fill_price or 0.0),
                    fees=float(o.commission or 0.0),
                    day=traded.date(),
                    order_id=str(o.id),
                )
            )
        return fills

    def account(self, prices: Mapping[str, float]) -> Account:
        positions = {
            p.contract.symbol: float(p.quantity)
            for p in self.client.get_positions(account=self.account_id, sec_type="STK", market="US")
            or []
            if p.quantity
        }
        cash, equity, notes = self._balances()
        if equity is None:
            equity = cash + sum(q * prices.get(s, 0.0) for s, q in positions.items())
        return Account(cash, positions, equity, notes)

    def _balances(self) -> tuple[float, float | None, tuple[str, ...]]:
        """USD cash available to trade and net liquidation value."""
        try:
            assets = self.client.get_prime_assets(account=self.account_id, base_currency="USD")
            segment = assets.segments["S"]
            usd = segment.currency_assets.get("USD")
            cash = _finite(usd.cash_available_for_trade if usd else None)
            if cash is None:
                cash = _finite(segment.cash_available_for_trade)
            notes: tuple[str, ...] = ()
            if usd is None or _finite(usd.cash_balance) == 0:
                notes = ("no USD cash: convert NZD to USD in Tiger before Kea can buy US ETFs",)
            return cash or 0.0, _finite(segment.net_liquidation), notes
        except Exception:  # standard (non-prime) accounts use the older assets endpoint
            portfolio = self.client.get_assets(account=self.account_id)[0]
            summary = portfolio.summary
            cash = _finite(summary.available_funds) or _finite(summary.cash) or 0.0
            return cash, _finite(summary.net_liquidation), ()

    def open_orders(self) -> list[Order]:
        orders = self.client.get_open_orders(account=self.account_id, sec_type="STK", market="US")
        result = []
        for o in orders or []:
            remaining = float(o.quantity or 0) - float(o.filled or 0)
            if remaining > 0:
                side = Side(str(o.action).upper())
                result.append(Order(o.contract.symbol, side, remaining, float(o.limit_price or 0)))
        return result

    def not_ready_reason(self) -> str | None:
        if self.quotes is None:
            return None
        statuses = self.quotes.get_market_status(market="US") or []
        status = statuses[0] if statuses else None
        if status is None or status.trading_status != "TRADING":
            label = getattr(status, "status", "unknown")
            return (
                f"US market is not open (status: {label}); Tiger orders go in during trading hours"
            )
        return None

    def submit(self, orders: Sequence[Order], day: date) -> list[OrderTicket]:
        from tigeropen.common.util.contract_utils import stock_contract
        from tigeropen.common.util.order_utils import market_order

        tickets = []
        for order in orders:
            if not float(order.quantity).is_integer():
                tickets.append(
                    OrderTicket(order, "rejected", message="Tiger orders must be whole shares")
                )
                continue
            contract = stock_contract(symbol=order.symbol, currency="USD")
            tiger_order = market_order(
                account=self.account_id,
                contract=contract,
                action=order.side.value,
                quantity=int(order.quantity),
            )
            tiger_order.user_mark = ORDER_MARK
            try:
                broker_id = self.client.place_order(tiger_order)
            except Exception as exc:  # surface the broker's message, never crash mid-batch
                tickets.append(OrderTicket(order, "rejected", message=str(exc)))
                continue
            tickets.append(OrderTicket(order, "submitted", broker_id=str(broker_id)))
        return tickets


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
