"""Simulated order execution, shared by the backtester and the paper broker.

Keeping one implementation means a paper-trading account fills exactly the way
the backtest assumed it would: same slippage, same fee schedule, same refusal
to ever borrow.
"""

from __future__ import annotations

import math
from datetime import date

from kea.config import ExecutionConfig
from kea.costs import FeeModel, order_fees, slipped_price
from kea.orders import Fill, Order, Side


def simulate_fill(
    order: Order,
    base_price: float,
    held: float,
    cash: float,
    day: date,
    fee_model: FeeModel,
    config: ExecutionConfig,
    waive_commission: bool = False,
) -> Fill | None:
    """Fill `order` at `base_price` (normally the session open), or None if impossible.

    Sells are capped at the shares held. Buys are shrunk until the cost plus fees
    fits in the available cash, so the account never goes negative.
    """
    if not (math.isfinite(base_price) and base_price > 0):
        return None
    price = slipped_price(base_price, order.side, config.slippage_bps)

    def fees(quantity: float) -> float:
        return order_fees(fee_model, quantity, price, order.side, waive_commission=waive_commission)

    if order.side is Side.SELL:
        quantity = min(order.quantity, held)
    else:
        quantity = order.quantity
        for _ in range(10):
            shortfall = quantity * price + fees(quantity) - cash
            if shortfall <= 1e-9:
                break
            quantity -= shortfall / price
            if config.whole_shares:
                quantity = math.floor(quantity)
            if quantity <= 0:
                return None
        else:
            return None
    if quantity <= 1e-12:
        return None
    return Fill(order.symbol, order.side, quantity, price, fees(quantity), day)
