import math

import pandas as pd
import pytest

from kea.config import ExecutionConfig, RiskConfig
from kea.costs import BasisPointFees, TigerNZFees, order_fees, slipped_price
from kea.orders import Order, Side, plan_orders
from kea.risk import RiskManager, breaker_reason, data_problems
from tests.conftest import make_history

RISKY = ["UP", "FLAT", "DOWN", "SPY"]

# ------------------------------------------------------------------- risk


def manager(**overrides) -> RiskManager:
    return RiskManager(RiskConfig(**overrides), RISKY, "BIL", 252)


def test_volatility_cap_scales_down_risky_portfolios(history):
    raw = pd.Series({"UP": 0.25, "FLAT": 0.25, "DOWN": 0.25, "SPY": 0.25})
    decision = manager(target_vol=0.05, max_weight=1.0).apply(raw, history)
    assert decision.raw_vol > 0.05
    assert decision.predicted_vol == pytest.approx(0.05)
    assert decision.scale < 1
    assert decision.weights["BIL"] == pytest.approx(1 - decision.exposure)


def test_volatility_cap_never_levers_up(history):
    raw = pd.Series({"UP": 0.1})
    decision = manager(target_vol=0.9).apply(raw, history)
    assert decision.scale == 1.0
    assert decision.weights["UP"] == pytest.approx(0.1)


def test_weights_are_capped_long_only_and_unlevered(history):
    raw = pd.Series({"UP": 0.9, "FLAT": 0.6, "DOWN": -0.3, "SPY": 0.0})
    decision = manager(max_weight=0.35, target_vol=1.0).apply(raw, history)
    assert decision.weights["DOWN"] == 0
    assert decision.weights[RISKY].max() <= 0.35 + 1e-12
    assert decision.weights.sum() == pytest.approx(1.0)
    assert "UP" in decision.capped


def test_breaker_trips_on_drawdown_and_on_a_bad_day():
    config = RiskConfig(halt_drawdown=0.2, halt_daily_loss=0.05)
    assert breaker_reason(pd.Series([100, 110, 108]), config) is None
    assert "drawdown" in breaker_reason(pd.Series([100, 120, 115, 105, 100, 95]), config)
    assert "one-day loss" in breaker_reason(pd.Series([100, 101, 95]), config)


def test_data_problems_flag_missing_stale_and_absurd_prices():
    history = make_history({"A": 0.0, "B": 0.0}, days=50)
    close = history.close.copy()
    close.iloc[-1, 0] = close.iloc[-2, 0] * 1.8
    close.iloc[-1, 1] = float("nan")
    broken = history.__class__(history.open, history.high, history.low, close, history.volume)
    problems = data_problems(broken, ["A", "B", "C"], RiskConfig(max_price_jump=0.35))
    assert any("A: 80%" in p for p in problems)
    assert any(p.startswith("B: no bar") for p in problems)
    assert "C: no data" in problems


# ------------------------------------------------------------------ costs


@pytest.mark.parametrize(
    ("shares", "price", "commission"),
    [
        (100, 50.0, 2.0),  # flat fee up to 200 shares
        (200, 500.0, 2.0),
        (1000, 50.0, 10.0),  # min(0.01 x 1000, max(2, 1% x 50,000))
        (1000, 0.5, 5.0),  # min(10, max(2, 1% x 500))
        (0.5, 100.0, 0.5),  # fractional: 1% of value...
        (0.9, 1000.0, 1.0),  # ...capped at USD 1
    ],
)
def test_tiger_nz_commission_schedule(shares, price, commission):
    assert TigerNZFees().commission(shares, price, Side.BUY) == pytest.approx(commission)


def test_tiger_nz_pass_through_fees_hit_sells_harder():
    fees = TigerNZFees()
    assert fees.pass_through(100, 50.0, Side.BUY) == pytest.approx(0.30)
    sell = fees.pass_through(100, 50.0, Side.SELL)
    assert sell == pytest.approx(0.30 + 0.103 + 0.0195)
    assert order_fees(fees, 100, 50.0, Side.BUY, waive_commission=True) == pytest.approx(0.30)


def test_basis_point_fees_and_slippage():
    assert BasisPointFees(10).commission(10, 100.0, Side.SELL) == pytest.approx(1.0)
    assert slipped_price(100.0, Side.BUY, 5) == pytest.approx(100.05)
    assert slipped_price(100.0, Side.SELL, 5) == pytest.approx(99.95)


# ----------------------------------------------------------------- orders

PRICES = {"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}


def test_planner_buys_from_cash_in_whole_shares_keeping_a_buffer():
    plan = plan_orders({"SPY": 0.5, "TLT": 0.5}, {}, 10_000.0, PRICES, ExecutionConfig())
    assert [(o.symbol, o.side, o.quantity) for o in plan.orders] == [
        ("SPY", Side.BUY, 49),
        ("TLT", Side.BUY, 99),
    ]
    assert sum(o.notional for o in plan.orders) <= 10_000 * 0.99


def test_planner_skips_drift_inside_the_band():
    positions = {"SPY": 50.0, "TLT": 100.0}
    plan = plan_orders(
        {"SPY": 0.51, "TLT": 0.48}, positions, 0.0, PRICES, ExecutionConfig(cash_buffer=0)
    )
    assert plan.orders == []
    assert plan.skipped == {"SPY": "inside drift band", "TLT": "inside drift band"}


def test_planner_exits_unwanted_assets_and_sells_before_buying():
    positions = {"GLD": 10.5, "SPY": 10.0}
    plan = plan_orders({"SPY": 0.9}, positions, 0.0, PRICES, ExecutionConfig())
    assert plan.orders[0] == Order("GLD", Side.SELL, 10.5, 200.0, "exit")
    assert plan.orders[1].side is Side.BUY
    assert plan.turnover == pytest.approx(sum(o.notional for o in plan.orders) / (2 * 3100.0))


def test_planner_ignores_symbols_without_a_price():
    plan = plan_orders({"XYZ": 0.5}, {}, 1_000.0, {"XYZ": math.nan}, ExecutionConfig())
    assert plan.orders == []
    assert plan.skipped == {"XYZ": "no price"}


def test_fractional_shares_when_allowed():
    config = ExecutionConfig(whole_shares=False, cash_buffer=0, min_trade_value=0)
    plan = plan_orders({"GLD": 0.5}, {}, 1_000.0, PRICES, config)
    assert plan.orders[0].quantity == pytest.approx(2.5)


def test_orders_round_trip_through_dicts():
    order = Order("SPY", Side.SELL, 3, 100.0, "reduce")
    assert Order.from_dict(order.to_dict()) == order
    with pytest.raises(ValueError):
        Order("SPY", Side.BUY, 0, 100.0)
