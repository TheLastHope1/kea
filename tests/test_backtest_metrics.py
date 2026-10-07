import dataclasses

import numpy as np
import pandas as pd
import pytest

from kea.backtest import Backtester
from kea.config import ExecutionConfig
from kea.data.history import PriceHistory
from kea.metrics import (
    annual_returns,
    deflated_sharpe,
    expected_max_sharpe,
    longest_drawdown_days,
    performance,
    probabilistic_sharpe,
    trading_stats,
)
from kea.research import account_size_sensitivity, compare
from kea.strategies import build_strategy


def test_accounting_identity_and_no_borrowing(config, history):
    result = Backtester(config, build_strategy(config, "trend")).run(history)
    assert result.cash.min() >= -1e-6
    positions = pd.Series(0.0, index=result.weights.columns)
    for fill in result.fills:
        positions[fill.symbol] += fill.signed_quantity
    last_close = history.close.loc[result.equity.index[-1], positions.index]
    implied = result.cash.iloc[-1] + (positions * last_close).sum()
    assert implied == pytest.approx(result.equity.iloc[-1])


def test_orders_fill_at_the_next_open_with_slippage(config, history):
    result = Backtester(config, build_strategy(config, "buy_and_hold"), risk_overlay=False).run(
        history
    )
    decision, fill = result.decisions[0], result.fills[0]
    next_day = history.dates[history.dates.get_loc(pd.Timestamp(decision.day)) + 1]
    assert fill.day == next_day.date()
    assert fill.price == pytest.approx(history.open.at[next_day, "SPY"] * 1.0005)


def test_decisions_are_blind_to_the_future(config, history):
    """Scrambling every price after a date must not change any decision before it."""
    cut = history.dates[len(history) // 2]
    rng = np.random.default_rng(99)

    def scramble(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        later = frame.index > cut
        frame.loc[later] = frame.loc[later].to_numpy() * rng.uniform(
            0.5, 1.5, frame.loc[later].shape
        )
        return frame

    future_changed = PriceHistory(
        *(scramble(getattr(history, f)) for f in ("open", "high", "low", "close", "volume"))
    )
    for name in ("trend", "momentum"):
        original = Backtester(config, build_strategy(config, name)).run(history)
        altered = Backtester(config, build_strategy(config, name)).run(future_changed)
        before = [d for d in original.decisions if pd.Timestamp(d.day) <= cut]
        after = [d for d in altered.decisions if pd.Timestamp(d.day) <= cut]
        assert len(before) > 3
        for a, b in zip(before, after, strict=True):
            pd.testing.assert_series_equal(a.target, b.target)


def test_buy_and_hold_trades_once_and_rides_the_asset(config, history):
    zero_cost = dataclasses.replace(
        config,
        execution=ExecutionConfig(fees="zero", slippage_bps=0, whole_shares=False, cash_buffer=0),
    )
    strategy = build_strategy(zero_cost, "buy_and_hold")
    result = Backtester(zero_cost, strategy, risk_overlay=False).run(history)
    [fill] = result.fills  # sized at the close, filled at the next open, then left alone
    leftover = result.initial_cash - fill.quantity * fill.price
    assert 0 <= leftover < 0.03 * result.initial_cash
    final_price = history.close.loc[result.equity.index[-1], "SPY"]
    assert result.equity.iloc[-1] == pytest.approx(leftover + fill.quantity * final_price)


def test_fees_are_charged_and_free_trades_waived(config, history):
    paid = Backtester(config, build_strategy(config, "trend")).run(history)
    free = dataclasses.replace(
        config, execution=dataclasses.replace(config.execution, free_orders_per_month=100)
    )
    waived = Backtester(free, build_strategy(free, "trend")).run(history)
    assert paid.total_fees > waived.total_fees > 0  # pass-through fees are never waived


def test_warmup_errors_are_explained(config, history):
    with pytest.raises(ValueError, match="not enough history"):
        Backtester(config, build_strategy(config, "trend")).run(history.head(200))


def test_compare_and_account_size_sensitivity(config, history):
    comparison = compare(config, history, strategies=("trend", "momentum"), trials=10)
    table = comparison.table()
    assert list(table.index) == ["trend", "momentum", "buy_and_hold"]  # no IEF: no 60/40
    assert table.loc["buy_and_hold", "dsr"] is None or pd.isna(table.loc["buy_and_hold", "dsr"])
    assert 0 <= table.loc["trend", "dsr"] <= 1
    assert comparison.trials == 10
    sizes = account_size_sensitivity(config, history, "trend", [2_000, 100_000])
    assert sizes.loc[2_000, "fee_drag"] > sizes.loc[100_000, "fee_drag"]


# ---------------------------------------------------------------- metrics


def daily_equity(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_performance_on_a_known_path():
    equity = pd.Series(
        [100.0, 120.0, 90.0, 180.0],
        index=pd.to_datetime(["2020-01-01", "2020-06-01", "2020-09-01", "2021-01-01"]),
    )
    perf = performance(equity, 252)
    assert perf.total_return == pytest.approx(0.8)
    assert perf.cagr == pytest.approx(1.8 ** (365.25 / 366) - 1)
    assert perf.max_drawdown == pytest.approx(0.25)
    assert annual_returns(equity).round(6).to_dict() == {2020: -0.1, 2021: 1.0}


def test_longest_drawdown_counts_calendar_days():
    equity = daily_equity([100, 90, 95, 101, 99, 98])  # Wed 1 Jan 2020 onwards
    assert longest_drawdown_days(equity) == 4  # underwater Thu 2 Jan until the new high on Mon 6


def test_probabilistic_and_deflated_sharpe_behave():
    rng = np.random.default_rng(3)
    skilled = pd.Series(rng.normal(0.001, 0.01, 2500))
    noise = pd.Series(rng.normal(0.0, 0.01, 2500))
    assert probabilistic_sharpe(skilled) > 0.99
    assert 0.0 < probabilistic_sharpe(noise) < 1.0
    trials = [0.02, 0.05, -0.01, 0.03]
    assert expected_max_sharpe(trials, 1000) > expected_max_sharpe(trials, 4) > 0
    assert deflated_sharpe(skilled, trials, 1000) < deflated_sharpe(skilled, trials, 4)


def test_trading_stats_annualise_fees():
    equity = daily_equity([1000.0] * 253, start="2020-01-01")
    fills = pd.DataFrame({"quantity": [10.0, 10.0], "price": [50.0, 50.0], "fees": [2.0, 2.0]})
    stats = trading_stats(fills, equity)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    assert stats["fee_drag"] == pytest.approx(4.0 / 1000 / years)
    assert stats["turnover"] == pytest.approx(1000 / 2 / 1000 / years)
