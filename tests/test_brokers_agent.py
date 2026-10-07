import dataclasses
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from kea.agent import Agent, resume
from kea.brokers.paper import PaperBroker
from kea.brokers.tiger import LIVE_ENV_FLAG, LiveTradingBlocked, TigerBroker
from kea.config import ExecutionConfig, RiskConfig, StrategyConfig
from kea.data.history import PriceHistory
from kea.ledger import Ledger
from kea.orders import Order, Side
from kea.strategies import build_strategy

# ------------------------------------------------------------------ paper


def test_paper_orders_fill_at_the_next_open_and_persist(tmp_path, history):
    path = tmp_path / "paper.json"
    broker = PaperBroker(path, 10_000.0, ExecutionConfig(slippage_bps=0))
    day = history.dates[100]
    broker.submit([Order("SPY", Side.BUY, 10, 100.0)], day.date())
    assert broker.sync(history.until(day)) == []  # no later session yet

    fills = broker.sync(history.until(history.dates[105]))
    assert len(fills) == 1
    assert fills[0].day == history.dates[101].date()
    assert fills[0].price == pytest.approx(history.open.at[history.dates[101], "SPY"])

    reloaded = PaperBroker(path, 10_000.0, ExecutionConfig())
    assert reloaded.state["positions"] == {"SPY": 10}
    assert reloaded.state["cash"] == pytest.approx(10_000 - 10 * fills[0].price - fills[0].fees)
    assert reloaded.open_orders() == []


def test_paper_detached_copy_never_touches_disk(tmp_path, history):
    broker = PaperBroker(tmp_path / "paper.json", 10_000.0, ExecutionConfig())
    broker.submit([Order("SPY", Side.BUY, 1, 100.0)], history.dates[100].date())
    clone = broker.detached()
    clone.sync(history)
    assert clone.state["positions"]
    assert (
        PaperBroker(tmp_path / "paper.json", 10_000.0, ExecutionConfig()).state["positions"] == {}
    )


def test_paper_waives_commission_for_promotional_free_trades(tmp_path, history):
    config = ExecutionConfig(free_orders_per_month=1, slippage_bps=0)
    broker = PaperBroker(tmp_path / "paper.json", 10_000.0, config)
    day = history.dates[100].date()
    broker.submit([Order("SPY", Side.BUY, 1, 100.0), Order("UP", Side.BUY, 1, 100.0)], day)
    first, second = broker.sync(history)
    assert first.fees == pytest.approx(0.003)  # settlement only
    assert second.fees == pytest.approx(2.003)


# ------------------------------------------------------------------ agent


class FakeMarket:
    """Serves a fixed history; `upto` simulates the clock moving forward."""

    def __init__(self, history: PriceHistory) -> None:
        self.full = history
        self.upto = len(history)
        self.sources = {s: "fake" for s in history.symbols}
        self.price_only: list[str] = []

    def history(self, symbols, now=None) -> PriceHistory:
        return self.full.head(self.upto).select(symbols)


@pytest.fixture
def agent_setup(config, history):
    config = dataclasses.replace(config, strategy=StrategyConfig(name="trend"))
    market = FakeMarket(history)
    market.upto = 400
    broker = PaperBroker(
        config.broker.state_dir / "paper_account.json", 100_000.0, config.execution
    )
    ledger = Ledger(config.broker.state_dir)
    agent = Agent(config, market, build_strategy(config), broker, ledger)
    return agent, market, ledger


NOW = datetime(2030, 1, 1, tzinfo=UTC)


def test_agent_trades_then_is_idempotent_then_settles(agent_setup):
    agent, market, ledger = agent_setup
    first = agent.run(now=NOW)
    assert first.status == "traded"
    assert first.tickets and all(t.status == "queued" for t in first.tickets)

    assert agent.run(now=NOW).status == "skipped"  # same session: harmless rerun

    market.upto += 1
    nxt = agent.run(now=NOW)
    assert nxt.status == "held"
    assert len(nxt.fills) == len(first.tickets)
    assert agent.broker.state["positions"]
    assert len(ledger.equity()) == 2
    assert [e["status"] for e in ledger.journal()] == ["traded", "skipped", "held"]


def test_dry_run_changes_nothing(agent_setup):
    agent, _, ledger = agent_setup
    report = agent.run(dry_run=True, now=NOW)
    assert report.status == "dry-run"
    assert report.tickets
    assert not ledger.state_path.exists()
    assert not ledger.equity_path.exists()
    assert ledger.journal() == []
    assert agent.broker.state["pending"] == []


def test_agent_halts_on_drawdown_until_resumed(agent_setup):
    agent, market, ledger = agent_setup
    agent.config = dataclasses.replace(agent.config, risk=RiskConfig(halt_drawdown=0.01))
    agent.run(now=NOW)
    ledger.record_equity(date(2000, 1, 1), 1_000_000.0, 0.0, 0.0, None)  # an old, higher peak
    market.upto += 1
    halted = agent.run(now=NOW)
    assert halted.status == "halted"
    market.upto += 1
    assert agent.run(now=NOW).status == "halted"  # stays halted on later sessions
    assert "Resumed" in resume(ledger)
    assert ledger.load_state().halted is None


def test_agent_skips_trading_on_suspicious_data(agent_setup, history):
    agent, market, _ = agent_setup
    close = history.close.copy()
    close.iloc[399, close.columns.get_loc("UP")] *= 3
    market.full = PriceHistory(history.open, history.high, history.low, close, history.volume)
    report = agent.run(now=NOW)
    assert report.status == "skipped"
    assert "UP" in report.message


# ------------------------------------------------------------------ tiger


class FakeTradeClient:
    def __init__(self):
        self.placed = []

    def get_positions(self, **kwargs):
        contract = SimpleNamespace(symbol="SPY")
        return [
            SimpleNamespace(contract=contract, quantity=3),
            SimpleNamespace(contract=contract, quantity=0),
        ]

    def get_prime_assets(self, **kwargs):
        usd = SimpleNamespace(cash_balance=1500.0, cash_available_for_trade=1200.0)
        segment = SimpleNamespace(
            currency_assets={"USD": usd},
            cash_available_for_trade=float("inf"),
            net_liquidation=3500.0,
        )
        return SimpleNamespace(segments={"S": segment})

    def get_open_orders(self, **kwargs):
        contract = SimpleNamespace(symbol="TLT")
        return [
            SimpleNamespace(
                contract=contract, action="BUY", quantity=10, filled=4, limit_price=None
            )
        ]

    def get_filled_orders(self, **kwargs):
        contract = SimpleNamespace(symbol="SPY")
        return [
            SimpleNamespace(
                contract=contract,
                action="BUY",
                filled=3,
                avg_fill_price=600.0,
                commission=2.0,
                trade_time=1_790_000_000_000,
                order_time=None,
                id=42,
            )
        ]

    def place_order(self, order):
        self.placed.append(order)
        return 1000 + len(self.placed)


PAPER_ACCOUNT = "21234567890123456"  # 17 digits: a Tiger paper account


def test_tiger_refuses_real_money_without_both_switches(monkeypatch):
    client = FakeTradeClient()
    with pytest.raises(LiveTradingBlocked, match="REAL-MONEY"):
        TigerBroker(allow_live=False, trade_client=client, account="U1234567")
    monkeypatch.setenv(LIVE_ENV_FLAG, "no")
    with pytest.raises(LiveTradingBlocked):
        TigerBroker(allow_live=True, trade_client=client, account="U1234567")
    monkeypatch.setenv(LIVE_ENV_FLAG, "yes")
    assert TigerBroker(allow_live=True, trade_client=client, account="U1234567").live_money


def test_tiger_maps_account_orders_and_fills():
    client = FakeTradeClient()
    broker = TigerBroker(trade_client=client, account=PAPER_ACCOUNT)
    assert not broker.live_money

    account = broker.account({"SPY": 600.0})
    assert account.positions == {"SPY": 3.0}
    assert (account.cash, account.equity) == (1200.0, 3500.0)

    assert broker.open_orders() == [Order("TLT", Side.BUY, 6.0, 0.0)]
    [fill] = broker.sync(history=None)
    assert (fill.symbol, fill.quantity, fill.order_id) == ("SPY", 3.0, "42")

    tickets = broker.submit(
        [Order("SPY", Side.BUY, 2, 600.0), Order("GLD", Side.BUY, 1.5, 300.0)], date.today()
    )
    assert [t.status for t in tickets] == ["submitted", "rejected"]
    assert client.placed[0].contract.symbol == "SPY" and client.placed[0].quantity == 2


def test_tiger_waits_for_the_market_and_dry_runs_cannot_trade():
    quotes = SimpleNamespace(
        get_market_status=lambda market: [SimpleNamespace(status="Closed", trading_status="CLOSED")]
    )
    broker = TigerBroker(trade_client=FakeTradeClient(), quote_client=quotes, account=PAPER_ACCOUNT)
    assert "not open" in broker.not_ready_reason()
    with pytest.raises(RuntimeError, match="must never submit"):
        broker.detached().submit([Order("SPY", Side.BUY, 1, 1.0)], date.today())


def test_ledger_equity_round_trips(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_equity(date(2026, 1, 2), 100.0, 50.0, 0.5, 600.0)
    frame = ledger.record_equity(date(2026, 1, 5), 101.0, 50.0, 0.5, None)
    reloaded = ledger.equity()
    assert list(reloaded.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert reloaded["equity"].tolist() == frame["equity"].tolist() == [100.0, 101.0]
