"""The autonomous agent: one careful decision cycle per run.

Run it on a schedule (cron, launchd, GitHub Actions). Each run:

1. loads the latest *completed* market data and settles the broker;
2. records the account's equity in the track record;
3. trips the circuit breaker on a deep drawdown or a big one-day loss, after
   which it does nothing until a human runs `kea resume`;
4. skips trading if the data looks wrong, orders are still working, or this
   session was already handled (so running twice is harmless);
5. on a rebalance day, asks the strategy for targets, shapes them with the risk
   manager, plans orders with the same planner the backtest uses, and submits;
6. journals everything, including the reasons for doing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kea.brokers.base import Account, Broker, OrderTicket
from kea.config import Config
from kea.data.store import MarketData
from kea.ledger import AgentState, Ledger
from kea.orders import Fill, plan_orders
from kea.risk import RiskManager, breaker_reason, data_problems
from kea.schedule import is_rebalance_day
from kea.strategies import Strategy

log = logging.getLogger(__name__)

MAX_RETRIES = 3


@dataclass
class RunReport:
    status: str  # traded | held | skipped | waiting | halted | dry-run
    message: str
    as_of: str | None = None
    account: Account | None = None
    targets: dict[str, float] = field(default_factory=dict)
    tickets: list[OrderTicket] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "as_of": self.as_of,
            "account": self.account.to_dict() if self.account else None,
            "targets": {k: round(v, 4) for k, v in self.targets.items() if v},
            "orders": [t.to_dict() for t in self.tickets],
            "fills": [f.to_dict() for f in self.fills],
            **self.details,
        }


class Agent:
    def __init__(
        self,
        config: Config,
        market_data: MarketData,
        strategy: Strategy,
        broker: Broker,
        ledger: Ledger,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self.strategy = strategy
        self.broker = broker
        self.ledger = ledger
        u = config.universe
        self.risk = RiskManager(config.risk, u.symbols, u.cash_symbol, u.periods_per_year)

    def run(self, dry_run: bool = False, now: datetime | None = None) -> RunReport:
        """One decision cycle. A dry run plans as usual but changes nothing on disk."""
        with self.ledger.lock():
            state = self.ledger.load_state()
            report = self._cycle(state, dry_run, now or datetime.now(UTC))
            if not dry_run:
                self.ledger.save_state(state)
                self.ledger.record({"kind": "run", **report.to_event()})
            return report

    def _cycle(self, state: AgentState, dry_run: bool, now: datetime) -> RunReport:
        cfg = self.config
        if state.halted:
            return RunReport(
                "halted",
                f"halted since {state.halted['since']}: {state.halted['reason']}. "
                "Review the journal, then run `kea resume`.",
            )

        broker = self.broker.detached() if dry_run else self.broker
        held = broker.account({}).positions
        symbols = list(dict.fromkeys([*cfg.universe.all_symbols, *held]))
        history = self.market_data.history(symbols, now)
        as_of = history.last_date
        prices = {s: float(p) for s, p in history.latest_prices().dropna().items()}
        details: dict[str, Any] = {"data_sources": dict(self.market_data.sources)}

        fills = [f for f in broker.sync(history) if self._is_new(f, state)]
        account = broker.account(prices)
        exposure = sum(
            w for s, w in account.weights(prices).items() if s != cfg.universe.cash_symbol
        )
        benchmark = prices.get(cfg.universe.benchmark)
        track = self.ledger.record_equity(
            as_of, account.equity, account.cash, exposure, benchmark, persist=not dry_run
        )

        def report(status: str, message: str) -> RunReport:
            return RunReport(
                status, message, as_of.isoformat(), account, fills=fills, details=details
            )

        reason = breaker_reason(track["equity"], cfg.risk)
        if reason:
            if not dry_run:
                state.halted = {"reason": reason, "since": as_of.isoformat()}
            return report("halted", f"circuit breaker: {reason}")

        if state.last_processed_date and as_of <= state.last_processed_date:
            return report("skipped", f"session {as_of} was already handled")

        problems = data_problems(history, cfg.universe.tradable, cfg.risk)
        if problems:
            return report("skipped", "data problems: " + "; ".join(problems))

        if price_only := self.market_data.price_only:
            details["warning"] = f"price-only data (dividends ignored) for {', '.join(price_only)}"

        working = broker.open_orders()
        if working:
            return report("waiting", f"{len(working)} order(s) still working at the broker")

        rebalance = is_rebalance_day(as_of, state.last_rebalance_date, cfg.strategy.rebalance)
        if rebalance:
            raw = self.strategy.target_weights(history)
            decision = self.risk.apply(raw, history)
            targets = {s: float(w) for s, w in decision.weights.items()}
            details["risk"] = {
                "predicted_vol": round(decision.predicted_vol, 4),
                "raw_vol": round(decision.raw_vol, 4),
                "vol_scale": round(decision.scale, 4),
                "capped": list(decision.capped),
                "strategy_weights": {s: round(float(w), 4) for s, w in raw.items() if w},
            }
        elif state.retries_left > 0 and state.targets:
            targets = state.targets
            details["retry"] = f"completing the {state.last_rebalance} rebalance"
        else:
            if not dry_run:
                state.last_processed = as_of.isoformat()
            next_period = "week" if cfg.strategy.rebalance == "weekly" else "month"
            return report(
                "held",
                f"no rebalance due; next decision in the first session of next {next_period}",
            )

        plan = plan_orders(targets, account.positions, account.cash, prices, cfg.execution)
        details["skipped"] = plan.skipped
        details["turnover"] = round(plan.turnover, 4)
        result = RunReport("held", "", as_of.isoformat(), account, targets, [], fills, details)

        if not plan.orders:
            result.message = "portfolio already within drift bands of its targets"
        elif dry_run:
            result.status = "dry-run"
            result.message = f"would place {len(plan.orders)} order(s)"
            result.tickets = [OrderTicket(o, "dry-run") for o in plan.orders]
            return result
        elif not_ready := broker.not_ready_reason():
            result.status = "waiting"
            result.message = not_ready
            return result
        else:
            result.tickets = broker.submit(plan.orders, as_of)
            rejected = [t for t in result.tickets if t.rejected]
            result.status = "traded"
            result.message = f"sent {len(result.tickets) - len(rejected)} order(s)" + (
                f", {len(rejected)} rejected" if rejected else ""
            )
            # A rejected order (e.g. not enough USD cash yet) is retried on the next few
            # sessions toward the same targets; a fresh rebalance restarts the count.
            if rejected:
                state.retries_left = MAX_RETRIES if rebalance else max(state.retries_left - 1, 0)
            else:
                state.retries_left = 0

        if not dry_run:
            state.last_processed = as_of.isoformat()
            if rebalance:
                state.last_rebalance = as_of.isoformat()
                state.targets = targets
        return result

    def _is_new(self, fill: Fill, state: AgentState) -> bool:
        if fill.order_id is None:
            return True
        if fill.order_id in state.seen_fills:
            return False
        state.seen_fills = [*state.seen_fills, fill.order_id][-500:]
        return True


def resume(ledger: Ledger) -> str:
    state = ledger.load_state()
    if not state.halted:
        return "Kea is not halted."
    reason = state.halted["reason"]
    state.halted = None
    ledger.save_state(state)
    ledger.record({"kind": "resume", "message": f"resumed by operator after: {reason}"})
    return f"Resumed. The halt was: {reason}"
