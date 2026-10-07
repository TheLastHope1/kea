"""Head-to-head comparison of strategies and benchmarks over one common window."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from kea.backtest import Backtester, BacktestResult
from kea.config import Config
from kea.data.history import PriceHistory
from kea.fmt import count, money, num, pct
from kea.metrics import (
    Performance,
    deflated_sharpe,
    per_period_sharpe,
    performance,
    trading_stats,
)
from kea.strategies import BENCHMARK_NAMES, build_strategy
from kea.strategies.ml import score_predictions


@dataclass
class Entry:
    name: str
    result: BacktestResult
    performance: Performance
    trading: dict[str, float]
    benchmark: bool
    beats_benchmark: float | None = None

    def row(self) -> dict[str, Any]:
        p = self.performance
        return {
            "strategy": self.name,
            "cagr": p.cagr,
            "volatility": p.volatility,
            "sharpe": p.sharpe,
            "max_drawdown": p.max_drawdown,
            "calmar": p.calmar,
            "worst_year": p.worst_year,
            "orders_per_year": self.trading["orders_per_year"],
            "fee_drag": self.trading["fee_drag"],
            "psr": p.psr,
            "beats_benchmark": self.beats_benchmark,
        }


@dataclass
class Comparison:
    entries: list[Entry]
    start: str
    end: str
    initial_cash: float
    trials: int
    risk_free: str | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    ml_skill: dict[str, float] = field(default_factory=dict)
    data_sources: dict[str, str] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([e.row() for e in self.entries]).set_index("strategy")

    def get(self, name: str) -> Entry:
        return next(e for e in self.entries if e.name == name)

    @property
    def price_only(self) -> list[str]:
        return [s for s, source in self.data_sources.items() if "price-only" in source]

    def header(self) -> str:
        return (
            f"{self.start} to {self.end}, start {money(self.initial_cash)}. "
            f"'Beats B&H' is the probability a strategy's Sharpe ratio truly beats buy-and-hold, "
            f"deflated for {self.trials} trial(s)."
        )

    def ml_summary(self) -> str | None:
        s = self.ml_skill
        if not s:
            return None
        return (
            f"ML out-of-sample skill: AUC {s['auc']:.3f} (0.5 = coin flip), accuracy "
            f"{pct(s['accuracy'])} vs a base rate of {pct(s['base_rate'])}, Brier "
            f"{s['brier']:.4f} vs {s['brier_baseline']:.4f} for always guessing the base "
            f"rate (n={s['n']})"
        )

    def to_text(self) -> str:
        table = self.table().to_string(formatters={k: f for k, f, _ in COLUMNS if k != "strategy"})
        parts = [self.header(), "", table]
        if summary := self.ml_summary():
            parts += ["", summary]
        parts += [f"skipped {name}: {why}" for name, why in self.skipped.items()]
        return "\n".join(parts)

    def to_markdown(self, sizes: pd.DataFrame | None = None) -> str:
        titles = [title for _, _, title in COLUMNS]
        lines = [
            f"### Kea backtest: {self.start} to {self.end}",
            "",
            f"Start {money(self.initial_cash)}. Sharpe is in excess of "
            f"{self.risk_free or 'zero (no cash proxy)'}. *Beats B&H* is the probability that a "
            f"strategy's Sharpe ratio genuinely beats buy-and-hold, deflated for "
            f"{self.trials} trial(s) (Bailey and López de Prado).",
            "",
            "| " + " | ".join(titles) + " |",
            "|---|" + "---:|" * (len(titles) - 1),
        ]
        for name, row in self.table().iterrows():
            cells = [name] + [fmt(row[key]) for key, fmt, _ in COLUMNS[1:]]
            lines.append("| " + " | ".join(cells) + " |")
        if summary := self.ml_summary():
            lines += ["", summary]
        if sizes is not None:
            lines += ["", sizes_markdown(sizes)]
        if self.price_only:
            lines += ["", f"**Warning:** price-only data for {', '.join(self.price_only)}."]
        lines += [f"\nSkipped {name}: {why}." for name, why in self.skipped.items()]
        return "\n".join(lines) + "\n"

    def to_json(self, sizes: pd.DataFrame | None = None) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "initial_cash": self.initial_cash,
            "trials": self.trials,
            "strategies": json.loads(self.table().reset_index().to_json(orient="records")),
            "ml_skill": self.ml_skill,
            "account_sizes": (
                None if sizes is None else json.loads(sizes.reset_index().to_json(orient="records"))
            ),
            "data_sources": self.data_sources,
            "skipped": self.skipped,
        }


COLUMNS = [
    ("strategy", str, "Strategy"),
    ("cagr", pct, "CAGR"),
    ("volatility", pct, "Volatility"),
    ("sharpe", num, "Sharpe"),
    ("max_drawdown", pct, "Max drawdown"),
    ("calmar", num, "Calmar"),
    ("worst_year", pct, "Worst year"),
    ("orders_per_year", count, "Orders/yr"),
    ("fee_drag", lambda v: pct(v, 2), "Fee drag"),
    ("psr", lambda v: pct(v, 0), "P(Sharpe>0)"),
    ("beats_benchmark", lambda v: pct(v, 0), "Beats B&H"),
]

SIZE_COLUMNS = [
    ("cagr", pct, "CAGR"),
    ("sharpe", num, "Sharpe"),
    ("fee_drag", lambda v: pct(v, 2), "Fee drag"),
    ("fees", money, "Fees paid"),
    ("orders_per_year", count, "Orders/yr"),
]


def sizes_text(sizes: pd.DataFrame) -> str:
    return sizes.to_string(formatters={k: f for k, f, _ in SIZE_COLUMNS})


def sizes_markdown(sizes: pd.DataFrame) -> str:
    lines = [
        "| Account size | " + " | ".join(t for _, _, t in SIZE_COLUMNS) + " |",
        "|---:|" + "---:|" * len(SIZE_COLUMNS),
    ]
    for size, row in sizes.iterrows():
        cells = [money(size, 0)] + [f(row[k]) for k, f, _ in SIZE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def default_benchmarks(config: Config) -> tuple[str, ...]:
    """Buy-and-hold the benchmark, plus a 60/40 portfolio when the universe has bonds."""
    return BENCHMARK_NAMES if "IEF" in config.universe.symbols else ("buy_and_hold",)


def default_strategies(config: Config) -> tuple[str, ...]:
    """The configured strategy, plus its members when it is an ensemble."""
    name = config.strategy.name
    return (*config.strategy.members, name) if name == "ensemble" else (name,)


def compare(
    config: Config,
    history: PriceHistory,
    strategies: Sequence[str] | None = None,
    benchmarks: Sequence[str] | None = None,
    initial_cash: float | None = None,
    trials: int | None = None,
) -> Comparison:
    """Backtest every strategy and benchmark from a shared start date.

    `trials` is how many strategy variants have been tried in total (default:
    the strategies compared here). Be honest with it: every configuration you
    tried and discarded counts, and the Deflated Sharpe Ratio gets stricter as
    it grows.
    """
    runners = [
        (name, Backtester(config, build_strategy(config, name), initial_cash), False)
        for name in (default_strategies(config) if strategies is None else strategies)
    ] + [
        (
            name,
            Backtester(config, build_strategy(config, name), initial_cash, risk_overlay=False),
            True,
        )
        for name in (default_benchmarks(config) if benchmarks is None else benchmarks)
    ]
    # Strategies whose warm-up exceeds the history are skipped (and reported), so a
    # short history still compares what it can.
    skipped: dict[str, str] = {}
    usable = []
    for name, runner, is_benchmark in runners:
        needed = runner.first_decision_index(history)
        if needed >= len(history) - 1:
            skipped[name] = f"needs {needed} sessions of history, only {len(history)} available"
        else:
            usable.append((name, runner, is_benchmark))
    if not any(not is_benchmark for _, _, is_benchmark in usable):
        raise ValueError("not enough history for any strategy: " + "; ".join(skipped.values()))
    runners = usable
    start_index = max(runner.first_decision_index(history) for _, runner, _ in runners)
    start = history.dates[start_index].date()
    cash_symbol = config.universe.cash_symbol
    risk_free = history.close[cash_symbol] if cash_symbol else None
    ppy = config.universe.periods_per_year

    entries = []
    for name, runner, is_benchmark in runners:
        result = runner.run(history, start=start, end=config.backtest.end)
        entries.append(
            Entry(
                name,
                result,
                performance(result.equity, ppy, risk_free),
                trading_stats(result.fills_frame(), result.equity),
                is_benchmark,
            )
        )

    contenders = [e for e in entries if not e.benchmark]
    n_trials = max(trials or config.report.trials or len(contenders), len(contenders))
    holder = next((e for e in entries if e.name == "buy_and_hold"), None)
    hurdle = per_period_sharpe(_excess(holder.result.equity, risk_free)) if holder else 0.0
    for entry in contenders:
        entry.beats_benchmark = deflated_sharpe(
            _excess(entry.result.equity, risk_free), n_trials, benchmark_sharpe=hurdle
        )

    ml_skill: dict[str, float] = {}
    for entry in contenders:
        predictions = entry.result.diagnostics.get("ml_predictions")
        if predictions is not None and not predictions.empty:
            u = config.universe
            ml_skill = score_predictions(
                predictions, history, u.symbols, u.cash_symbol, config.strategy.ml.horizon
            )
            break

    first = entries[0].result.equity
    return Comparison(
        entries=entries,
        start=first.index[0].date().isoformat(),
        end=first.index[-1].date().isoformat(),
        initial_cash=entries[0].result.initial_cash,
        trials=n_trials,
        risk_free=cash_symbol,
        skipped=skipped,
        ml_skill=ml_skill,
    )


def account_size_sensitivity(
    config: Config,
    history: PriceHistory,
    strategy: str,
    sizes: Sequence[float],
    start: date | None = None,
) -> pd.DataFrame:
    """How the same strategy fares at different account sizes under real fees.

    A flat USD 2 commission is noise on USD 100k and a tax on USD 2k. Pass the
    comparison's `start` so these numbers share its window.
    """
    cash_symbol = config.universe.cash_symbol
    risk_free = history.close[cash_symbol] if cash_symbol else None
    rows = []
    for size in sizes:
        runner = Backtester(config, build_strategy(config, strategy), size)
        result = runner.run(history, start=start, end=config.backtest.end)
        perf = performance(result.equity, config.universe.periods_per_year, risk_free)
        stats = trading_stats(result.fills_frame(), result.equity)
        rows.append(
            {
                "account_size": size,
                "cagr": perf.cagr,
                "sharpe": perf.sharpe,
                "fee_drag": stats["fee_drag"],
                "fees": stats["fees"],
                "orders_per_year": stats["orders_per_year"],
            }
        )
    return pd.DataFrame(rows).set_index("account_size")


def _excess(equity: pd.Series, risk_free: pd.Series | None) -> pd.Series:
    returns = equity.pct_change().dropna()
    if risk_free is None:
        return returns
    rf = risk_free.reindex(equity.index).ffill().pct_change().reindex(returns.index).fillna(0.0)
    return returns - rf
