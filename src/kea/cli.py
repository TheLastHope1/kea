"""Command-line interface.

kea data                  fetch market data and show where it came from
kea backtest              backtest the configured strategy
kea compare               every strategy vs benchmarks, with deflated Sharpe
kea run [--dry-run]       one autonomous decision cycle (schedule this)
kea status                account, positions, recent decisions
kea resume                clear a circuit-breaker halt after reviewing it
kea report                write the HTML dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from kea import __version__
from kea.config import Config, ConfigError, load_config
from kea.data import DataUnavailable, MarketData, PriceHistory
from kea.fmt import money, num, pct

if TYPE_CHECKING:
    from kea.agent import Agent

DEFAULT_CONFIG = Path("kea.toml")
ACCOUNT_SIZES = (2_000, 5_000, 10_000, 25_000, 100_000)

Handler = Callable[[Config, argparse.Namespace], int]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
        config = load_config(path, args.overrides or ())
        handler: Handler = args.handler
        return handler(config, args)
    except (ConfigError, DataUnavailable, RuntimeError, ValueError) as exc:
        print(f"kea: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kea", description="Kea: a risk-first autonomous trading agent."
    )
    parser.add_argument("--version", action="version", version=f"kea {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="TOML config (default: ./kea.toml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a setting, e.g. -s risk.target_vol=0.08 (repeatable)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: Handler, summary: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=summary, description=summary)
        p.set_defaults(handler=handler)
        return p

    command("data", cmd_data, "fetch market data and show its sources")

    p = command("backtest", cmd_backtest, "backtest one strategy")
    p.add_argument(
        "--strategy",
        help="trend, momentum, ml, trend_filter, ensemble, buy_and_hold or sixty_forty",
    )
    p.add_argument("--cash", type=float, help="starting capital in USD")
    p.add_argument("--start", type=date.fromisoformat, help="first decision date (YYYY-MM-DD)")
    p.add_argument("--end", type=date.fromisoformat, help="last date (YYYY-MM-DD)")

    p = command("compare", cmd_compare, "compare strategies against benchmarks")
    p.add_argument(
        "--strategies",
        type=strategy_list,
        help="comma-separated strategies (default: the configured one and its members)",
    )
    p.add_argument("--cash", type=float, help="starting capital in USD")
    p.add_argument("--trials", type=int, help="total strategy variants ever tried (for DSR)")
    p.add_argument("--sizes", action="store_true", help="also test account-size sensitivity")
    p.add_argument("--json", type=Path, help="write the comparison as JSON")
    p.add_argument("--markdown", type=Path, help="write the comparison as Markdown")
    p.add_argument("--html", type=Path, help="write the HTML report")

    p = command("run", cmd_run, "run one autonomous decision cycle")
    p.add_argument("--dry-run", action="store_true", help="plan, but change nothing")

    command("status", cmd_status, "show the account and recent decisions")
    command("resume", cmd_resume, "clear a circuit-breaker halt")

    p = command("report", cmd_report, "write the HTML dashboard (track record + backtests)")
    p.add_argument("--html", type=Path, default=Path("reports/index.html"))
    p.add_argument("--strategies", type=strategy_list, help="strategies to compare")
    p.add_argument("--cash", type=float, help="starting capital in USD for the backtests")
    p.add_argument("--no-backtest", action="store_true", help="track record only (fast)")
    p.add_argument(
        "--study",
        type=Path,
        action="append",
        default=[],
        metavar="CONFIG",
        help="add a study from another config, e.g. config/century.toml (repeatable)",
    )
    p.add_argument("--fragment", action="store_true", help="omit <html>/<head>/<body> wrappers")
    return parser


# ------------------------------------------------------------------ helpers


def strategy_list(text: str) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()]


def load_market(config: Config) -> tuple[MarketData, PriceHistory]:
    market = MarketData(config.data, config.universe.asset_class)
    history = market.history(config.universe.all_symbols)
    if market.price_only:
        print(
            f"WARNING: price-only data (dividends ignored) for {', '.join(market.price_only)}; "
            "returns of income-paying assets are understated.",
            file=sys.stderr,
        )
    return market, history


def build_agent(config: Config) -> Agent:
    from kea.agent import Agent
    from kea.brokers import build_broker
    from kea.ledger import Ledger
    from kea.strategies import build_strategy

    return Agent(
        config,
        MarketData(config.data, config.universe.asset_class),
        build_strategy(config),
        build_broker(config),
        Ledger(config.broker.state_dir),
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ----------------------------------------------------------------- commands


def cmd_data(config: Config, args: argparse.Namespace) -> int:
    market, history = load_market(config)
    rows = []
    for symbol in history.symbols:
        series = history.close[symbol].dropna()
        rows.append(
            {
                "symbol": symbol,
                "first": series.index[0].date(),
                "last": series.index[-1].date(),
                "bars": len(series),
                "close": num(series.iloc[-1]),
                "source": market.sources.get(symbol, "?"),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_backtest(config: Config, args: argparse.Namespace) -> int:
    from kea.backtest import Backtester
    from kea.metrics import performance, trading_stats
    from kea.strategies import BENCHMARK_NAMES, build_strategy

    _, history = load_market(config)
    name = args.strategy or config.strategy.name
    runner = Backtester(
        config,
        build_strategy(config, name),
        args.cash,
        risk_overlay=name not in BENCHMARK_NAMES,
    )
    result = runner.run(history, start=args.start or config.backtest.start, end=args.end)
    cash_symbol = config.universe.cash_symbol
    risk_free = history.close[cash_symbol] if cash_symbol else None
    perf = performance(result.equity, config.universe.periods_per_year, risk_free)
    stats = trading_stats(result.fills_frame(), result.equity)
    lines = [
        f"{name}: {perf.start} to {perf.end} ({perf.years:.1f} years)",
        f"  start / end      {money(result.initial_cash)} -> {money(result.equity.iloc[-1])}",
        f"  CAGR             {pct(perf.cagr, 2)}",
        f"  volatility       {pct(perf.volatility)}",
        f"  Sharpe (excess)  {num(perf.sharpe)}   P(true Sharpe > 0) {pct(perf.psr, 0)}",
        f"  max drawdown     {pct(perf.max_drawdown)}   longest {perf.longest_drawdown_days} days",
        f"  worst year       {pct(perf.worst_year)}   best {pct(perf.best_year)}",
        f"  orders per year  {stats['orders_per_year']:.0f}",
        f"  fees             {money(stats['fees'])} ({pct(stats['fee_drag'], 2)} a year)",
    ]
    lines += [
        f"  circuit breaker would have halted the agent on {day}: {reason}"
        for day, reason in result.breaker_trips
    ]
    print("\n".join(lines))
    return 0


def cmd_compare(config: Config, args: argparse.Namespace) -> int:
    from kea.research import account_size_sensitivity, compare, sizes_text

    market, history = load_market(config)
    comparison = compare(
        config, history, strategies=args.strategies, initial_cash=args.cash, trials=args.trials
    )
    comparison.data_sources = dict(market.sources)
    print(comparison.to_text())
    sizes = None
    if args.sizes:
        sizes = account_size_sensitivity(
            config,
            history,
            config.strategy.name,
            ACCOUNT_SIZES,
            start=date.fromisoformat(comparison.start),
        )
        print(f"\nAccount size vs fees ({config.strategy.name}, {config.execution.fees} fees):")
        print(sizes_text(sizes))
    if args.json:
        write_text(args.json, json.dumps(comparison.to_json(sizes), indent=2) + "\n")
    if args.markdown:
        write_text(args.markdown, comparison.to_markdown(sizes))
    if args.html:
        from kea.report import write_report

        write_report(args.html, config, history, comparison, sizes=sizes)
        print(f"\nreport written to {args.html}")
    return 0


def cmd_run(config: Config, args: argparse.Namespace) -> int:
    agent = build_agent(config)
    report = agent.run(dry_run=args.dry_run)
    lines = [
        f"[{report.status}] as of {report.as_of}: {report.message}",
        f"broker: {agent.broker.describe()}",
    ]
    if account := report.account:
        lines.append(f"equity {money(account.equity)}, cash {money(account.cash)}")
        lines += [f"note: {note}" for note in account.notes]
    for fill in report.fills:
        lines.append(
            f"filled {fill.side} {fill.quantity:g} {fill.symbol} @ {fill.price:.2f} "
            f"(fees {fill.fees:.2f})"
        )
    for ticket in report.tickets:
        o = ticket.order
        lines.append(
            f"order {o.side} {o.quantity:g} {o.symbol} (~{money(o.notional)}) -> "
            f"{ticket.status} {ticket.message}".rstrip()
        )
    if warning := report.details.get("warning"):
        lines.append(f"WARNING: {warning}")
    print("\n".join(lines))
    return 2 if report.status == "halted" else 0


def cmd_status(config: Config, args: argparse.Namespace) -> int:
    from kea.brokers import build_broker
    from kea.ledger import Ledger

    ledger = Ledger(config.broker.state_dir)
    state = ledger.load_state()
    broker = build_broker(config)
    lines = [f"broker: {broker.describe()}"]
    if state.halted:
        lines.append(f"HALTED since {state.halted['since']}: {state.halted['reason']}")
    lines.append(f"last session handled: {state.last_processed or 'never'}")
    lines.append(f"last rebalance: {state.last_rebalance or 'never'}")
    track = ledger.equity()
    if not track.empty:
        first, last = track["equity"].iloc[0], track["equity"].iloc[-1]
        lines.append(
            f"track record: {track.index[0].date()} to {track.index[-1].date()}, "
            f"{money(first)} -> {money(last)} ({pct(last / first - 1, 2)})"
        )
        positions = broker.account({}).positions
        held = ", ".join(f"{s} {q:g}" for s, q in sorted(positions.items())) or "none"
        lines.append(f"positions: {held}")
    for event in ledger.journal(limit=5):
        status = event.get("status", event.get("kind"))
        lines.append(f"  {event['ts']}  [{status}] {event.get('message', '')}")
    print("\n".join(lines))
    return 0


def cmd_resume(config: Config, args: argparse.Namespace) -> int:
    from kea.agent import resume
    from kea.ledger import Ledger

    print(resume(Ledger(config.broker.state_dir)))
    return 0


def cmd_report(config: Config, args: argparse.Namespace) -> int:
    from kea.ledger import Ledger
    from kea.report import Study, write_report
    from kea.research import compare

    market, history = load_market(config)
    comparison = None
    if not args.no_backtest:
        comparison = compare(config, history, strategies=args.strategies, initial_cash=args.cash)
        comparison.data_sources = dict(market.sources)
    studies = []
    for path in args.study:
        study_config = load_config(path)
        study_market, study_history = load_market(study_config)
        study = compare(study_config, study_history)
        study.data_sources = dict(study_market.sources)
        studies.append(Study(path.stem, study_config, study))
    write_report(
        args.html,
        config,
        history,
        comparison,
        ledger=Ledger(config.broker.state_dir),
        standalone=not args.fragment,
        studies=studies,
    )
    print(f"report written to {args.html}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
