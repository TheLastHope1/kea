"""The HTML report: the paper track record and the backtest evidence on one page.

The page is self-contained (inline CSS, JS and data; web fonts are optional),
so it can be opened from disk, committed next to the ledger, or published.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from kea import __version__
from kea.config import Config
from kea.data.history import PriceHistory
from kea.fmt import count, money, num, pct
from kea.ledger import Ledger
from kea.metrics import annual_returns, drawdown_series
from kea.research import Comparison, Entry

TITLE = "Kea Trading Report"
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family='
    "Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500"
    '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
)
COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"]
ASSET_CLASSES = {
    **dict.fromkeys(
        ["SPY", "VOO", "IVV", "SPLG", "VTI", "VONE", "VTHR", "QQQ", "QQQM", "IWM", "VTWO"],
        "US stocks",
    ),
    **dict.fromkeys(
        ["EFA", "EEM", "VEA", "VWO", "VXUS", "ACWX", "IXUS", "EMXC", "IEFA", "IEMG"], "Intl stocks"
    ),
    **dict.fromkeys(["VNQ", "VNQI", "SCHH"], "Real estate"),
    **dict.fromkeys(
        ["TLT", "IEF", "LQD", "VCIT", "VGIT", "BND", "AGG", "SHY", "IEI", "TIP", "EMB"], "Bonds"
    ),
    **dict.fromkeys(["GLD", "IAU", "GLDM"], "Gold"),
    **dict.fromkeys(["DBC", "PDBC", "GSG"], "Commodities"),
    **dict.fromkeys(["BIL", "SHV", "SGOV"], "Cash"),
}
CLASS_ORDER = ["US stocks", "Intl stocks", "Real estate", "Bonds", "Gold", "Commodities", "Other"]
MAX_POINTS = 700

esc = html.escape


INDEX_NAMES = {"MARKET": "the US market", "TBILL": "T-bills"}


def strategy_label(name: str, config: Config) -> str:
    benchmark = INDEX_NAMES.get(config.universe.benchmark, config.universe.benchmark)
    labels = {
        "trend": "Trend following",
        "momentum": "Dual momentum",
        "ml": "ML forecaster",
        "trend_filter": "200-day filter",
        "ensemble": "Ensemble",
        "buy_and_hold": f"Buy and hold {benchmark}",
        "sixty_forty": "60/40 stocks and bonds",
    }
    return labels.get(name, name)


# --------------------------------------------------------------- page parts


@dataclass(frozen=True)
class Line:
    name: str
    values: pd.Series
    color: str
    area: bool = False


@dataclass(frozen=True)
class Study:
    """An extra comparison shown after the main backtest, such as the century check."""

    key: str
    config: Config
    comparison: Comparison


def entity_colors(names: Sequence[str]) -> dict[str, str]:
    """Colour follows the entity: benchmarks keep their colour in every chart."""
    fixed = {"buy_and_hold": "--s2", "sixty_forty": "--s3"}
    taken = {fixed[n] for n in names if n in fixed}
    free = iter(c for c in COLORS if c not in taken)
    return {name: fixed.get(name) or next(free) for name in names}


def write_report(
    path: Path,
    config: Config,
    history: PriceHistory,
    comparison: Comparison | None = None,
    *,
    ledger: Ledger | None = None,
    sizes: pd.DataFrame | None = None,
    standalone: bool = True,
    studies: Sequence[Study] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page = render_report(config, history, comparison, ledger, sizes, standalone, studies)
    path.write_text(page)


def render_report(
    config: Config,
    history: PriceHistory,
    comparison: Comparison | None = None,
    ledger: Ledger | None = None,
    sizes: pd.DataFrame | None = None,
    standalone: bool = True,
    studies: Sequence[Study] = (),
) -> str:
    sections = [
        masthead(config, history, comparison, live=ledger is not None),
        paper_section(config, ledger) if ledger is not None else "",
        backtest_section(config, history, comparison) if comparison is not None else "",
        *(study_section(study) for study in studies),
        costs_section(config, comparison, sizes),
        method_section(config),
        footer(),
    ]
    body = '<main class="page">' + "\n".join(s for s in sections if s) + "</main>"
    css = (files("kea") / "assets" / "report.css").read_text()
    js = (files("kea") / "assets" / "report.js").read_text()
    head = f"<title>{TITLE}</title>\n{FONTS}\n<style>\n{css}</style>"
    script = f"<script>\n{js}</script>"
    if not standalone:  # e.g. published into a page skeleton that supplies <head>/<body>
        return f"{head}\n{body}\n{script}\n"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}\n</head>\n<body>\n{body}\n{script}\n</body>\n</html>\n"
    )


BRAND_MARK = (
    '<svg viewBox="0 0 32 32" aria-hidden="true"><path fill="currentColor" d="M27 7c-4-4-11-4-15 '
    '0-3 3-4 7-3 11l-5 6c-1 1 0 3 2 2l7-3c4 1 9 0 12-3 4-4 4-9 2-13z"/>'
    '<path fill="var(--flash)" d="M15 17c3 0 6 2 7 5-3 2-7 2-10 1z"/>'
    '<circle cx="21" cy="11" r="1.8" fill="var(--surface)"/></svg>'
)


def masthead(
    config: Config, history: PriceHistory, comparison: Comparison | None, live: bool
) -> str:
    u = config.universe
    universe = ", ".join(u.symbols)
    sources = sorted(set((comparison.data_sources if comparison else {}).values()))
    meta = [
        f"<span>Data through <strong>{history.last_date:%d %b %Y}</strong></span>",
        f"<span>Universe <strong class='mono'>{esc(universe)}</strong></span>",
        f"<span>Cash parked in <strong class='mono'>{esc(u.cash_symbol or 'cash')}</strong></span>",
    ]
    if sources:
        meta.append(f"<span>Sources <strong>{esc(', '.join(sources))}</strong></span>")
    warning = ""
    if comparison is not None and comparison.price_only:
        warning = (
            '<div class="callout warn"><strong>Price-only data</strong><span>No dividend history was '
            f"available for {esc(', '.join(comparison.price_only))}, so their returns are understated "
            "by their yield. Treat these numbers as a rough draft.</span></div>"
        )
    period = "week" if config.strategy.rebalance == "weekly" else "month"
    shows = [
        s
        for s, on in (
            ("its live paper-trading record", live),
            ("a backtest", comparison is not None),
        )
        if on
    ]
    evidence = " and ".join(shows) or "its configuration"
    return f"""
<header class="masthead">
  <div class="brand">{BRAND_MARK}<span>Kea · autonomous trading agent</span></div>
  <h1>Kea's track record, and the evidence behind it</h1>
  <p class="lede">Kea rebalances a diversified ETF portfolio once a {period} using trend, momentum and
  a machine-learning forecaster, with a volatility cap and circuit breakers. This page shows {evidence},
  with real broker fees, judged against simply buying and holding.</p>
  <div class="meta-row">{"".join(meta)}</div>
  {warning}
</header>"""


def paper_section(config: Config, ledger: Ledger) -> str:
    track = ledger.equity()
    events = [e for e in ledger.journal() if e.get("kind") == "run"]
    state = ledger.load_state()
    head = """
  <div class="section-head"><span class="eyebrow">Live paper trading</span>
  <h2>The forward record</h2>
  <p class="secondary">The only performance that cannot be overfitted: decisions made in real time,
  filled at the next session's open, committed to git every day.</p></div>"""
    if len(track) < 2:
        if len(track):
            lead = f"Paper trading started on {track.index[0]:%d %b %Y}."
            body = "The first orders fill at the next session's open, and the record grows by one point each trading day."
        else:
            lead = "Paper trading has not started yet."
            body = "The record begins with the first scheduled run and grows by one point each trading day."
        return f"""<section id="paper">{head}
  <div class="callout"><strong>{lead}</strong>
  <span>{body} Check back in a few weeks; judge it after a few months, not a few days.</span></div>
  {journal_table(events)}</section>"""

    equity, bench = track["equity"], track["benchmark"].astype(float)
    kea_return = equity.iloc[-1] / equity.iloc[0] - 1
    bench_return = bench.iloc[-1] / bench.iloc[0] - 1 if bench.notna().all() else math.nan
    drawdown = float(-drawdown_series(equity).min())
    tiles = [
        tile("Account value", money(equity.iloc[-1]), f"started at {money(equity.iloc[0])}"),
        tile(
            "Return since start",
            signed_pct(kea_return),
            f"{config.universe.benchmark} {signed_pct(bench_return)}",
            tone(kea_return),
        ),
        tile(
            "Worst drawdown so far",
            pct(drawdown),
            f"breaker trips at {pct(config.risk.halt_drawdown, 0)}",
        ),
        tile("Sessions recorded", count(len(track)), f"since {track.index[0]:%d %b %Y}"),
    ]
    if state.halted:
        tiles.append(tile("Status", "Halted", esc(state.halted["reason"]), "down"))
    lines = [
        Line("Kea paper account", 100 * equity / equity.iloc[0], "--s1"),
        Line(f"{config.universe.benchmark} buy and hold", 100 * bench / bench.iloc[0], "--s2"),
    ]
    chart = line_chart(
        "paper-chart",
        "Paper account vs the benchmark",
        "Both indexed to 100 on the first day",
        lines,
        fmt="index",
        height=280,
    )
    return f"""<section id="paper">{head}
  <div class="tiles">{"".join(tiles)}</div>
  {chart}
  {positions_table(events)}
  {journal_table(events)}
</section>"""


def backtest_section(config: Config, history: PriceHistory, comparison: Comparison) -> str:
    entries = {e.name: e for e in comparison.entries}
    main = entries.get(config.strategy.name) or next(
        e for e in comparison.entries if not e.benchmark
    )
    bench = entries.get("buy_and_hold")
    p = main.performance
    bp = bench.performance if bench else None
    rf = config.universe.cash_symbol or "zero"
    tiles = [
        tile(
            "Annual return (CAGR)",
            pct(p.cagr),
            f"{strategy_label('buy_and_hold', config)}: {pct(bp.cagr)}" if bp else None,
        ),
        tile(
            "Sharpe ratio",
            num(p.sharpe),
            f"benchmark {num(bp.sharpe)}; excess of {esc(rf)}" if bp else None,
        ),
        tile(
            "Worst peak-to-trough fall",
            pct(p.max_drawdown),
            f"benchmark {pct(bp.max_drawdown)}" if bp else None,
        ),
        tile(
            "Chance it truly beats buy and hold",
            pct(main.beats_benchmark, 0),
            f"risk-adjusted, deflated for {comparison.trials} strategies tried",
        ),
    ]
    contenders = [main] + [entries[n] for n in ("buy_and_hold", "sixty_forty") if n in entries]
    growth, drawdowns = growth_and_drawdowns("", config, comparison, contenders)
    return f"""<section id="backtest">
  <div class="section-head"><span class="eyebrow">Backtest · {comparison.start} to {comparison.end}</span>
  <h2>{esc(strategy_label(main.name, config))} against simply holding the market</h2>
  <p class="secondary">Signals at each close, fills at the next open, Tiger's fee schedule, no leverage.
  Parameters were fixed from published research before testing, not tuned to this data.</p></div>
  <div class="tiles">{"".join(tiles)}</div>
  {skill_callout(main, bench, comparison)}
  {growth}
  {drawdowns}
  {allocation_chart(config, main)}
  {comparison_table(config, comparison)}
  {annual_table(config, contenders)}
</section>"""


def growth_and_drawdowns(
    key: str, config: Config, comparison: Comparison, entries: Sequence[Entry]
) -> tuple[str, str]:
    colors = entity_colors([e.name for e in entries])
    prefix = f"{key}-" if key else ""
    growth = line_chart(
        f"{prefix}growth-chart",
        f"Growth of {money(comparison.initial_cash, 0)}",
        "Log scale, after fees and slippage",
        [Line(strategy_label(e.name, config), e.result.equity, colors[e.name]) for e in entries],
        fmt="money",
        log=True,
        height=340,
    )
    drawdowns = line_chart(
        f"{prefix}drawdown-chart",
        "Drawdowns",
        "Distance below the previous high",
        [
            Line(
                strategy_label(e.name, config),
                drawdown_series(e.result.equity),
                colors[e.name],
                area=i == 0,
            )
            for i, e in enumerate(entries)
        ],
        fmt="pct",
        zero="top",
        height=240,
    )
    return growth, drawdowns


def study_section(study: Study) -> str:
    config, comparison = study.config, study.comparison
    entries = {e.name: e for e in comparison.entries}
    holder = entries.get("buy_and_hold")
    rules = [e for e in comparison.entries if not e.benchmark and e.name != "ml"]
    charted = [*rules[:3], *([holder] if holder else [])]
    growth, drawdowns = growth_and_drawdowns(study.key, config, comparison, charted)
    years = charted[0].performance.years
    title = config.report.title or f"Study: {', '.join(config.universe.symbols)}"
    summary = (
        f"<p class='secondary'>{esc(config.report.summary)}</p>" if config.report.summary else ""
    )
    callout = ""
    if holder is not None and rules:
        worst = [e.performance.max_drawdown for e in rules]
        parts = [
            f"Over {years:.0f} years, buy and hold fell as much as {pct(holder.performance.max_drawdown, 0)}; "
            f"the rules' worst falls were {pct(min(worst), 0)} to {pct(max(worst), 0)}."
        ]
        if holder.performance.cagr > 0:
            kept = [e.performance.cagr / holder.performance.cagr for e in rules]
            parts.append(
                f"They kept {pct(min(kept), 0)} to {pct(max(kept), 0)} of its annual return."
            )
        edges = [e.beats_benchmark for e in rules if e.beats_benchmark is not None]
        if edges:
            parts.append(
                "The chance any of them truly beats buy and hold on a risk-adjusted basis is at most "
                f"{pct(max(edges), 0)} after the {comparison.trials}-trial haircut."
            )
        callout = (
            '<div class="callout"><strong>The short version</strong><span>'
            + " ".join(parts)
            + "</span></div>"
        )
    return f"""<section id="{esc(study.key)}">
  <div class="section-head"><span class="eyebrow">Study · {comparison.start[:4]} to {comparison.end[:4]}</span>
  <h2>{esc(title)}</h2>{summary}</div>
  {callout}
  {growth}
  {drawdowns}
  {comparison_table(config, comparison, heading="Every rule in this study")}
</section>"""


def skill_callout(main: Entry, bench: Entry | None, comparison: Comparison) -> str:
    p = main.performance
    lines = []
    if bench is not None:
        b = bench.performance
        lines.append(
            f"It returned {pct(p.cagr)} a year against {pct(b.cagr)} for buy and hold, with "
            f"{pct(p.max_drawdown)} as its worst fall against {pct(b.max_drawdown)}."
        )
    edge = main.beats_benchmark
    if edge is not None and not math.isnan(edge):
        verdict = (
            "strong evidence of an edge"
            if edge >= 0.95
            else "suggestive, not conclusive"
            if edge >= 0.8
            else "no reliable edge over simply holding the market"
        )
        lines.append(
            f"After discounting for the {comparison.trials} strategies tried, the chance its "
            f"risk-adjusted return genuinely beats buy and hold is {pct(edge, 0)}: {verdict}."
        )
    s = comparison.ml_skill
    if s:
        lines.append(
            f"The ML forecaster's out-of-sample AUC is {s['auc']:.3f}, where 0.5 is a coin flip"
            + (", so it adds little on its own." if s["auc"] < 0.55 else ".")
        )
    if not lines:
        return ""
    return (
        '<div class="callout"><strong>The short version</strong><span>'
        + " ".join(lines)
        + "</span></div>"
    )


def allocation_chart(config: Config, entry: Entry) -> str:
    weights = entry.result.weights
    frame = asset_class_weights(weights, config)
    if frame.empty:
        return ""
    frame = downsample(frame)
    names = list(frame.columns)
    colors = [("--cash" if n == "Cash" else COLORS[i % len(COLORS)]) for i, n in enumerate(names)]
    spec = {
        "kind": "stack",
        "title": "What Kea held",
        "height": 260,
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": [
            {"name": n, "color": c, "values": clean(frame[n])}
            for n, c in zip(names, colors, strict=True)
        ],
    }
    legend = [(n, c, "box") for n, c in zip(names, colors, strict=True)]
    yearly = frame.groupby(frame.index.year).mean()
    table = data_table(
        ["Year", *names],
        [[str(y), *(pct(v, 0) for v in row)] for y, row in yearly.iterrows()],
    )
    return figure(
        "allocation-chart",
        "What Kea held",
        "Share of the account by asset class; grey is cash or T-bills",
        spec,
        legend,
        table,
    )


def asset_class_weights(weights: pd.DataFrame, config: Config) -> pd.DataFrame:
    risky = list(config.universe.symbols)
    known = all(s in ASSET_CLASSES for s in risky)
    groups: dict[str, pd.Series] = {}
    for symbol in weights.columns:
        if symbol == config.universe.cash_symbol:
            continue
        if known:
            group = ASSET_CLASSES[symbol]
        else:
            group = symbol if len(risky) <= len(COLORS) - 1 else "Other"
        groups[group] = groups.get(group, 0) + weights[symbol]
    frame = pd.DataFrame(groups)
    order = [c for c in CLASS_ORDER if c in frame] + [c for c in frame if c not in CLASS_ORDER]
    frame = frame[order].clip(lower=0)
    frame["Cash"] = (1 - frame.sum(axis=1)).clip(lower=0)
    return frame.loc[:, frame.max() > 0.005]


def comparison_table(
    config: Config, comparison: Comparison, heading: str = "Every strategy, same window, same fees"
) -> str:
    rows, classes = [], []
    for e in comparison.entries:
        p, t = e.performance, e.trading
        rows.append(
            [
                esc(strategy_label(e.name, config))
                + (" <span class='muted'>(benchmark)</span>" if e.benchmark else ""),
                pct(p.cagr),
                pct(p.volatility),
                num(p.sharpe),
                pct(p.max_drawdown),
                pct(p.worst_year),
                count(t["orders_per_year"]),
                pct(t["fee_drag"], 2),
                "n/a" if e.beats_benchmark is None else pct(e.beats_benchmark, 0),
            ]
        )
        classes.append(
            "featured" if e.name == config.strategy.name else "benchmark" if e.benchmark else ""
        )
    headers = [
        "Strategy",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Max drawdown",
        "Worst year",
        "Orders/yr",
        "Fee drag",
        "Beats buy and hold",
    ]
    return (
        f'<div class="section-head"><h3>{esc(heading)}</h3>'
        '<p class="secondary">Sharpe is measured in excess of T-bills. "Beats buy and hold" is the '
        "probability that a strategy's Sharpe ratio genuinely exceeds buy-and-hold's, after "
        "discounting for how many strategies were tried (the deflated Sharpe ratio). Benchmarks are "
        "not scored.</p></div>" + data_table(headers, rows, classes, raw=True)
    )


def annual_table(config: Config, entries: Sequence[Entry]) -> str:
    years = pd.DataFrame(
        {strategy_label(e.name, config): annual_returns(e.result.equity) for e in entries}
    )
    rows = [[str(y), *(signed_pct(v) for v in row)] for y, row in years.iterrows()]
    return (
        "<details><summary>Year by year</summary>"
        + data_table(["Year", *years.columns], rows)
        + "</details>"
    )


def costs_section(config: Config, comparison: Comparison | None, sizes: pd.DataFrame | None) -> str:
    fees = config.execution
    if fees.fees == "tiger_nz":
        schedule = (
            "Tiger Brokers (NZ) charges USD 2 per US order of up to 200 shares, plus small "
            "settlement and regulatory pass-through fees. NZD to USD conversion costs another 0.35%."
        )
    else:
        schedule = f"Fees are modelled as {fees.fee_bps:g} basis points of each trade."
    main = None
    if comparison is not None:
        main = next((e for e in comparison.entries if e.name == config.strategy.name), None)
    stats = ""
    if main is not None:
        t = main.trading
        stats = (
            f"<p>The backtest places about <strong>{count(t['orders_per_year'])} orders a year</strong> "
            f"and pays <strong>{money(t['fees'])}</strong> in fees in total, a drag of "
            f"<strong>{pct(t['fee_drag'], 2)}</strong> a year on a {money(comparison.initial_cash, 0)} account."
            "</p>"
        )
    size_table = ""
    if sizes is not None:
        rows = [
            [
                money(size, 0),
                pct(r["cagr"]),
                num(r["sharpe"]),
                pct(r["fee_drag"], 2),
                money(r["fees"]),
                count(r["orders_per_year"]),
            ]
            for size, r in sizes.iterrows()
        ]
        size_table = data_table(
            ["Account size", "CAGR", "Sharpe", "Fee drag", "Fees paid", "Orders/yr"], rows
        )
    return f"""<section id="costs">
  <div class="section-head"><span class="eyebrow">Costs</span><h2>Fees decide whether a small account can win</h2>
  <p class="secondary">{schedule} A flat fee is noise on a large account and a heavy tax on a small one,
  which is why Kea skips trades smaller than {pct(fees.drift_band, 0)} of the account or {money(fees.min_trade_value, 0)}.</p></div>
  {stats}
  {size_table}
</section>"""


def method_section(config: Config) -> str:
    r, e = config.risk, config.execution
    return f"""<section id="method">
  <div class="section-head"><span class="eyebrow">Method</span><h2>How Kea decides, and what this does not prove</h2></div>
  <div class="notes">
    <div><h3>Each rebalance</h3><ul>
      <li>Trend: hold each ETF in proportion to how many of its 1, 3, 6 and 12-month returns beat T-bills, risk-weighted by inverse volatility.</li>
      <li>Momentum: the top {config.strategy.momentum.top_k} ETFs by 12-1 month return, only if they beat T-bills.</li>
      <li>ML: a gradient-boosted classifier refit each quarter on data available then, predicting which ETFs beat cash next month.</li>
      <li>The three are averaged, capped at {pct(r.max_weight, 0)} per ETF, and scaled down whenever predicted volatility exceeds {pct(r.target_vol, 0)} a year.</li>
    </ul></div>
    <div><h3>Safety rails</h3><ul>
      <li>Never leveraged or short; a {pct(e.cash_buffer, 0)} cash buffer absorbs overnight gaps.</li>
      <li>Halts after a {pct(r.halt_drawdown, 0)} drawdown or a {pct(r.halt_daily_loss, 0)} day, and waits for a human to resume.</li>
      <li>Skips trading when data is stale or a price moves more than {pct(r.max_price_jump, 0)} in a day.</li>
      <li>A real-money Tiger account is refused unless two separate switches are set by hand.</li>
    </ul></div>
    <div><h3>Not in these numbers</h3><ul>
      <li>Taxes: US dividend withholding (15% with a W-8BEN) and NZ tax on foreign investments.</li>
      <li>Currency: returns are in USD; an NZD investor also carries NZD/USD moves and conversion costs.</li>
      <li>Hindsight in the ETF list itself, and a sample that holds only a few full market cycles.</li>
      <li>Past performance, simulated or live, does not guarantee future results. This is research, not advice.</li>
    </ul></div>
  </div>
</section>"""


def footer() -> str:
    stamp = datetime.now(UTC).strftime("%d %b %Y %H:%M UTC")
    return f"<footer>Generated by Kea {__version__} on {stamp}. Paper trading only unless stated otherwise.</footer>"


# ------------------------------------------------------------ building blocks


def tile(label: str, value: str, note: str | None = None, value_class: str = "") -> str:
    note_html = f'<span class="tile-note">{note}</span>' if note else ""
    cls = f"tile-value {value_class}".strip()
    return f'<div class="tile"><span class="tile-label">{esc(label)}</span><span class="{cls}">{value}</span>{note_html}</div>'


def tone(value: float) -> str:
    if value is None or math.isnan(value) or value == 0:
        return ""
    return "up" if value > 0 else "down"


def signed_pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return ("+" if value > 0 else "") + pct(value, digits)


def downsample(frame: pd.DataFrame | pd.Series) -> Any:
    if len(frame) <= MAX_POINTS:
        return frame
    weekly = frame.resample("W-FRI").last().dropna(how="all")
    if len(weekly) <= MAX_POINTS:
        return weekly
    return frame.resample("ME").last().dropna(how="all")


def clean(values: Iterable[float]) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        out.append(None if v is None or not math.isfinite(v) else float(f"{v:.6g}"))
    return out


def line_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    lines: Sequence[Line],
    fmt: str,
    log: bool = False,
    zero: str | None = None,
    height: int = 300,
) -> str:
    frame = downsample(pd.DataFrame({line.name: line.values for line in lines}))
    spec: dict[str, Any] = {
        "kind": "line",
        "title": title,
        "format": fmt,
        "log": log,
        "height": height,
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": [
            {
                "name": line.name,
                "color": line.color,
                "values": clean(frame[line.name]),
                "area": line.area,
            }
            for line in lines
        ],
    }
    if zero:
        spec["zero"] = zero
    legend = [(line.name, line.color, "line") for line in lines]
    yearly = frame.groupby(frame.index.year).last()
    value = {"money": money, "pct": lambda v: pct(v), "index": lambda v: num(v, 1)}[fmt]
    table = data_table(
        ["Year end", *frame.columns],
        [[str(y), *(value(v) for v in row)] for y, row in yearly.iterrows()],
    )
    return figure(chart_id, title, subtitle, spec, legend, table)


def figure(
    chart_id: str,
    title: str,
    subtitle: str,
    spec: dict[str, Any],
    legend: Sequence[tuple[str, str, str]],
    table: str,
) -> str:
    keys = "".join(
        f'<li><span class="key{" key-box" if kind == "box" else ""}" style="background:var({color})"></span>{esc(name)}</li>'
        for name, color, kind in legend
    )
    legend_html = f'<ul class="legend">{keys}</ul>' if len(legend) > 1 else ""
    data = json.dumps(spec, separators=(",", ":")).replace("</", "<\\/")
    return f"""<figure class="chart" id="{chart_id}" tabindex="0" aria-label="{esc(title)}">
  <div class="chart-head"><h3>{esc(title)}</h3><span class="chart-sub">{esc(subtitle)}</span></div>
  {legend_html}
  <div class="chart-plot"><div class="chart-tip" hidden></div></div>
  <script type="application/json">{data}</script>
  <details><summary>Table view</summary>{table}</details>
</figure>"""


def data_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    row_classes: Sequence[str] | None = None,
    raw: bool = False,
) -> str:
    head = "".join(f"<th scope='col'>{esc(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = row_classes[i] if row_classes else ""
        cells = "".join(f"<td>{c if raw else esc(c)}</td>" for c in row)
        body.append(f'<tr class="{cls}">{cells}</tr>' if cls else f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def latest_account(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("account"):
            return event["account"]
    return None


def positions_table(events: Sequence[dict[str, Any]]) -> str:
    account = latest_account(events)
    if not account or not account.get("positions"):
        return ""
    equity = account["equity"] or 1
    rows = [
        [esc(symbol), num(qty, 0 if float(qty).is_integer() else 3)]
        for symbol, qty in sorted(account["positions"].items())
    ]
    rows.append(["Cash", money(account["cash"])])
    return (
        f'<div class="section-head"><h3>Holdings</h3><p class="secondary">Account value {money(equity)} at the last run.</p></div>'
        + data_table(["Position", "Shares"], rows)
    )


def journal_table(events: Sequence[dict[str, Any]], limit: int = 12) -> str:
    if not events:
        return ""
    rows, classes = [], []
    for event in list(reversed(events))[:limit]:
        status = str(event.get("status", ""))
        orders = event.get("orders") or []
        rows.append(
            [
                esc(str(event.get("as_of") or event.get("ts", ""))[:10]),
                f'<span class="pill pill-{esc(status)}">{esc(status)}</span>',
                f'<span class="wrap">{esc(str(event.get("message", "")))}</span>',
                count(len(orders)),
            ]
        )
        classes.append("")
    table = data_table(["Session", "Decision", "Reason", "Orders"], rows, classes, raw=True)
    return f'<div class="section-head"><h3>Decision journal</h3><p class="secondary">The agent records why it acted, or why it did nothing.</p></div>{table}'
