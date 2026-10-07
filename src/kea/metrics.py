"""Performance statistics, including the ones that tell you when to stop celebrating.

Besides the usual CAGR / Sharpe / drawdown, this module implements:

* the Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012): the
  probability that the true Sharpe ratio is above zero, given the sample length
  and how skewed and fat-tailed the returns are;
* the Deflated Sharpe Ratio (Bailey & López de Prado, 2014): the same
  probability after accounting for how many strategy variants were tried. Try
  enough variants and one will look brilliant by luck alone; DSR prices that in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Performance:
    start: str
    end: str
    years: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    longest_drawdown_days: int
    calmar: float
    best_year: float
    worst_year: float
    positive_months: float
    skew: float
    kurtosis: float
    psr: float

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def performance(
    equity: pd.Series, periods_per_year: int, risk_free: pd.Series | None = None
) -> Performance:
    """Summary statistics for a daily equity curve.

    `risk_free` is an optional price series for the cash proxy; Sharpe and Sortino
    are then computed on returns in excess of it, which matters when cash itself
    paid 5% a year.
    """
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    excess = returns
    if risk_free is not None:
        rf = risk_free.reindex(equity.index).ffill().pct_change().reindex(returns.index).fillna(0.0)
        excess = returns - rf
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    downside = float(np.sqrt((np.minimum(excess, 0) ** 2).mean()) * math.sqrt(periods_per_year))
    mean_excess = float(excess.mean() * periods_per_year)
    drawdown = drawdown_series(equity)
    max_dd = float(-drawdown.min())
    yearly = annual_returns(equity)
    monthly = equity.resample("ME").last().pct_change().dropna()
    return Performance(
        start=equity.index[0].date().isoformat(),
        end=equity.index[-1].date().isoformat(),
        years=years,
        total_return=float(total),
        cagr=float(cagr),
        volatility=vol,
        sharpe=mean_excess / vol if vol > 0 else 0.0,
        sortino=mean_excess / downside if downside > 0 else 0.0,
        max_drawdown=max_dd,
        longest_drawdown_days=longest_drawdown_days(equity),
        calmar=float(cagr / max_dd) if max_dd > 0 else 0.0,
        best_year=float(yearly.max()) if len(yearly) else 0.0,
        worst_year=float(yearly.min()) if len(yearly) else 0.0,
        positive_months=float((monthly > 0).mean()) if len(monthly) else 0.0,
        skew=float(excess.skew()),
        kurtosis=float(excess.kurt() + 3.0),
        psr=probabilistic_sharpe(excess),
    )


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def longest_drawdown_days(equity: pd.Series) -> int:
    """Longest stretch, in calendar days, spent below a previous peak."""
    underwater = equity < equity.cummax()
    longest, start = 0, None
    for day, below in underwater.items():
        if below and start is None:
            start = day
        elif not below and start is not None:
            longest = max(longest, (day - start).days)
            start = None
    if start is not None:
        longest = max(longest, (equity.index[-1] - start).days)
    return longest


def annual_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year returns; partial first and last years are measured from/to their ends."""
    year_end = equity.groupby(equity.index.year).last()
    year_start = pd.concat(
        [pd.Series([equity.iloc[0]], index=[year_end.index[0]]), year_end.iloc[:-1]]
    )
    year_start.index = year_end.index
    return year_end / year_start - 1


def per_period_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std) if std > 0 else 0.0


def probabilistic_sharpe(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """P(true per-period Sharpe > `benchmark_sharpe`) given sample size, skew and kurtosis."""
    returns = returns.dropna()
    n = len(returns)
    if n < 3:
        return float("nan")
    sr = per_period_sharpe(returns)
    skew = float(returns.skew())
    kurt = float(returns.kurt() + 3.0)
    denominator = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denominator <= 0:
        return float("nan")
    return _NORMAL.cdf((sr - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denominator))


def expected_max_sharpe(trial_sharpes: Sequence[float], n_trials: int | None = None) -> float:
    """Sharpe the best of N skill-less trials would show by luck alone (per period).

    The spread comes from the Sharpe ratios observed across `trial_sharpes`; N is
    `n_trials` when more variants were tried than were kept (it must be honest).
    """
    n = max(n_trials or 0, len(trial_sharpes))
    if n < 2 or len(trial_sharpes) < 2:
        return 0.0
    spread = float(np.std(trial_sharpes, ddof=1))
    return spread * (
        (1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n)
        + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n * math.e))
    )


def deflated_sharpe(
    returns: pd.Series, trial_sharpes: Sequence[float], n_trials: int | None = None
) -> float:
    """Probabilistic Sharpe against the luck threshold implied by all trials run.

    `trial_sharpes` are per-period Sharpe ratios of the variants compared
    (including this one). Read the result as the probability the strategy has
    real skill after accounting for the search that found it.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes, n_trials))


def trading_stats(fills: pd.DataFrame, equity: pd.Series) -> dict[str, float]:
    """Turnover, order count and fee drag, annualised over the equity curve's span."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    if fills.empty:
        return {"orders_per_year": 0.0, "turnover": 0.0, "fees": 0.0, "fee_drag": 0.0}
    notional = (fills["quantity"] * fills["price"]).sum()
    average_equity = float(equity.mean())
    fees = float(fills["fees"].sum())
    return {
        "orders_per_year": len(fills) / years,
        "turnover": float(notional / 2 / average_equity / years),
        "fees": fees,
        "fee_drag": fees / average_equity / years,
    }
