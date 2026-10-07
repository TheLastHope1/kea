"""Rebalance calendar.

A decision is due on the first completed session of each new week or month. The
rule only looks backwards (has the period changed since the last decision?), so
the backtest and the live agent trigger on exactly the same days without needing
an exchange holiday calendar.
"""

from __future__ import annotations

from datetime import date

from kea.config import Rebalance


def period_key(day: date, rebalance: Rebalance) -> tuple[int, int]:
    if rebalance == "weekly":
        iso = day.isocalendar()
        return (iso.year, iso.week)
    return (day.year, day.month)


def is_rebalance_day(day: date, last_rebalance: date | None, rebalance: Rebalance) -> bool:
    return last_rebalance is None or period_key(day, rebalance) != period_key(
        last_rebalance, rebalance
    )
