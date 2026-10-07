"""Number formatting shared by the CLI and the reports."""

from __future__ import annotations

import math


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if _missing(value) else f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    return "n/a" if _missing(value) else f"{value:,.{digits}f}"


def money(value: float | None, digits: int = 2) -> str:
    if _missing(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{digits}f}"


def count(value: float | None) -> str:
    return "n/a" if _missing(value) else f"{value:,.0f}"
