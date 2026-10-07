"""Aligned price history: the only view of the market a strategy ever gets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import pandas as pd

BAR_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class PriceHistory:
    """Total-return-adjusted daily bars for a set of symbols.

    One DataFrame per field, indexed by session date with one column per symbol.
    Missing values mean "no data" (e.g. before an ETF launched). Strategies receive
    `history.until(day)`, so anything after `day` is physically absent and cannot
    leak into a decision.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    MAX_GAP_FILL: ClassVar[int] = 3

    @classmethod
    def from_bars(cls, bars: Mapping[str, pd.DataFrame]) -> PriceHistory:
        """Align per-symbol OHLCV frames on the union of their dates.

        Short interior gaps (a missing day or two for one symbol) are filled with the
        previous close, i.e. "no trade that day". Leading and trailing gaps are left
        empty so a symbol whose feed stopped is visibly stale rather than silently
        frozen.
        """
        if not bars:
            raise ValueError("no bars to build a history from")
        index = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in bars.values()))))
        close = pd.DataFrame({s: f["close"] for s, f in bars.items()}).reindex(index)
        filled_close = close.ffill(limit=cls.MAX_GAP_FILL, limit_area="inside")
        gap = close.isna() & filled_close.notna()

        def field(name: str) -> pd.DataFrame:
            frame = pd.DataFrame({s: f[name] for s, f in bars.items()}).reindex(index)
            if name == "volume":
                return frame.mask(gap, 0.0)
            return frame.mask(gap, filled_close)

        return cls(
            open=field("open"),
            high=field("high"),
            low=field("low"),
            close=filled_close,
            volume=field("volume"),
        )

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def last_date(self) -> date:
        return self.dates[-1].date()

    def __len__(self) -> int:
        return len(self.close.index)

    def until(self, when: date | pd.Timestamp) -> PriceHistory:
        """Everything up to and including `when`, nothing after it."""
        stop = self.dates.searchsorted(pd.Timestamp(when), side="right")
        return self.head(int(stop))

    def head(self, n: int) -> PriceHistory:
        return PriceHistory(*(getattr(self, f).iloc[:n] for f in BAR_FIELDS))

    def select(self, symbols: Iterable[str]) -> PriceHistory:
        cols = [s for s in symbols if s in self.close.columns]
        return PriceHistory(*(getattr(self, f)[cols] for f in BAR_FIELDS))

    def returns(self) -> pd.DataFrame:
        """Simple daily close-to-close returns."""
        return self.close.pct_change(fill_method=None)

    def latest_prices(self) -> pd.Series:
        """Most recent close per symbol (NaN if the symbol has no data yet)."""
        return self.close.iloc[-1]

    def available(self, min_bars: int = 1) -> list[str]:
        """Symbols with a current price and at least `min_bars` of history."""
        counts = self.close.notna().sum()
        current = self.close.iloc[-1].notna()
        return [s for s in self.symbols if current[s] and counts[s] >= min_bars]
