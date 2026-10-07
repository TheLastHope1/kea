"""Loading price history through a provider chain, with an on-disk cache."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kea.config import DataConfig
from kea.data.clock import SessionClock, clock_for
from kea.data.history import PriceHistory
from kea.data.providers import DataUnavailable, PriceProvider, provider_chain

log = logging.getLogger(__name__)

PRICE_ONLY = "price-only"


@dataclass
class MarketData:
    """Fetches adjusted daily bars, falling back across providers and caching results.

    The cache only ever holds completed sessions, and an entry counts as fresh only
    if it is recent *and* already contains the latest completed session, so a bar
    cached mid-session can never masquerade as a final close. `sources` records
    where each symbol's bars came from, including any price-only fallbacks.
    """

    config: DataConfig
    asset_class: str = "equity"
    providers: Sequence[PriceProvider] | None = None
    clock: SessionClock = field(init=False)
    sources: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.clock = clock_for(self.asset_class)
        if self.providers is None:
            self.providers = provider_chain(
                self.config.provider, self.asset_class, self.config.allow_price_only
            )

    @property
    def price_only(self) -> list[str]:
        return [s for s, source in self.sources.items() if PRICE_ONLY in source]

    def history(self, symbols: Iterable[str], now: datetime | None = None) -> PriceHistory:
        now = now or datetime.now(UTC)
        return PriceHistory.from_bars({symbol: self.bars(symbol, now) for symbol in symbols})

    def bars(self, symbol: str, now: datetime | None = None) -> pd.DataFrame:
        now = now or datetime.now(UTC)
        cached = self._cached(symbol, now, fresh_only=True)
        if cached is not None:
            return cached
        errors = []
        for provider in self.providers or ():
            try:
                frame = self.clock.drop_incomplete(provider.fetch(symbol, self.config.start), now)
            except DataUnavailable as exc:
                errors.append(str(exc))
                log.info("%s", exc)
                continue
            label = provider.name
            if symbol in getattr(provider, "price_only", ()):
                label = f"{provider.name}-{PRICE_ONLY}"
                log.warning(
                    "%s: using price-only bars from %s (dividends ignored)", symbol, provider.name
                )
            self._write(label, symbol, frame)
            self.sources[symbol] = label
            return frame
        stale = self._cached(symbol, now, fresh_only=False)
        if stale is not None:
            log.warning("using stale cached data for %s (%s)", symbol, "; ".join(errors))
            return stale
        raise DataUnavailable(f"no data for {symbol}: {'; '.join(errors) or 'no providers'}")

    def _labels(self) -> list[str]:
        names = [p.name for p in self.providers or ()]
        if self.config.allow_price_only:
            names += [f"{n}-{PRICE_ONLY}" for n in names]
        return names

    def _path(self, label: str, symbol: str) -> Path:
        return self.config.cache_dir / label / f"{symbol}.csv"

    def _write(self, label: str, symbol: str, frame: pd.DataFrame) -> None:
        path = self._path(label, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.rename_axis("date").to_csv(tmp)
        tmp.replace(path)

    def _cached(self, symbol: str, now: datetime, fresh_only: bool) -> pd.DataFrame | None:
        if fresh_only and self.config.max_cache_age_hours <= 0:
            return None
        latest_session = pd.Timestamp(self.clock.last_complete_session(now))
        for label in self._labels():
            path = self._path(label, symbol)
            if not path.exists():
                continue
            frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
            if fresh_only:
                modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
                if (now - modified).total_seconds() / 3600 > self.config.max_cache_age_hours:
                    continue
                if frame.empty or frame.index[-1] < latest_session:
                    continue
            self.sources[symbol] = f"{label} (cache)" if fresh_only else f"{label} (stale cache)"
            return frame
        return None
