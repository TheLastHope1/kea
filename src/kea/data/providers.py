"""Market data providers.

Each provider returns total-return-adjusted daily OHLCV bars for one symbol as a
DataFrame indexed by session date. Free sources are flaky (Yahoo rate-limits
shared cloud IPs; Nasdaq only serves ten years), so `load_history` chains them
and caches whatever succeeds.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kea.data.history import BAR_FIELDS

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


class DataUnavailable(RuntimeError):
    """A provider could not supply data for a symbol (rate limit, unknown symbol, outage)."""


class PriceProvider(Protocol):
    name: str

    def fetch(self, symbol: str, start: date) -> pd.DataFrame: ...


def http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BROWSER_UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _get_json(session: requests.Session, url: str, params: dict[str, Any], source: str) -> Any:
    try:
        response = session.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise DataUnavailable(f"{source}: {exc}") from exc
    if response.status_code != 200:
        raise DataUnavailable(f"{source}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise DataUnavailable(f"{source}: response was not JSON") from exc


def finalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the bar schema: sorted unique dates, float OHLCV, no missing closes."""
    frame = frame.loc[:, list(BAR_FIELDS)].astype(float)
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.index.name = "date"
    if frame.empty:
        raise DataUnavailable("provider returned no usable rows")
    return frame


# --------------------------------------------------------------------------- Yahoo


class YahooProvider:
    """Yahoo Finance chart API: long history with split and dividend adjustment."""

    name = "yahoo"
    URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or http_session()

    def fetch(self, symbol: str, start: date) -> pd.DataFrame:
        params = {
            "period1": int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp()),
            "period2": int(_time.time()) + 86_400,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        payload = _get_json(self.session, self.URL.format(symbol=symbol), params, f"yahoo {symbol}")
        chart = payload.get("chart") or {}
        if chart.get("error") or not chart.get("result"):
            raise DataUnavailable(f"yahoo {symbol}: {chart.get('error') or 'empty result'}")
        return parse_yahoo_chart(chart["result"][0])


def parse_yahoo_chart(result: dict[str, Any]) -> pd.DataFrame:
    stamps = result.get("timestamp") or []
    if not stamps:
        raise DataUnavailable("yahoo: no rows")
    tz = result.get("meta", {}).get("exchangeTimezoneName") or "America/New_York"
    dates = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({f: quote.get(f) for f in BAR_FIELDS}, index=dates, dtype=float)
    adjclose = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adjclose is not None:
        adjusted_close = pd.Series(adjclose, index=dates, dtype=float)
        factor = adjusted_close / frame["close"]
        frame[["open", "high", "low"]] = frame[["open", "high", "low"]].mul(factor, axis=0)
        frame["close"] = adjusted_close
    return finalize_bars(frame)


# --------------------------------------------------------------------------- Nasdaq


class NasdaqProvider:
    """Nasdaq.com quote API: ten years of raw bars plus dividend history.

    Prices arrive unadjusted for dividends, so we back-adjust them ourselves;
    skipping this would understate a bond ETF's return by its whole yield.
    Nasdaq only publishes dividends for Nasdaq-listed securities; for anything
    else (SPY trades on NYSE Arca) the provider refuses unless price-only data
    has been explicitly allowed.
    """

    name = "nasdaq"
    HISTORICAL = "https://api.nasdaq.com/api/quote/{symbol}/historical"
    DIVIDENDS = "https://api.nasdaq.com/api/quote/{symbol}/dividends"
    ASSET_CLASSES = ("etf", "stocks")

    def __init__(
        self, session: requests.Session | None = None, allow_price_only: bool = False
    ) -> None:
        self.session = session or http_session()
        self.allow_price_only = allow_price_only
        self.price_only: set[str] = set()

    def fetch(self, symbol: str, start: date) -> pd.DataFrame:
        for asset_class in self.ASSET_CLASSES:
            params = {
                "assetclass": asset_class,
                "fromdate": start.isoformat(),
                "todate": date.today().isoformat(),
                "limit": 9999,
            }
            payload = _get_json(
                self.session, self.HISTORICAL.format(symbol=symbol), params, f"nasdaq {symbol}"
            )
            rows = (((payload or {}).get("data") or {}).get("tradesTable") or {}).get("rows")
            if rows:
                break
        else:
            raise DataUnavailable(f"nasdaq {symbol}: no price history")
        bars = parse_nasdaq_rows(rows)
        dividends = self._dividends(symbol, asset_class)
        if dividends is None:
            if not self.allow_price_only:
                raise DataUnavailable(
                    f"nasdaq {symbol}: no dividend history for non-Nasdaq listings "
                    "(set data.allow_price_only = true to accept price-only bars)"
                )
            self.price_only.add(symbol)
            return finalize_bars(bars)
        return finalize_bars(adjust_for_dividends(bars, dividends))

    def _dividends(self, symbol: str, asset_class: str) -> pd.Series | None:
        """Cash dividends by ex-date, or None if Nasdaq does not publish them for `symbol`."""
        payload = _get_json(
            self.session,
            self.DIVIDENDS.format(symbol=symbol),
            {"assetclass": asset_class},
            f"nasdaq {symbol} dividends",
        )
        if "not available" in str((payload or {}).get("message") or "").lower():
            return None
        rows = ((((payload or {}).get("data") or {}).get("dividends")) or {}).get("rows") or []
        return parse_nasdaq_dividends(rows)


def _number(text: Any) -> float:
    if text is None:
        return float("nan")
    cleaned = str(text).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def parse_nasdaq_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    index = pd.to_datetime([r["date"] for r in rows], format="%m/%d/%Y")
    return pd.DataFrame(
        {f: [_number(r.get(f)) for r in rows] for f in BAR_FIELDS},
        index=index,
    ).sort_index()


def parse_nasdaq_dividends(rows: list[dict[str, Any]]) -> pd.Series:
    cash = [r for r in rows if str(r.get("type", "Cash")).lower() == "cash"]
    amounts = pd.Series(
        [_number(r.get("amount")) for r in cash],
        index=pd.to_datetime(
            [r.get("exOrEffDate") for r in cash], format="%m/%d/%Y", errors="coerce"
        ),
        dtype=float,
    )
    amounts = amounts[amounts.index.notna() & (amounts > 0)]
    return amounts.groupby(level=0).sum().sort_index()


def adjust_for_dividends(bars: pd.DataFrame, dividends: pd.Series) -> pd.DataFrame:
    """Back-adjust prices so close-to-close changes are total returns.

    For each ex-date with cash amount D, every price before the ex-date is scaled by
    1 - D / C, where C is the last close before the ex-date (the CRSP convention).
    """
    close = bars["close"]
    factor = pd.Series(1.0, index=bars.index)
    for ex_date, amount in dividends.items():
        before = close.index < ex_date
        prior_close = close[before].dropna()
        if prior_close.empty:
            continue
        ratio = 1.0 - amount / prior_close.iloc[-1]
        if 0.0 < ratio < 1.0:
            factor[before] *= ratio
    adjusted = bars.copy()
    prices = ["open", "high", "low", "close"]
    adjusted[prices] = bars[prices].mul(factor, axis=0)
    return adjusted


# --------------------------------------------------------------------------- Binance


class BinanceProvider:
    """Binance public market-data mirror (no account or API key needed)."""

    name = "binance"
    URL = "https://data-api.binance.vision/api/v3/klines"
    PAGE = 1000

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or http_session()

    def fetch(self, symbol: str, start: date) -> pd.DataFrame:
        cursor = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
        rows: list[list[Any]] = []
        while True:
            params = {"symbol": symbol, "interval": "1d", "startTime": cursor, "limit": self.PAGE}
            batch = _get_json(self.session, self.URL, params, f"binance {symbol}")
            if not isinstance(batch, list):
                raise DataUnavailable(f"binance {symbol}: unexpected payload")
            rows.extend(batch)
            if len(batch) < self.PAGE:
                break
            cursor = int(batch[-1][0]) + 1
        if not rows:
            raise DataUnavailable(f"binance {symbol}: no rows")
        index = pd.to_datetime([int(r[0]) for r in rows], unit="ms").normalize()
        frame = pd.DataFrame(
            {f: [float(r[i]) for r in rows] for i, f in enumerate(BAR_FIELDS, start=1)},
            index=index,
        )
        return finalize_bars(frame)


# --------------------------------------------------------------------------- Tiger


class TigerProvider:
    """Tiger Brokers OpenAPI bars, adjusted with Tiger's `br` (forward) mode.

    Needs API credentials (see `kea.tiger`); the quote client is created lazily so
    importing this module never requires the optional SDK.
    """

    name = "tiger"

    def __init__(self, client_factory: Callable[[], Any] | None = None) -> None:
        self._factory = client_factory
        self._client: Any = None

    def _quote_client(self) -> Any:
        if self._client is None:
            if self._factory is None:
                from kea.tiger import quote_client

                self._factory = quote_client
            self._client = self._factory()
        return self._client

    def fetch(self, symbol: str, start: date) -> pd.DataFrame:
        try:
            frame = self._quote_client().get_bars_by_page(
                symbol,
                period="day",
                begin_time=start.isoformat(),
                end_time=-1,
                total=20_000,
                page_size=1_000,
                right="br",
            )
        except Exception as exc:  # the SDK raises a zoo of exception types
            raise DataUnavailable(f"tiger {symbol}: {exc}") from exc
        if frame is None or frame.empty:
            raise DataUnavailable(f"tiger {symbol}: no rows")
        stamps = pd.to_datetime(frame["time"], unit="ms", utc=True)
        index = stamps.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        bars = frame[list(BAR_FIELDS)].set_axis(pd.DatetimeIndex(index), axis=0)
        return finalize_bars(bars)


PROVIDERS: dict[str, Callable[[], PriceProvider]] = {
    "yahoo": YahooProvider,
    "nasdaq": NasdaqProvider,
    "binance": BinanceProvider,
    "tiger": TigerProvider,
}


def provider_chain(
    name: str, asset_class: str, allow_price_only: bool = False
) -> list[PriceProvider]:
    """Providers to try, in order, for a configured provider name."""
    if name == "auto":
        names = ["binance"] if asset_class == "crypto" else ["yahoo", "nasdaq"]
    else:
        names = [name]
    return [
        NasdaqProvider(allow_price_only=allow_price_only) if n == "nasdaq" else PROVIDERS[n]()
        for n in names
    ]
