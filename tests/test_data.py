from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from kea.config import DataConfig
from kea.data import CRYPTO, US_EQUITIES, DataUnavailable, MarketData, PriceHistory
from kea.data.providers import (
    NasdaqProvider,
    adjust_for_dividends,
    parse_nasdaq_dividends,
    parse_nasdaq_rows,
    parse_yahoo_chart,
)

NY = ZoneInfo("America/New_York")


def bars(closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=dates,
    )


# --------------------------------------------------------------- history


def test_history_aligns_and_only_fills_interior_gaps():
    a = bars([1, 2, 3, 4, 5, 6])
    b = bars([10, 11, 12, 13, 14, 15]).drop(index=a.index[2]).iloc[:-1]
    history = PriceHistory.from_bars({"A": a, "B": b})
    assert history.close["B"].iloc[2] == 11  # interior gap: previous close, "no trade"
    assert history.volume["B"].iloc[2] == 0
    assert np.isnan(history.close["B"].iloc[-1])  # trailing gap stays visibly stale
    assert history.available() == ["A"]


def test_until_excludes_the_future():
    history = PriceHistory.from_bars({"A": bars(range(1, 11))})
    cut = history.dates[4]
    view = history.until(cut)
    assert len(view) == 5
    assert view.dates[-1] == cut
    assert view.close["A"].iloc[-1] == 5


# ------------------------------------------------------------- providers


def test_yahoo_adjusted_close_scales_the_whole_bar():
    stamps = [1704205800, 1704292200]  # 2024-01-02 and 2024-01-03, 09:30 New York
    chart = {
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": stamps,
        "indicators": {
            "quote": [
                {
                    "open": [10, 20],
                    "high": [11, 21],
                    "low": [9, 19],
                    "close": [10, 20],
                    "volume": [1, 2],
                }
            ],
            "adjclose": [{"adjclose": [5, 20]}],
        },
    }
    frame = parse_yahoo_chart(chart)
    assert list(frame.index.date.astype(str)) == ["2024-01-02", "2024-01-03"]
    assert frame.iloc[0][["open", "high", "low", "close"]].tolist() == [5, 5.5, 4.5, 5]
    assert frame.iloc[1]["close"] == 20


def test_nasdaq_rows_parse_currency_and_thousands():
    rows = [
        {
            "date": "01/03/2024",
            "close": "$101.50",
            "open": "100",
            "high": "102",
            "low": "99",
            "volume": "1,234",
        },
        {"date": "01/02/2024", "close": "N/A", "open": "1", "high": "1", "low": "1", "volume": "1"},
    ]
    frame = parse_nasdaq_rows(rows)
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    assert np.isnan(frame["close"].iloc[0])
    assert frame["close"].iloc[1] == 101.5
    assert frame["volume"].iloc[1] == 1234


def test_dividend_adjustment_turns_price_returns_into_total_returns():
    raw = bars([100.0, 100.0, 99.0, 99.0])  # $1 dividend goes ex on day 3
    ex_date = raw.index[2]
    adjusted = adjust_for_dividends(raw, pd.Series({ex_date: 1.0}))
    total_return = adjusted["close"].iloc[-1] / adjusted["close"].iloc[0] - 1
    assert total_return == pytest.approx(0.0)  # price fell by exactly the dividend paid
    assert adjusted["close"].iloc[2:].tolist() == [99.0, 99.0]  # after ex-date: unchanged


def test_dividend_parser_keeps_only_cash_distributions():
    rows = [
        {"exOrEffDate": "03/01/2024", "type": "Cash", "amount": "$0.50"},
        {"exOrEffDate": "03/01/2024", "type": "Cash", "amount": "$0.10"},
        {"exOrEffDate": "04/01/2024", "type": "Stock", "amount": "$9.99"},
        {"exOrEffDate": "N/A", "type": "Cash", "amount": "$1"},
    ]
    dividends = parse_nasdaq_dividends(rows)
    assert dividends.to_dict() == {pd.Timestamp("2024-03-01"): pytest.approx(0.6)}


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, dividends_payload):
        self.dividends_payload = dividends_payload

    def get(self, url, params=None, timeout=None):
        if url.endswith("/historical"):
            rows = [
                {
                    "date": "01/02/2024",
                    "close": "10",
                    "open": "10",
                    "high": "10",
                    "low": "10",
                    "volume": "1",
                }
            ]
            return FakeResponse({"data": {"tradesTable": {"rows": rows}}})
        return FakeResponse(self.dividends_payload)


NOT_AVAILABLE = {
    "data": {"dividends": {"rows": None}},
    "message": "Dividend History for Non-Nasdaq symbols is not available",
}


def test_nasdaq_refuses_price_only_data_by_default():
    provider = NasdaqProvider(session=FakeSession(NOT_AVAILABLE))
    with pytest.raises(DataUnavailable, match="allow_price_only"):
        provider.fetch("SPY", pd.Timestamp("2024-01-01").date())


def test_nasdaq_price_only_is_opt_in_and_flagged():
    provider = NasdaqProvider(session=FakeSession(NOT_AVAILABLE), allow_price_only=True)
    frame = provider.fetch("SPY", pd.Timestamp("2024-01-01").date())
    assert len(frame) == 1
    assert provider.price_only == {"SPY"}


# ---------------------------------------------------------------- clocks


def test_equity_session_is_final_half_an_hour_after_the_close():
    tuesday = datetime(2026, 10, 6, 16, 20, tzinfo=NY)
    assert str(US_EQUITIES.last_complete_session(tuesday)) == "2026-10-05"
    assert (
        str(US_EQUITIES.last_complete_session(tuesday.replace(hour=16, minute=31))) == "2026-10-06"
    )


def test_weekends_roll_back_to_friday():
    sunday = datetime(2026, 10, 4, 12, tzinfo=NY)
    assert str(US_EQUITIES.last_complete_session(sunday)) == "2026-10-02"


def test_crypto_bar_is_final_after_utc_midnight():
    assert str(CRYPTO.last_complete_session(datetime(2026, 10, 7, 10, tzinfo=UTC))) == "2026-10-06"
    assert (
        str(CRYPTO.last_complete_session(datetime(2026, 10, 8, 0, 6, tzinfo=UTC))) == "2026-10-07"
    )


# ----------------------------------------------------------------- store


class StubProvider:
    def __init__(self, name, frame=None, error=None):
        self.name = name
        self.frame = frame
        self.error = error
        self.calls = 0

    def fetch(self, symbol, start):
        self.calls += 1
        if self.error:
            raise DataUnavailable(self.error)
        return self.frame


def test_store_falls_back_caches_and_drops_the_forming_bar(tmp_path):
    frame = bars([1, 2, 3, 4], start="2026-10-01")  # Thu 1 .. Tue 6 October
    broken = StubProvider("broken", error="HTTP 429")
    working = StubProvider("working", frame)
    market = MarketData(DataConfig(cache_dir=tmp_path), providers=[broken, working])
    now = datetime(2026, 10, 6, 12, tzinfo=NY)  # Tuesday, market still open

    first = market.bars("SPY", now)
    assert first.index[-1] == pd.Timestamp("2026-10-05")  # today's bar is still forming
    assert market.sources["SPY"] == "working"

    again = market.bars("SPY", now)
    assert working.calls == 1  # served from the fresh cache
    assert again.equals(first) or (again.values == first.values).all()
    assert "cache" in market.sources["SPY"]


def test_store_serves_stale_cache_when_every_provider_fails(tmp_path):
    frame = bars([1, 2, 3], start="2026-09-01")
    MarketData(DataConfig(cache_dir=tmp_path), providers=[StubProvider("p", frame)]).bars(
        "SPY", datetime(2026, 9, 4, 18, tzinfo=NY)
    )
    market = MarketData(DataConfig(cache_dir=tmp_path), providers=[StubProvider("p", error="down")])
    stale = market.bars("SPY", datetime(2026, 10, 6, 18, tzinfo=NY))
    assert len(stale) == 3
    assert "stale" in market.sources["SPY"]


def test_store_raises_when_nothing_is_available(tmp_path):
    market = MarketData(DataConfig(cache_dir=tmp_path), providers=[StubProvider("p", error="down")])
    with pytest.raises(DataUnavailable, match="down"):
        market.bars("SPY")


FRENCH_SAMPLE = """This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_french_daily_file_becomes_total_return_indices():
    from kea.data.providers import FrenchProvider, parse_french_daily

    returns = parse_french_daily(FRENCH_SAMPLE)
    assert list(returns.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert returns.loc["1926-07-02", "Mkt-RF"] == pytest.approx(0.45)

    provider = FrenchProvider(session=object())
    provider._returns = returns
    market = provider.fetch("MARKET", pd.Timestamp("1926-01-01").date())
    assert market["close"].tolist() == pytest.approx([100 * 1.001, 100 * 1.001 * 1.0046])
    assert (market["open"] == market["close"]).all()
    with pytest.raises(DataUnavailable, match="unknown symbol"):
        provider.fetch("SPY", pd.Timestamp("1926-01-01").date())
