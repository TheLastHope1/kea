"""When is a daily bar final? Decisions must only ever use completed sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class SessionClock:
    """Knows when a market's daily bar is complete.

    `close` is the session close in the exchange's local time; a bar is treated as
    final `settle` after that, giving data vendors time to publish the official
    close. Holidays need no calendar: on a holiday there is simply no bar.
    """

    tz: ZoneInfo
    close: time
    settle: timedelta
    weekdays_only: bool

    def last_complete_session(self, now: datetime) -> date:
        local = now.astimezone(self.tz)
        bar_final_at = datetime.combine(local.date(), self.close, self.tz) + self.settle
        day = local.date() if local >= bar_final_at else local.date() - timedelta(days=1)
        if self.weekdays_only:
            while day.weekday() >= 5:
                day -= timedelta(days=1)
        return day

    def drop_incomplete(self, frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """Remove any still-forming bar (e.g. today's, while the market is open)."""
        cutoff = pd.Timestamp(self.last_complete_session(now))
        return frame.loc[frame.index <= cutoff]


US_EQUITIES = SessionClock(
    tz=ZoneInfo("America/New_York"),
    close=time(16, 0),
    settle=timedelta(minutes=30),
    weekdays_only=True,
)

# Crypto trades around the clock; exchanges cut daily bars at 00:00 UTC.
CRYPTO = SessionClock(
    tz=ZoneInfo("UTC"),
    close=time(23, 59, 59),
    settle=timedelta(minutes=5),
    weekdays_only=False,
)


def clock_for(asset_class: str) -> SessionClock:
    return CRYPTO if asset_class == "crypto" else US_EQUITIES
