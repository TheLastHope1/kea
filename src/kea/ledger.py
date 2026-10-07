"""The agent's memory: run state, an append-only journal, and the equity track record.

Everything lives in one directory of small text files (JSON, JSONL, CSV) so it
can be committed to git, diffed, and audited by a human.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

EQUITY_COLUMNS = ["date", "equity", "cash", "exposure", "benchmark", "recorded_at"]


@dataclass
class AgentState:
    last_processed: str | None = None
    last_rebalance: str | None = None
    targets: dict[str, float] | None = None
    retries_left: int = 0
    halted: dict[str, str] | None = None
    seen_fills: list[str] = field(default_factory=list)

    @property
    def last_processed_date(self) -> date | None:
        return date.fromisoformat(self.last_processed) if self.last_processed else None

    @property
    def last_rebalance_date(self) -> date | None:
        return date.fromisoformat(self.last_rebalance) if self.last_rebalance else None


class Ledger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state_path = directory / "agent_state.json"
        self.journal_path = directory / "journal.jsonl"
        self.equity_path = directory / "equity.csv"
        self.lock_path = directory / ".lock"

    # ----------------------------------------------------------------- state

    def load_state(self) -> AgentState:
        if not self.state_path.exists():
            return AgentState()
        return AgentState(**json.loads(self.state_path.read_text()))

    def save_state(self, state: AgentState) -> None:
        self._atomic_write(self.state_path, json.dumps(asdict(state), indent=2, sort_keys=True))

    # --------------------------------------------------------------- journal

    def record(self, event: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
        with self.journal_path.open("a") as fh:
            fh.write(json.dumps(entry, default=_json_default, sort_keys=True) + "\n")

    def journal(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        lines = self.journal_path.read_text().splitlines()
        if limit is not None:
            lines = lines[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    # ---------------------------------------------------------------- equity

    def equity(self) -> pd.DataFrame:
        if not self.equity_path.exists():
            return pd.DataFrame(columns=EQUITY_COLUMNS).set_index("date")
        return pd.read_csv(self.equity_path, index_col="date", parse_dates=["date"])

    def record_equity(
        self,
        day: date,
        equity: float,
        cash: float,
        exposure: float,
        benchmark: float | None,
        persist: bool = True,
    ) -> pd.DataFrame:
        """Upsert the row for `day` and return the whole track record."""
        frame = self.equity()
        frame.loc[pd.Timestamp(day)] = {
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "exposure": round(exposure, 4),
            "benchmark": benchmark,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        frame = frame.sort_index()
        frame.index.name = "date"
        if not persist:
            return frame
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.equity_path.with_suffix(".tmp")
        frame.to_csv(tmp, date_format="%Y-%m-%d")
        tmp.replace(self.equity_path)
        return frame

    # ------------------------------------------------------------------ lock

    @contextmanager
    def lock(self, stale_after: float = 3600.0) -> Iterator[None]:
        """Stop two agent runs from interleaving (e.g. a cron job and a manual run)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists() and time.time() - self.lock_path.stat().st_mtime > stale_after:
            self.lock_path.unlink()
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"another Kea run holds {self.lock_path}") from None
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text + "\n")
        tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, date | datetime | pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)
