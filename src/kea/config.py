"""Typed configuration, loaded from TOML.

Every knob the agent uses lives here. Defaults are chosen *a priori* from the
published literature (12-month momentum, multi-horizon trend, 10% volatility
target) rather than tuned on our own backtests: tuning on the same data you
then report is the most common way trading bots fool their authors.
"""

from __future__ import annotations

import dataclasses
import difflib
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

AssetClass = Literal["equity", "crypto"]
Provider = Literal["auto", "yahoo", "nasdaq", "binance", "tiger"]
Rebalance = Literal["weekly", "monthly"]
FeeModelName = Literal["tiger_nz", "bps", "zero"]
BrokerKind = Literal["paper", "tiger"]
ModelKind = Literal["gbm", "logistic"]


class ConfigError(ValueError):
    """An invalid, missing or misspelt configuration value."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class UniverseConfig:
    """What the agent may hold, what it parks cash in, and what it is judged against."""

    symbols: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "DBC",
    )  # fmt: skip
    cash_symbol: str | None = "BIL"
    benchmark: str = "SPY"
    asset_class: AssetClass = "equity"

    def __post_init__(self) -> None:
        _require(len(self.symbols) > 0, "symbols must not be empty")
        _require(len(set(self.symbols)) == len(self.symbols), "symbols contains duplicates")
        _require(self.cash_symbol not in self.symbols, "cash_symbol must not also be a symbol")

    @property
    def tradable(self) -> tuple[str, ...]:
        """Everything the agent may hold, including the cash proxy."""
        return (*self.symbols, self.cash_symbol) if self.cash_symbol else self.symbols

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """Everything that needs market data: tradables plus the benchmark."""
        return tuple(dict.fromkeys((*self.tradable, self.benchmark)))

    @property
    def periods_per_year(self) -> int:
        return 365 if self.asset_class == "crypto" else 252


@dataclass(frozen=True)
class DataConfig:
    provider: Provider = "auto"
    start: date = date(2005, 1, 1)
    cache_dir: Path = Path(".kea/cache")
    max_cache_age_hours: float = 12.0
    # Accept price-only bars when a source cannot supply dividends. Off by default:
    # ignoring dividends understates bond and equity returns by their yield and
    # skews every signal against income-paying assets.
    allow_price_only: bool = False

    def __post_init__(self) -> None:
        _require(self.max_cache_age_hours >= 0, "max_cache_age_hours must be >= 0")


@dataclass(frozen=True)
class TrendConfig:
    """Multi-horizon time-series momentum (Moskowitz, Ooi & Pedersen 2012)."""

    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    vol_lookback: int = 63

    def __post_init__(self) -> None:
        _require(all(n > 0 for n in self.lookbacks), "lookbacks must be positive")
        _require(self.vol_lookback > 1, "vol_lookback must be > 1")


@dataclass(frozen=True)
class MomentumConfig:
    """Cross-sectional 12-1 momentum with an absolute-momentum filter (Antonacci)."""

    lookback: int = 252
    skip: int = 21
    top_k: int = 3

    def __post_init__(self) -> None:
        _require(self.lookback > self.skip >= 0, "need lookback > skip >= 0")
        _require(self.top_k >= 1, "top_k must be >= 1")


@dataclass(frozen=True)
class MLConfig:
    """Walk-forward classifier predicting whether each asset beats cash over `horizon`."""

    model: ModelKind = "gbm"
    horizon: int = 21
    retrain_every: int = 63
    min_train_days: int = 756
    sample_every: int = 5
    conviction_scale: float = 0.10
    vol_lookback: int = 63
    random_state: int = 7

    def __post_init__(self) -> None:
        _require(self.horizon >= 1, "horizon must be >= 1")
        _require(self.retrain_every >= 1, "retrain_every must be >= 1")
        _require(self.min_train_days > self.horizon, "min_train_days must exceed horizon")
        _require(self.sample_every >= 1, "sample_every must be >= 1")
        _require(0 < self.conviction_scale <= 0.5, "conviction_scale must be in (0, 0.5]")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "ensemble"
    members: tuple[str, ...] = ("trend", "momentum", "ml")
    rebalance: Rebalance = "monthly"
    trend: TrendConfig = field(default_factory=TrendConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    ml: MLConfig = field(default_factory=MLConfig)


@dataclass(frozen=True)
class RiskConfig:
    """Portfolio construction limits plus the agent's circuit breakers."""

    target_vol: float = 0.10
    max_gross: float = 1.0
    max_weight: float = 0.35
    cov_lookback: int = 126
    shrinkage: float = 0.2
    halt_drawdown: float = 0.25
    halt_daily_loss: float = 0.06
    max_price_jump: float = 0.35

    def __post_init__(self) -> None:
        _require(0 < self.target_vol <= 1, "target_vol must be in (0, 1]")
        _require(0 < self.max_gross <= 1, "max_gross must be in (0, 1]: Kea never uses leverage")
        _require(0 < self.max_weight <= 1, "max_weight must be in (0, 1]")
        _require(self.cov_lookback > 20, "cov_lookback must be > 20")
        _require(0 <= self.shrinkage <= 1, "shrinkage must be in [0, 1]")
        _require(0 < self.halt_drawdown < 1, "halt_drawdown must be in (0, 1)")
        _require(0 < self.halt_daily_loss < 1, "halt_daily_loss must be in (0, 1)")
        _require(self.max_price_jump > 0, "max_price_jump must be > 0")


@dataclass(frozen=True)
class ExecutionConfig:
    """How target weights become orders, and what those orders cost."""

    fees: FeeModelName = "tiger_nz"
    fee_bps: float = 0.0
    free_orders_per_month: int = 0
    slippage_bps: float = 5.0
    whole_shares: bool = True
    drift_band: float = 0.02
    min_trade_value: float = 250.0
    cash_buffer: float = 0.01

    def __post_init__(self) -> None:
        _require(self.fee_bps >= 0, "fee_bps must be >= 0")
        _require(self.free_orders_per_month >= 0, "free_orders_per_month must be >= 0")
        _require(self.slippage_bps >= 0, "slippage_bps must be >= 0")
        _require(0 <= self.drift_band < 1, "drift_band must be in [0, 1)")
        _require(self.min_trade_value >= 0, "min_trade_value must be >= 0")
        _require(0 <= self.cash_buffer < 0.5, "cash_buffer must be in [0, 0.5)")


@dataclass(frozen=True)
class BrokerConfig:
    kind: BrokerKind = "paper"
    initial_cash: float = 10_000.0
    state_dir: Path = Path(".kea/state")
    allow_live: bool = False

    def __post_init__(self) -> None:
        _require(self.initial_cash > 0, "initial_cash must be > 0")


@dataclass(frozen=True)
class BacktestConfig:
    start: date | None = None
    end: date | None = None


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def replace(self, **sections: Any) -> Config:
        return dataclasses.replace(self, **sections)


def load_config(path: Path | str | None = None, overrides: Sequence[str] = ()) -> Config:
    """Load a TOML file (or the defaults) plus `section.key=value` overrides.

    Override values are parsed as TOML, falling back to plain strings, so both
    `risk.target_vol=0.08` and `broker.state_dir=ledger/state` work.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override '{item}' must look like section.key=value")
        *parents, leaf = key.strip().split(".")
        table = raw
        for part in parents:
            table = table.setdefault(part, {})
            if not isinstance(table, dict):
                raise ConfigError(f"override '{item}': '{part}' is not a table")
        table[leaf] = _parse_value(text.strip())
    return config_from_dict(raw)


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def config_from_dict(raw: Mapping[str, Any]) -> Config:
    return _build(Config, raw, "")


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{path.rstrip('.')}' must be a table")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    for key in data:
        if key not in names:
            guess = difflib.get_close_matches(key, names, n=1)
            hint = f" (did you mean '{guess[0]}'?)" if guess else ""
            raise ConfigError(f"unknown setting '{path}{key}'{hint}")
    kwargs = {key: _coerce(hints[key], value, f"{path}{key}") for key, value in data.items()}
    try:
        return cls(**kwargs)
    except ValueError as exc:
        raise ConfigError(f"[{path.rstrip('.') or 'config'}] {exc}") from exc


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, f"{path}.")
    if origin in (types.UnionType, typing.Union):
        is_none = value is None or (isinstance(value, str) and value.lower() == "none")
        if is_none and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value, path)
    if origin is Literal:
        if value not in args:
            choices = ", ".join(map(repr, args))
            raise ConfigError(f"'{path}' must be one of {choices}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"'{path}' must be a list")
        return tuple(_coerce(args[0], item, path) for item in value)
    if tp is Path:
        return Path(value)
    if tp is date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            raise ConfigError(f"'{path}' must be a date like 2015-01-31") from None
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{path}' must be true or false")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"'{path}' must be a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{path}' must be an integer")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{path}' must be a string")
        return value
    raise ConfigError(f"'{path}': unsupported setting type {tp!r}")  # pragma: no cover
