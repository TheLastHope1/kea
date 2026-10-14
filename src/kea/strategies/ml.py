"""A walk-forward machine-learning allocator, held to the same standard as the rules.

The model predicts, for each asset, the probability that it beats cash over the
next `horizon` sessions. Three things keep the evaluation honest:

* **Walk-forward training.** The model is refit once per calendar quarter using
  only data up to the end of the previous quarter, then used unchanged until
  the next refit. The live agent reproduces exactly the same schedule.
* **Purged labels.** A training example at day t needs the price at t+horizon,
  so examples whose label would peek past the training cutoff are dropped.
* **Every prediction is logged**, so the report can score the model's
  out-of-sample skill (AUC, Brier score) instead of just admiring its returns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from kea.config import MLConfig
from kea.data.history import PriceHistory
from kea.strategies.base import Strategy, eligible, inverse_vol_weights, trailing_vol

FEATURE_LOOKBACK = 252
MIN_TRAINING_ROWS = 250


def build_features(
    close: pd.DataFrame, symbols: Sequence[str], cash_symbol: str | None
) -> pd.DataFrame:
    """Feature panel indexed by (date, symbol). Each row only uses prices up to its date."""
    risky = close[list(symbols)]
    returns = risky.pct_change(fill_method=None)
    vol_21 = returns.rolling(21, min_periods=17).std()
    vol_63 = returns.rolling(63, min_periods=50).std()
    frames: dict[str, pd.DataFrame] = {
        f"ret_{n}": risky / risky.shift(n) - 1 for n in (5, 21, 63, 126, 252)
    }
    frames["vol_63"] = vol_63 * math.sqrt(252)
    frames["vol_ratio"] = vol_21 / vol_63
    frames["sharpe_63"] = frames["ret_63"] / (vol_63 * math.sqrt(63))
    frames["sma_50_gap"] = risky / risky.rolling(50).mean() - 1
    frames["sma_200_gap"] = risky / risky.rolling(200).mean() - 1
    frames["drawdown_252"] = risky / risky.rolling(252, min_periods=126).max() - 1
    frames["rank_63"] = frames["ret_63"].rank(axis=1, pct=True)
    frames["rank_252"] = frames["ret_252"].rank(axis=1, pct=True)
    for n in (63, 252):
        cash = (close[cash_symbol] / close[cash_symbol].shift(n) - 1) if cash_symbol else 0.0
        frames[f"excess_{n}"] = frames[f"ret_{n}"].sub(cash, axis=0)
    index = pd.MultiIndex.from_product([risky.index, risky.columns], names=["date", "symbol"])
    return pd.DataFrame({k: v.to_numpy().reshape(-1) for k, v in frames.items()}, index=index)


def forward_excess_returns(
    close: pd.DataFrame, symbols: Sequence[str], cash_symbol: str | None, horizon: int
) -> pd.Series:
    """Excess return over cash from each date to `horizon` sessions later.

    Indexed by (date, symbol). The last `horizon` dates have no label (NaN): their
    outcome is not yet known.
    """
    risky = close[list(symbols)]
    forward = risky.shift(-horizon) / risky - 1
    if cash_symbol:
        forward = forward.sub(close[cash_symbol].shift(-horizon) / close[cash_symbol] - 1, axis=0)
    index = pd.MultiIndex.from_product([risky.index, risky.columns], names=["date", "symbol"])
    return pd.Series(forward.to_numpy().reshape(-1), index=index)


def make_model(kind: str, random_state: int) -> Any:
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_depth=3,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )


def training_cutoff(dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Last session of the previous calendar quarter: the model's knowledge horizon."""
    current = dates[-1].to_period("Q")
    earlier = dates[dates.to_period("Q") < current]
    return earlier[-1] if len(earlier) else None


class MLForecaster(Strategy):
    """Gradient-boosted (or logistic) classifier turned into conviction-weighted allocations."""

    name = "ml"

    def __init__(
        self,
        symbols: Sequence[str],
        cash_symbol: str | None,
        periods_per_year: int,
        config: MLConfig,
    ) -> None:
        super().__init__(symbols, cash_symbol, periods_per_year)
        self.config = config
        self._model: Any = None
        self._cutoff: pd.Timestamp | None = None
        self.predictions: list[dict[str, Any]] = []

    @property
    def warmup(self) -> int:
        # Features need a year, the model `min_train_days` of examples, and the first
        # quarterly refit can lag the decision date by up to a quarter.
        return FEATURE_LOOKBACK + self.config.min_train_days + 66

    def target_weights(self, history: PriceHistory) -> pd.Series:
        cutoff = training_cutoff(history.dates)
        if cutoff is None:
            return self._empty()
        if cutoff != self._cutoff:
            self._model = self._fit(history.until(cutoff))
            self._cutoff = cutoff
        if self._model is None:
            return self._empty()

        assets = eligible(history, self.symbols, FEATURE_LOOKBACK + 1)
        recent = history.close.iloc[-(FEATURE_LOOKBACK + 10) :]
        features = build_features(recent, self.symbols, self.cash_symbol)
        today = features.xs(recent.index[-1], level="date").loc[assets].dropna()
        if today.empty:
            return self._empty()
        probability = pd.Series(
            self._model.predict_proba(today.to_numpy())[:, 1], index=today.index
        )
        self._log(history.dates[-1], probability)

        conviction = ((probability - 0.5) / self.config.conviction_scale).clip(0.0, 1.0)
        vol = trailing_vol(
            history.close[conviction.index], self.config.vol_lookback, self.periods_per_year
        )
        weights = inverse_vol_weights(vol) * conviction
        return weights.reindex(self.symbols).fillna(0.0)

    def _fit(self, history: PriceHistory) -> Any:
        cfg = self.config
        if len(history) < FEATURE_LOOKBACK + cfg.min_train_days:
            return None
        close = history.close
        features = build_features(close, self.symbols, self.cash_symbol)
        labels = forward_excess_returns(close, self.symbols, self.cash_symbol, cfg.horizon)
        # Sample every few sessions, counting back from the newest labelled day, to
        # thin out heavily overlapping forward windows.
        labelled_days = close.index[: len(close) - cfg.horizon]
        sample_days = labelled_days[::-1][:: cfg.sample_every]
        rows = features.index.get_level_values("date").isin(sample_days)
        frame = features[rows].assign(label=labels[rows]).dropna()
        if len(frame) < MIN_TRAINING_ROWS or frame["label"].gt(0).nunique() < 2:
            return None
        model = make_model(cfg.model, cfg.random_state)
        model.fit(frame.drop(columns="label").to_numpy(), (frame["label"] > 0).to_numpy())
        return model

    def _log(self, day: pd.Timestamp, probability: pd.Series) -> None:
        self.predictions.extend(
            {"date": day, "symbol": symbol, "probability": float(p)}
            for symbol, p in probability.items()
        )

    def prediction_frame(self) -> pd.DataFrame:
        if not self.predictions:
            return pd.DataFrame(columns=["date", "symbol", "probability"])
        return pd.DataFrame(self.predictions)


def score_predictions(
    predictions: pd.DataFrame,
    history: PriceHistory,
    symbols: Sequence[str],
    cash_symbol: str | None,
    horizon: int,
) -> dict[str, float]:
    """Out-of-sample skill of logged predictions against what actually happened."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    if predictions.empty:
        return {}
    outcomes = forward_excess_returns(history.close, symbols, cash_symbol, horizon)
    keys = pd.MultiIndex.from_frame(predictions[["date", "symbol"]])
    realized = outcomes.reindex(keys).to_numpy()
    known = ~np.isnan(realized)
    if known.sum() < 30:
        return {}
    y = realized[known] > 0
    p = predictions["probability"].to_numpy()[known]
    if len(set(y)) < 2:
        return {}
    base_rate = float(y.mean())
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, np.full_like(p, base_rate))),
        "base_rate": base_rate,
        "accuracy": float(((p > 0.5) == y).mean()),
        "n": int(known.sum()),
    }
