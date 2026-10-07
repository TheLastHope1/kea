import numpy as np
import pandas as pd
import pytest

from kea.config import MLConfig, MomentumConfig, TrendConfig
from kea.strategies import Ensemble, StaticWeights, build_strategy
from kea.strategies.ml import (
    MLForecaster,
    build_features,
    forward_excess_returns,
    training_cutoff,
)
from kea.strategies.trend import DualMomentum, TrendFollowing
from tests.conftest import make_history

RISKY = ["UP", "FLAT", "DOWN", "SPY"]


def assert_valid_weights(weights, symbols):
    assert list(weights.index) == list(symbols)
    assert (weights >= 0).all()
    assert weights.sum() <= 1 + 1e-9


def test_trend_backs_uptrends_and_avoids_downtrends(history):
    strategy = TrendFollowing(RISKY, "BIL", 252, TrendConfig(lookbacks=(126, 252)))
    weights = strategy.target_weights(history)
    assert_valid_weights(weights, RISKY)
    assert weights["UP"] > 0
    assert weights["DOWN"] == 0


def test_trend_holds_nothing_before_its_warmup(history):
    strategy = TrendFollowing(RISKY, "BIL", 252, TrendConfig())
    assert strategy.target_weights(history.head(100)).sum() == 0


def test_dual_momentum_picks_leaders_that_beat_cash(history):
    strategy = DualMomentum(RISKY, "BIL", 252, MomentumConfig(top_k=2))
    weights = strategy.target_weights(history)
    assert_valid_weights(weights, RISKY)
    assert weights["UP"] == pytest.approx(0.5)
    assert weights["DOWN"] == 0


def test_static_weights_skip_assets_without_data(history):
    strategy = StaticWeights("mix", {"SPY": 0.6, "NOPE": 0.4}, 252)
    weights = strategy.target_weights(history)
    assert weights.to_dict() == {"SPY": 0.6, "NOPE": 0.0}


def test_ensemble_averages_its_members(history):
    trend = TrendFollowing(RISKY, "BIL", 252, TrendConfig())
    momentum = DualMomentum(RISKY, "BIL", 252, MomentumConfig(top_k=2))
    blend = Ensemble([trend, momentum]).target_weights(history)
    expected = (trend.target_weights(history) + momentum.target_weights(history)) / 2
    pd.testing.assert_series_equal(blend, expected)


def test_build_strategy_knows_every_name(config):
    for name in ("trend", "momentum", "ml", "ensemble", "buy_and_hold", "sixty_forty"):
        assert build_strategy(config, name).name == name
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy(config, "astrology")


# --------------------------------------------------------------------- ML


def test_features_only_use_the_past(history):
    full = build_features(history.close, RISKY, "BIL")
    day = history.dates[400]
    truncated = build_features(history.until(day).close, RISKY, "BIL")
    pd.testing.assert_frame_equal(full.loc[day], truncated.loc[day])


def test_labels_for_the_last_horizon_days_are_unknown(history):
    labels = forward_excess_returns(history.close, RISKY, "BIL", horizon=21)
    last_dates = labels.index.get_level_values("date").unique()[-21:]
    assert labels.loc[last_dates].isna().all()
    assert labels.notna().sum() > 0


def test_training_cutoff_is_the_end_of_the_previous_quarter():
    dates = pd.bdate_range("2024-01-01", "2024-05-15")
    assert training_cutoff(dates) == pd.Timestamp("2024-03-29")
    assert training_cutoff(pd.bdate_range("2024-01-01", "2024-02-01")) is None


@pytest.mark.parametrize("model", ["gbm", "logistic"])
def test_ml_forecaster_trains_walk_forward_and_logs_predictions(model):
    history = make_history({"UP": 0.0008, "DOWN": -0.0008, "SPY": 0.0003, "BIL": 0.0001}, days=1500)
    config = MLConfig(model=model, min_train_days=600, sample_every=5)
    strategy = MLForecaster(["UP", "DOWN", "SPY"], "BIL", 252, config)
    weights = strategy.target_weights(history)
    assert_valid_weights(weights, ["UP", "DOWN", "SPY"])
    assert weights["UP"] >= weights["DOWN"]
    predictions = strategy.prediction_frame()
    assert set(predictions["symbol"]) <= {"UP", "DOWN", "SPY"}
    assert predictions["probability"].between(0, 1).all()


def test_ml_forecaster_is_deterministic():
    history = make_history({"UP": 0.0008, "DOWN": -0.0008, "BIL": 0.0001}, days=1300)
    config = MLConfig(min_train_days=600)
    first = MLForecaster(["UP", "DOWN"], "BIL", 252, config).target_weights(history)
    second = MLForecaster(["UP", "DOWN"], "BIL", 252, config).target_weights(history)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


def test_ml_forecaster_abstains_without_enough_training_data(history):
    strategy = MLForecaster(RISKY, "BIL", 252, MLConfig())
    assert strategy.target_weights(history.head(500)).sum() == 0


def test_trend_filter_holds_the_market_only_above_its_average():
    from kea.config import FilterConfig
    from kea.strategies.trend import TrendFilter

    rising = make_history(
        {"SPY": 0.002, "BIL": 0.0001}, days=300, vol={"SPY": 0.002, "BIL": 0.0003}
    )
    falling = make_history(
        {"SPY": -0.002, "BIL": 0.0001}, days=300, vol={"SPY": 0.002, "BIL": 0.0003}
    )
    strategy = TrendFilter(["SPY"], "BIL", 252, FilterConfig(sma_days=200), "SPY")
    assert strategy.target_weights(rising)["SPY"] == 1.0
    assert strategy.target_weights(falling)["SPY"] == 0.0
    assert strategy.target_weights(rising.head(150))["SPY"] == 0.0  # not enough history yet
    with pytest.raises(ValueError, match="must be one of"):
        TrendFilter(["SPY"], "BIL", 252, FilterConfig(), "QQQ")
