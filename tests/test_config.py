from datetime import date
from pathlib import Path

import pytest

from kea.config import Config, ConfigError, config_from_dict, load_config

CONFIG_DIR = Path(__file__).parent.parent / "config"


def test_defaults_are_valid():
    config = load_config(None)
    assert config == Config()
    assert config.universe.tradable[-1] == "SHY"
    assert "ml" not in config.strategy.members  # opt-in until it shows real skill
    assert config.risk.max_gross <= 1.0


def test_nested_overrides_and_type_coercion():
    config = config_from_dict(
        {
            "universe": {"symbols": ["SPY", "TLT"], "cash_symbol": "none"},
            "data": {"start": "2010-01-31"},
            "strategy": {"rebalance": "weekly", "trend": {"lookbacks": [21, 252]}},
            "risk": {"target_vol": 0.12},
        }
    )
    assert config.universe.symbols == ("SPY", "TLT")
    assert config.universe.cash_symbol is None
    assert config.data.start == date(2010, 1, 31)
    assert config.strategy.trend.lookbacks == (21, 252)
    assert config.risk.target_vol == 0.12


def test_unknown_key_suggests_the_right_one():
    with pytest.raises(ConfigError, match=r"unknown setting 'risk.target_volatility'.*target_vol"):
        config_from_dict({"risk": {"target_volatility": 0.1}})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"strategy": {"rebalance": "daily"}}, "must be one of"),
        ({"risk": {"target_vol": "high"}}, "must be a number"),
        ({"risk": {"max_gross": 1.5}}, "never uses leverage"),
        ({"execution": {"whole_shares": "yes"}}, "true or false"),
        ({"universe": {"symbols": ["SPY", "SPY"]}}, "duplicates"),
        ({"universe": "SPY"}, "must be a table"),
    ],
)
def test_invalid_values_are_rejected(raw, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    assert isinstance(load_config(path), Config)
