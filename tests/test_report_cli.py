import dataclasses
import json
from datetime import date

import pytest

from kea import cli
from kea.config import StrategyConfig
from kea.ledger import Ledger
from kea.report import render_report
from kea.research import compare


@pytest.fixture
def comparison(config, history):
    return compare(config, history, strategies=("trend", "momentum"))


def test_report_renders_every_section(config, history, comparison):
    config = dataclasses.replace(config, strategy=StrategyConfig(name="trend"))
    page = render_report(config, history, comparison)
    assert page.startswith("<!doctype html>")
    assert "<title>Kea Trading Report</title>" in page
    for chart in ("growth-chart", "drawdown-chart", "allocation-chart"):
        assert f'id="{chart}"' in page
    assert "Table view" in page
    assert 'id="paper"' not in page  # no ledger, no paper section
    fragment = render_report(config, history, comparison, standalone=False)
    assert fragment.startswith("<title>") and "<html" not in fragment


def test_report_escapes_untrusted_journal_text(config, history, tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_equity(date(2026, 1, 2), 10_000.0, 10_000.0, 0.0, 100.0)
    ledger.record(
        {
            "kind": "run",
            "status": "skipped",
            "as_of": "2026-01-02",
            "message": "broker said <script>alert(1)</script>",
            "account": {"cash": 1.0, "equity": 1.0, "positions": {}},
        }
    )
    page = render_report(config, history, ledger=ledger)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "Paper trading started" in page  # one data point: the empty state


def test_report_paper_section_with_a_track_record(config, history, tmp_path):
    ledger = Ledger(tmp_path)
    for day, equity, bench in [(date(2026, 1, 2), 10_000, 100), (date(2026, 1, 5), 10_250, 101)]:
        ledger.record_equity(day, equity, 500.0, 0.9, bench)
    ledger.record(
        {
            "kind": "run",
            "status": "traded",
            "message": "sent 2 order(s)",
            "account": {"cash": 500.0, "equity": 10_250.0, "positions": {"SPY": 3.0}},
            "orders": [{}, {}],
        }
    )
    page = render_report(config, history, ledger=ledger)
    assert 'id="paper-chart"' in page
    assert "+2.5%" in page  # return since start
    assert "Holdings" in page


class FakeMarketData:
    def __init__(self, history):
        self._history = history
        self.sources = dict.fromkeys(history.symbols, "fake")
        self.price_only: list[str] = []

    def history(self, symbols, now=None):
        return self._history.select(symbols)


def test_cli_backtest_compare_and_overrides(monkeypatch, tmp_path, capsys, history):
    monkeypatch.setattr(cli, "MarketData", lambda *a, **k: FakeMarketData(history))
    universe = [
        "-s", 'universe.symbols=["UP","FLAT","DOWN","SPY"]',
        "-s", "universe.cash_symbol=BIL",
        "-s", "universe.benchmark=SPY",
        "-s", f"broker.state_dir={tmp_path / 'state'}",
    ]  # fmt: skip
    assert cli.main([*universe, "backtest", "--strategy", "trend"]) == 0
    assert "CAGR" in capsys.readouterr().out

    out_json = tmp_path / "c.json"
    out_md = tmp_path / "c.md"
    code = cli.main(
        [
            *universe,
            "-s",
            'strategy.members=["trend"]',
            "compare",
            "--strategies",
            "trend,ml",
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text())
    assert {row["strategy"] for row in payload["strategies"]} >= {"trend", "buy_and_hold"}
    markdown = out_md.read_text()
    assert markdown.startswith("### Kea backtest")
    assert "Skipped ml: needs" in markdown  # 800 sessions is too short for the ML warm-up


def test_cli_reports_config_errors_cleanly(capsys):
    assert cli.main(["-s", "risk.target_volatility=0.1", "data"]) == 1
    assert "did you mean 'target_vol'" in capsys.readouterr().err


def test_report_studies_keep_benchmark_colours(config, history, comparison):
    from kea.report import Study, entity_colors

    page = render_report(
        config, history, comparison, studies=[Study("century", config, comparison)]
    )
    assert 'id="growth-chart"' in page and 'id="century-growth-chart"' in page
    assert "Every rule in this study" in page
    assert entity_colors(["trend", "buy_and_hold"]) == {"trend": "--s1", "buy_and_hold": "--s2"}
    assert entity_colors(["buy_and_hold", "trend", "momentum"])["buy_and_hold"] == "--s2"
