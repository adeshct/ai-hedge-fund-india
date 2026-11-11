import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.broker.automation import (
    check_daily_loss_limit,
    is_nse_market_open,
    load_risk_settings,
    read_ticker_csv,
)

IST = ZoneInfo("Asia/Kolkata")


def test_read_ticker_csv_supports_header_comments_and_deduplication(tmp_path):
    path = tmp_path / "symbols.csv"
    path.write_text("ticker\nRELIANCE.NS\n# note\nTCS.NS\nRELIANCE.NS\n", encoding="utf-8")

    assert read_ticker_csv(path) == ["RELIANCE.NS", "TCS.NS"]


def test_read_ticker_csv_rejects_multiple_columns(tmp_path):
    path = tmp_path / "symbols.csv"
    path.write_text("ticker,weight\nRELIANCE.NS,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one ticker"):
        read_ticker_csv(path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 11, 9, 9, 15, tzinfo=IST), True),
        (datetime(2026, 11, 9, 15, 29, tzinfo=IST), True),
        (datetime(2026, 11, 9, 15, 30, tzinfo=IST), False),
        (datetime(2026, 11, 7, 10, 0, tzinfo=IST), False),
    ],
)
def test_market_hours(value, expected):
    assert is_nse_market_open(value) is expected


def test_daily_loss_limit_persists_halt_for_session(tmp_path):
    state_path = tmp_path / "risk.json"
    now = datetime(2026, 11, 9, 10, 0, tzinfo=IST)
    portfolio = {"cash": 1000.0, "positions": {}}

    assert check_daily_loss_limit(portfolio, {}, 0.06, state_path, now) == (False, 1000.0, 0.0)
    portfolio["cash"] = 930.0
    halted, equity, drawdown = check_daily_loss_limit(portfolio, {}, 0.06, state_path, now)
    assert halted is True
    assert equity == 930.0
    assert drawdown == pytest.approx(0.07)

    portfolio["cash"] = 1000.0
    halted, _, _ = check_daily_loss_limit(portfolio, {}, 0.06, state_path, now)
    assert halted is True
    assert json.loads(state_path.read_text(encoding="utf-8"))["trading_halted"] is True


def test_daily_loss_limit_fails_closed_for_unpriced_open_position(tmp_path):
    portfolio = {
        "cash": 1000.0,
        "positions": {"RELIANCE.NS": {"long": 2, "short": 0}},
    }

    with pytest.raises(ValueError, match="missing valid price"):
        check_daily_loss_limit(
            portfolio,
            {},
            0.06,
            tmp_path / "risk.json",
            datetime(2026, 11, 9, 10, 0, tzinfo=IST),
        )


def test_risk_settings_load_daily_and_per_trade_limits():
    settings = load_risk_settings()

    assert settings == {
        "daily_heat_limit": 0.06,
        "risk_per_trade": 0.02,
        "max_positions": 15,
    }
