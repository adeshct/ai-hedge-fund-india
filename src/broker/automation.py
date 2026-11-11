"""Market-session, ticker-list, and account-level safeguards for watch mode."""

import csv
import json
import math
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
DEFAULT_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
DEFAULT_RISK_STATE = Path(__file__).resolve().parents[2] / "data" / "automation_risk_state.json"


def read_ticker_csv(path: Path | str) -> list[str]:
    """Read one ticker per row, with an optional ticker/symbol header."""
    tickers = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            values = [value.strip() for value in row]
            if not values or not any(values) or values[0].startswith("#"):
                continue
            if row_number == 1 and values[0].casefold() in {"ticker", "symbol"}:
                continue
            if len(values) != 1 or not values[0]:
                raise ValueError(f"{path}:{row_number}: expected exactly one ticker column.")
            if values[0].casefold() in {"ticker", "symbol"}:
                continue
            tickers.append(values[0])

    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        raise ValueError(f"No tickers found in {path}.")
    return unique_tickers


def is_nse_market_open(now: datetime | None = None) -> bool:
    now = (now or datetime.now(IST)).astimezone(IST)
    return now.weekday() < 5 and MARKET_OPEN <= now.time().replace(tzinfo=None) < MARKET_CLOSE


def load_risk_settings(path: Path | str = DEFAULT_SETTINGS) -> dict:
    with Path(path).open("r", encoding="utf-8") as settings_file:
        settings = yaml.safe_load(settings_file) or {}
    risk = settings.get("risk", {})
    if not isinstance(risk, dict):
        raise ValueError(f"Risk settings must be a mapping in {path}.")
    daily_limit = risk.get("daily_heat_limit", 0.06)
    risk_per_trade = risk.get("risk_per_trade", 0.02)
    max_positions = risk.get("max_positions", 15)
    if isinstance(daily_limit, bool) or not isinstance(daily_limit, (int, float)) or not 0 < daily_limit < 1:
        raise ValueError("risk.daily_heat_limit must be between 0 and 1.")
    if (isinstance(risk_per_trade, bool) or not isinstance(risk_per_trade, (int, float))
            or not 0 < risk_per_trade <= 1):
        raise ValueError("risk.risk_per_trade must be greater than 0 and at most 1.")
    if isinstance(max_positions, bool) or not isinstance(max_positions, int) or max_positions < 1:
        raise ValueError("risk.max_positions must be a positive integer.")
    return {
        "daily_heat_limit": float(daily_limit),
        "risk_per_trade": float(risk_per_trade),
        "max_positions": max_positions,
    }


def check_daily_loss_limit(
    portfolio: dict,
    prices: dict,
    limit_fraction: float,
    state_path: Path | str = DEFAULT_RISK_STATE,
    now: datetime | None = None,
) -> tuple[bool, float, float]:
    """Return halted, current equity, and session loss fraction; persist the session stop."""
    if not 0 < limit_fraction < 1 or not math.isfinite(limit_fraction):
        raise ValueError("Daily loss limit must be between 0 and 1.")
    now = (now or datetime.now(IST)).astimezone(IST)
    session_date = now.date().isoformat()
    cash = portfolio.get("cash")
    if not isinstance(cash, (int, float)) or isinstance(cash, bool) or not math.isfinite(cash) or cash < 0:
        raise ValueError("Portfolio cash must be a non-negative finite amount.")

    equity = float(cash)
    for ticker, position in portfolio.get("positions", {}).items():
        long_quantity = int(position.get("long", 0) or 0)
        short_quantity = int(position.get("short", 0) or 0)
        quantity = long_quantity - short_quantity
        if not quantity:
            continue
        price = prices.get(ticker)
        if not isinstance(price, (int, float)) or isinstance(price, bool) or not math.isfinite(price) or price <= 0:
            raise ValueError(f"Cannot evaluate daily loss: missing valid price for open position {ticker}.")
        equity += quantity * price
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("Portfolio equity must be positive and finite.")

    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(state, dict):
        raise ValueError(f"Invalid risk state file: {path}")
    if state.get("session_date") != session_date:
        state = {
            "session_date": session_date,
            "starting_equity": equity,
            "trading_halted": False,
        }
    starting_equity = state.get("starting_equity")
    if (
        not isinstance(starting_equity, (int, float))
        or isinstance(starting_equity, bool)
        or not math.isfinite(starting_equity)
        or starting_equity <= 0
    ):
        raise ValueError(f"Invalid starting equity in risk state file: {path}")
    if not isinstance(state.get("trading_halted"), bool):
        raise ValueError(f"Invalid trading-halted flag in risk state file: {path}")

    loss_fraction = max(0.0, (starting_equity - equity) / starting_equity)
    if loss_fraction >= limit_fraction:
        state["trading_halted"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return bool(state.get("trading_halted")), equity, loss_fraction
