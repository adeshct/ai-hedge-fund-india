"""Confidence-gated order execution and persisted paper-trading state."""

import json
import math
from pathlib import Path
from typing import Optional

from src.broker.base import Broker

DEFAULT_PAPER_PORTFOLIO = Path(__file__).resolve().parents[2] / "data" / "paper_portfolio.json"


def load_paper_portfolio(path: Path | str, initial_cash: float, tickers: list[str]) -> dict:
    """Load the paper account, initializing it only when no account exists yet."""
    path = Path(path)
    if path.exists():
        portfolio = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(portfolio, dict)
            or not isinstance(portfolio.get("cash"), (int, float))
            or isinstance(portfolio.get("cash"), bool)
            or not math.isfinite(portfolio["cash"])
            or portfolio["cash"] < 0
        ):
            raise ValueError(f"Invalid paper portfolio file: {path}")
        if not isinstance(portfolio.get("positions"), dict):
            raise ValueError(f"Invalid paper portfolio positions: {path}")
        portfolio.setdefault("margin_requirement", 0.0)
        portfolio.setdefault("margin_used", 0.0)
        portfolio.setdefault("realized_gains", {})
        for ticker in tickers:
            portfolio["positions"].setdefault(ticker, _empty_position())
            portfolio["realized_gains"].setdefault(ticker, {"long": 0.0, "short": 0.0})
        return portfolio

    if initial_cash <= 0 or not math.isfinite(initial_cash):
        raise ValueError("Initial paper cash must be a positive finite amount.")
    return {
        "cash": float(initial_cash),
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {ticker: _empty_position() for ticker in tickers},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
    }


def execute_decisions(
    *,
    mode: str,
    broker: Optional[Broker],
    decisions: dict,
    risk_signals: dict,
    portfolio: dict,
    confidence_threshold: float,
    max_order_value: float,
    paper_portfolio_path: Path | str = DEFAULT_PAPER_PORTFOLIO,
) -> list[dict]:
    """Execute eligible long-only decisions within risk, cash, and notional limits."""
    if mode not in {"paper", "live"}:
        raise ValueError(f"Unsupported execution mode: {mode}")
    if mode == "live" and broker is None:
        raise ValueError("Live execution requires an authenticated broker.")
    if not 0 <= confidence_threshold <= 100 or not math.isfinite(confidence_threshold):
        raise ValueError("Confidence threshold must be between 0 and 100.")
    if max_order_value <= 0 or not math.isfinite(max_order_value):
        raise ValueError("Maximum order value must be positive and finite.")

    eligible_tickers = [
        ticker
        for ticker, decision in decisions.items()
        if isinstance(decision, dict)
        and decision.get("action") in {"buy", "sell"}
        and isinstance(decision.get("confidence"), (int, float))
        and not isinstance(decision.get("confidence"), bool)
        and math.isfinite(decision["confidence"])
        and confidence_threshold <= decision["confidence"] <= 100
        and isinstance(decision.get("quantity"), int)
        and not isinstance(decision.get("quantity"), bool)
        and decision["quantity"] > 0
    ]
    live_prices = broker.get_ltp(eligible_tickers) if mode == "live" and broker is not None and eligible_tickers else {}
    results = []
    portfolio_changed = False

    for ticker, decision in decisions.items():
        if not isinstance(decision, dict):
            results.append(_skipped(ticker, "Invalid decision"))
            continue

        action = decision.get("action")
        confidence = decision.get("confidence")
        requested = decision.get("quantity")
        if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not math.isfinite(confidence) or not 0 <= confidence <= 100):
            results.append(_skipped(ticker, "Invalid confidence"))
            continue
        if confidence < confidence_threshold:
            results.append(_skipped(ticker, f"Confidence {confidence:.0f}% below {confidence_threshold:.0f}% threshold"))
            continue
        if action not in {"buy", "sell"}:
            results.append(_skipped(ticker, f"Unsupported action: {action}"))
            continue
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            results.append(_skipped(ticker, "Invalid or zero quantity"))
            continue

        risk = risk_signals.get(ticker, {})
        risk_limit = risk.get("remaining_position_limit")
        if not isinstance(risk_limit, (int, float)) or not math.isfinite(risk_limit):
            results.append(_skipped(ticker, "Missing risk-manager position limit"))
            continue
        price = live_prices.get(ticker) if mode == "live" else risk.get("current_price")
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            results.append(_skipped(ticker, "Missing valid execution price"))
            continue

        position = portfolio.get("positions", {}).get(ticker, {})
        long_quantity = max(0, int(position.get("long", 0) or 0))
        cash = float(portfolio.get("cash", 0.0))
        if action == "buy":
            quantity = min(
                requested,
                max(0, int(max(0.0, risk_limit * confidence / 100) // price)),
                max(0, int(max(0.0, cash) // price)),
                max(0, int(max_order_value // price)),
            )
        else:
            quantity = min(requested, long_quantity, max(0, int(max_order_value // price)))

        if quantity <= 0:
            results.append(_skipped(ticker, "Blocked by position, cash, risk, or order-value limit"))
            continue

        order_price = float(price)
        order_id = None
        if mode == "live":
            instrument = broker.resolve(ticker)
            if not math.isfinite(instrument.tick_size) or instrument.tick_size <= 0:
                results.append(_skipped(ticker, "Invalid instrument tick size"))
                continue
            ticks = order_price / instrument.tick_size
            rounded_ticks = (
                math.floor(ticks + 1e-9)
                if action == "buy"
                else math.ceil(ticks - 1e-9)
            )
            order_price = rounded_ticks * instrument.tick_size
            quantity = min(quantity, max(0, int(max_order_value // order_price))) if order_price > 0 else 0
            if quantity <= 0:
                results.append(_skipped(ticker, "Price rounding exceeds order-value limit"))
                continue
            order_id = broker.place_limit_order(
                instrument,
                "BUY" if action == "buy" else "SELL",
                quantity,
                order_price,
            )
            # Reserve submitted buy value so subsequent orders cannot reuse cash.
            if action == "buy":
                portfolio["cash"] = cash - quantity * order_price
        else:
            _apply_paper_fill(portfolio, ticker, action, quantity, order_price)
            portfolio_changed = True

        results.append({
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "confidence": confidence,
            "price": order_price,
            "status": "submitted" if mode == "live" else "filled",
            "order_id": order_id,
        })

    if mode == "paper" and portfolio_changed:
        _save_paper_portfolio(Path(paper_portfolio_path), portfolio)
    return results


def _empty_position() -> dict:
    return {
        "long": 0,
        "short": 0,
        "long_cost_basis": 0.0,
        "short_cost_basis": 0.0,
        "short_margin_used": 0.0,
    }


def _apply_paper_fill(portfolio: dict, ticker: str, action: str, quantity: int, price: float) -> None:
    positions = portfolio.setdefault("positions", {})
    position = positions.setdefault(ticker, _empty_position())
    if action == "buy":
        old_quantity = max(0, int(position.get("long", 0) or 0))
        old_cost = float(position.get("long_cost_basis", 0.0) or 0.0)
        position["long_cost_basis"] = (old_quantity * old_cost + quantity * price) / (old_quantity + quantity)
        position["long"] = old_quantity + quantity
        portfolio["cash"] = float(portfolio["cash"]) - quantity * price
    else:
        old_quantity = max(0, int(position.get("long", 0) or 0))
        cost_basis = float(position.get("long_cost_basis", 0.0) or 0.0)
        position["long"] = old_quantity - quantity
        if not position["long"]:
            position["long_cost_basis"] = 0.0
        realized = portfolio.setdefault("realized_gains", {}).setdefault(ticker, {"long": 0.0, "short": 0.0})
        realized["long"] += (price - cost_basis) * quantity
        portfolio["cash"] = float(portfolio["cash"]) + quantity * price


def _save_paper_portfolio(path: Path, portfolio: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _skipped(ticker: str, reason: str) -> dict:
    return {"ticker": ticker, "status": "skipped", "reason": reason}
