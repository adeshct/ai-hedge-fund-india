import json

from src.broker.execution import execute_decisions, load_paper_portfolio
from src.broker.models import InstrumentRef
from src.broker.zerodha import ZerodhaBroker


class FakeBroker:
    def __init__(self, prices):
        self.prices = prices
        self.orders = []

    def get_ltp(self, tickers):
        return {ticker: self.prices[ticker] for ticker in tickers if ticker in self.prices}

    def get_open_orders(self):
        return []

    def resolve(self, ticker):
        return InstrumentRef(exchange="NSE", tradingsymbol=ticker.removesuffix(".NS"), instrument_token=1)

    def place_limit_order(self, instrument, transaction_type, quantity, price, product="CNC"):
        order = (instrument.key, transaction_type, quantity, price, product)
        self.orders.append(order)
        return f"order-{len(self.orders)}"


class FakeKiteClient:
    VARIETY_REGULAR = "regular"
    ORDER_TYPE_LIMIT = "LIMIT"
    VALIDITY_DAY = "DAY"

    def __init__(self):
        self.order = None

    def orders(self):
        return []

    def place_order(self, **kwargs):
        self.order = kwargs
        return "kite-order"


def _portfolio(cash=1000, quantity=0):
    return {
        "cash": cash,
        "positions": {
            "RELIANCE.NS": {
                "long": quantity,
                "short": 0,
                "long_cost_basis": 90.0 if quantity else 0.0,
                "short_cost_basis": 0.0,
            }
        },
        "realized_gains": {"RELIANCE.NS": {"long": 0.0, "short": 0.0}},
    }


def test_paper_execution_applies_confidence_risk_and_cash_caps(tmp_path):
    path = tmp_path / "paper.json"
    portfolio = load_paper_portfolio(path, 1000, ["RELIANCE.NS"])
    results = execute_decisions(
        mode="paper",
        broker=None,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 10, "confidence": 80}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 250}},
        portfolio=portfolio,
        confidence_threshold=75,
        max_order_value=1000,
        risk_per_trade=0.25,
        paper_portfolio_path=path,
    )

    assert results[0]["status"] == "filled"
    assert results[0]["quantity"] == 2
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["cash"] == 800
    assert persisted["positions"]["RELIANCE.NS"]["long"] == 2
    assert persisted["positions"]["RELIANCE.NS"]["long_cost_basis"] == 100


def test_paper_execution_skips_low_confidence_and_short_actions(tmp_path):
    results = execute_decisions(
        mode="paper",
        broker=None,
        decisions={
            "RELIANCE.NS": {"action": "buy", "quantity": 2, "confidence": 74},
            "OTHER.NS": {"action": "short", "quantity": 2, "confidence": 99},
        },
        risk_signals={
            "RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 500},
            "OTHER.NS": {"current_price": 100, "remaining_position_limit": 500},
        },
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=1000,
        risk_per_trade=0.25,
        paper_portfolio_path=tmp_path / "paper.json",
    )

    assert [result["status"] for result in results] == ["skipped", "skipped"]
    assert not (tmp_path / "paper.json").exists()


def test_buy_size_scales_with_confidence(tmp_path):
    results = execute_decisions(
        mode="paper",
        broker=None,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 20, "confidence": 80}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 1000}},
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=1000,
        risk_per_trade=1.0,
        paper_portfolio_path=tmp_path / "paper.json",
    )

    assert results[0]["quantity"] == 8


def test_live_execution_submits_limit_order_capped_by_risk_and_value():
    broker = FakeBroker({"RELIANCE.NS": 100.03})
    results = execute_decisions(
        mode="live",
        broker=broker,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 10, "confidence": 90}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 250}},
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=250,
        risk_per_trade=0.25,
        enforce_market_hours=False,
    )

    assert results[0]["status"] == "submitted"
    assert results[0]["quantity"] == 2
    assert broker.orders == [("NSE:RELIANCE", "BUY", 2, 100.0, "CNC")]
    assert results[0]["order_id"] == "order-1"


def test_live_sell_is_limited_to_existing_long_shares():
    broker = FakeBroker({"RELIANCE.NS": 100})
    results = execute_decisions(
        mode="live",
        broker=broker,
        decisions={"RELIANCE.NS": {"action": "sell", "quantity": 10, "confidence": 90}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 0}},
        portfolio=_portfolio(quantity=3),
        confidence_threshold=75,
        max_order_value=1000,
        risk_per_trade=0.25,
        enforce_market_hours=False,
    )

    assert results[0]["quantity"] == 3
    assert broker.orders[0][1:3] == ("SELL", 3)


def test_zerodha_adapter_submits_delivery_limit_orders():
    client = FakeKiteClient()
    broker = ZerodhaBroker(client)

    order_id = broker.place_limit_order(
        InstrumentRef(exchange="NSE", tradingsymbol="RELIANCE", instrument_token=1),
        "BUY",
        2,
        100.0,
    )

    assert order_id == "kite-order"
    assert client.order == {
        "variety": "regular",
        "exchange": "NSE",
        "tradingsymbol": "RELIANCE",
        "transaction_type": "BUY",
        "quantity": 2,
        "product": "CNC",
        "order_type": "LIMIT",
        "price": 100.0,
        "validity": "DAY",
    }


def test_live_executor_skips_ticker_with_pending_order():
    class BrokerWithOpenOrder(FakeBroker):
        def get_open_orders(self):
            return [{"ticker": "RELIANCE.NS", "status": "OPEN"}]

    broker = BrokerWithOpenOrder({"RELIANCE.NS": 100})
    result = execute_decisions(
        mode="live",
        broker=broker,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 2, "confidence": 90}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 500}},
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=1000,
        enforce_market_hours=False,
    )

    assert result[0]["status"] == "skipped"
    assert "already open" in result[0]["reason"]
    assert broker.orders == []


def test_live_executor_fails_closed_outside_market_hours(monkeypatch):
    import src.broker.execution as execution

    monkeypatch.setattr(execution, "is_nse_market_open", lambda: False)
    broker = FakeBroker({"RELIANCE.NS": 100})
    result = execute_decisions(
        mode="live",
        broker=broker,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 2, "confidence": 90}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 500}},
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=1000,
    )

    assert result[0]["reason"] == "Outside NSE market hours"
    assert broker.orders == []


def test_watch_paper_execution_skips_after_market_close(monkeypatch):
    import src.broker.execution as execution

    monkeypatch.setattr(execution, "is_nse_market_open", lambda: False)
    result = execute_decisions(
        mode="paper",
        broker=None,
        decisions={"RELIANCE.NS": {"action": "buy", "quantity": 2, "confidence": 90}},
        risk_signals={"RELIANCE.NS": {"current_price": 100, "remaining_position_limit": 500}},
        portfolio=_portfolio(),
        confidence_threshold=75,
        max_order_value=1000,
        session_only=True,
    )

    assert result[0]["reason"] == "Outside NSE market hours"


def test_existing_paper_account_is_loaded_without_resetting_cash(tmp_path):
    path = tmp_path / "paper.json"
    path.write_text(json.dumps({"cash": 123, "positions": {}}), encoding="utf-8")

    portfolio = load_paper_portfolio(path, 1000, ["RELIANCE.NS"])

    assert portfolio["cash"] == 123
    assert portfolio["positions"]["RELIANCE.NS"]["long"] == 0
