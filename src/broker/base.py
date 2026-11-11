"""Broker abstraction for account data and guarded order execution."""

from abc import ABC, abstractmethod
from typing import Optional

from src.broker.models import BrokerPosition, Funds, InstrumentRef


class Broker(ABC):
    """Broker account interface."""

    @abstractmethod
    def get_profile(self) -> dict:
        """Authenticated account profile (user id, broker, enabled exchanges)."""

    @abstractmethod
    def get_holdings(self) -> list[BrokerPosition]:
        """Settled delivery holdings (T+1 book)."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Net positions for the current day, including intraday."""

    @abstractmethod
    def get_funds(self) -> Funds:
        """Available cash and margin utilisation."""

    @abstractmethod
    def get_ltp(self, tickers: list[str]) -> dict[str, float]:
        """Last traded price per ticker; unresolved tickers are omitted."""

    @abstractmethod
    def resolve(self, ticker: str, exchange: Optional[str] = None) -> InstrumentRef:
        """Map an internal ticker to its exchange instrument."""

    @abstractmethod
    def get_open_orders(self) -> list[dict]:
        """Return active orders with their internal ticker and status."""

    def place_limit_order(
        self,
        instrument: InstrumentRef,
        transaction_type: str,
        quantity: int,
        price: float,
        product: str = "CNC",
    ) -> str:
        """Submit a delivery limit order and return the broker order ID."""
        raise NotImplementedError("This broker does not support order placement.")
