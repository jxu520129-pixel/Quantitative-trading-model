"""Minimal broker contract for switching paper/QMT/PTrade adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Account, OrderRequest, Position


class Broker(ABC):
    @abstractmethod
    def get_account(self) -> Account:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute_pending_orders(self, trade_date: str) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError
