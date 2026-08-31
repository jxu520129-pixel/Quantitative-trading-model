"""Minimal broker contract for switching paper/QMT/PTrade adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Account, OrderRequest, Position


class Broker(ABC):
    """券商适配器统一契约：模拟盘与未来实盘（QMT/PTrade）实现同一套接口。"""

    @abstractmethod
    def get_account(self) -> Account:
        """返回账户资金快照。"""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """返回全部非零持仓。"""
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> str:
        """提交一笔委托，返回委托 id。"""
        raise NotImplementedError

    @abstractmethod
    def execute_pending_orders(self, trade_date: str) -> dict[str, int]:
        """撮合待执行委托，返回 ``{filled, rejected, failed}`` 统计。"""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """撤销一笔待成交委托。"""
        raise NotImplementedError
