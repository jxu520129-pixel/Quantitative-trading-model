"""QMT adapter boundary. Live order submission is intentionally disabled by default."""

from __future__ import annotations

import os

from .base import Broker


class QMTBroker(Broker):
    """Integration point for xtquant/miniQMT on Windows.

    Implement account binding and the broker-specific callbacks only after a paper
    test. Setting ENABLE_LIVE_TRADING=true is deliberately required at runtime.
    """

    def __init__(self, account_id: str, path: str):
        if os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
            raise RuntimeError("QMT 实盘交易未启用。仅在完成验证后才可设置 ENABLE_LIVE_TRADING=true")
        self.account_id = account_id
        self.path = path

    def _not_implemented(self):
        """统一抛错：提示需接入本地 xtquant 客户端并实现账户绑定。"""
        raise NotImplementedError("请接入本地 xtquant 客户端，并为当前 QMT 账户实现此券商适配器")

    def get_account(self):
        self._not_implemented()

    def get_positions(self):
        self._not_implemented()

    def submit_order(self, request):
        self._not_implemented()

    def execute_pending_orders(self, trade_date):
        self._not_implemented()

    def cancel_order(self, order_id):
        self._not_implemented()
