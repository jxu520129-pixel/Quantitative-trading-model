"""PTrade adapter boundary for future server deployment."""

from __future__ import annotations

import os

from .base import Broker


class PTradeBroker(Broker):
    def __init__(self, account_id: str):
        if os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
            raise RuntimeError("PTrade 实盘交易未启用。仅在模拟验证完成后才可设置 ENABLE_LIVE_TRADING=true")
        self.account_id = account_id

    def _not_implemented(self):
        raise NotImplementedError("请依据券商提供的 PTrade 部署 SDK 与审批流程实现此适配器")

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
