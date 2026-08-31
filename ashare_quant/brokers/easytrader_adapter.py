"""Optional Eastmoney/easytrader boundary. Do not automate a real GUI by default."""

from __future__ import annotations

from .base import Broker


class EasyTraderBroker(Broker):
    """东方财富/easytrader 接入边界（V1 明确禁用，初始化即抛错，防止误自动化真实 GUI）。"""

    def __init__(self, client_name: str = "eastmoney"):
        self.client_name = client_name
        raise RuntimeError(
            "V1 未启用 easytrader 登录自动化。请使用内置模拟账户；"
            "接入真实东方财富账户前必须补充券商专属校验。"
        )

    def get_account(self):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError

    def submit_order(self, request):
        raise NotImplementedError

    def execute_pending_orders(self, trade_date):
        raise NotImplementedError

    def cancel_order(self, order_id):
        raise NotImplementedError
