"""A-share trading constraints shared by backtest and paper execution."""

from __future__ import annotations

from dataclasses import dataclass


LOT_SIZE = 100


@dataclass(frozen=True)
class TradingCosts:
    """A 股交易费用参数（佣金、印花税、滑点），回测与模拟盘共用。"""

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.001
    slippage_rate: float = 0.0005

    def commission(self, amount: float) -> float:
        """按成交额计算佣金，含 5 元最低佣金下限。"""
        return max(abs(amount) * self.commission_rate, self.min_commission)

    def stamp_duty(self, amount: float, is_sell: bool) -> float:
        """计算印花税：仅卖出收取（A 股单边征收）。"""
        return abs(amount) * self.stamp_duty_rate if is_sell else 0.0

    def slipped_price(self, price: float, is_buy: bool) -> float:
        """按滑点调整成交价：买入上浮、卖出下调（单边滑点）。"""
        return price * (1 + self.slippage_rate) if is_buy else price * (1 - self.slippage_rate)


def round_to_lot(quantity: float) -> int:
    """把股数向下取整到 100 股整数倍（A 股最小交易单位）。"""
    return max(0, int(quantity // LOT_SIZE) * LOT_SIZE)


def board_price_limit(code: str, is_st: bool = False) -> float:
    """Return the normal daily limit. ETFs follow the 10% convention here.

    自 2026-07-06 起，沪深主板 ST/*ST 涨跌幅由 5% 上调至 10%（与普通股一致），
    科创板/创业板/北交所的 ST 维持对应板块 ±20%/±20%/±30%，故 ST 不再单独降档。
    ``is_st`` 参数仅为兼容保留，不再影响涨跌幅。
    """
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10


def at_limit(price: float, pre_close: float | None, code: str, side: str, is_st: bool = False) -> bool:
    """Whether an order is blocked at a one-sided daily price limit."""
    if not pre_close or pre_close <= 0:
        return False
    limit = board_price_limit(code, is_st)
    if side == "BUY":
        return price >= round(pre_close * (1 + limit), 2) - 0.005
    return price <= round(pre_close * (1 - limit), 2) + 0.005


def is_supported_security(code: str, security_type: str, include_etf: bool = True) -> bool:
    """允许所有 A 股与 ETF；ST/退市由 is_st / is_delisted 字段单独过滤。"""
    if security_type.upper() == "ETF":
        return include_etf
    if security_type.upper() != "STOCK":
        return False
    return True
