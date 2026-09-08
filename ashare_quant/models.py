"""Shared domain models used by strategies, brokers, and services."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class SignalAction(str, Enum):
    """策略信号的交易动作：买入 / 卖出 / 持有（仅记录）。"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """委托方向：买入 / 卖出。"""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """委托生命周期状态：待成交 / 已成交 / 已拒绝 / 已撤销 / 失败。"""

    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TradeMode(str, Enum):
    """交易模式：PAPER 模拟盘 / LIVE 实盘。V1 默认且仅落地 PAPER。"""

    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class Signal:
    """策略生成的一条标准化调仓信号（不可变）。"""

    code: str
    action: SignalAction
    as_of_date: str
    strategy: str
    target_weight: float = 0.0
    score: float = 0.0
    reason: str = ""
    name: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class OrderRequest:
    """由信号转换而来、交给券商执行的一笔委托请求（不可变）。"""

    code: str
    side: OrderSide
    quantity: int
    strategy: str
    trade_date: str
    requested_price: Optional[float] = None
    reason: str = ""
    name: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Position:
    """单只证券的持仓快照（可变，随成交与估值更新）。"""

    code: str
    name: str
    quantity: int
    sellable_quantity: int
    avg_cost: float
    latest_price: float
    updated_at: str = ""

    @property
    def market_value(self) -> float:
        """持仓市值 = 数量 × 最新价。"""
        return self.quantity * self.latest_price

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏 =（最新价 − 成本价）× 数量。"""
        return (self.latest_price - self.avg_cost) * self.quantity


@dataclass
class Account:
    """账户资金快照。"""

    account_id: str
    cash: float
    market_value: float
    total_equity: float
    updated_at: str


def utc_now_text() -> str:
    """返回去时区、去微秒的 UTC 时间字符串（格式 ``YYYY-MM-DD HH:MM:SS``），用于落库统一时间戳。"""
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(sep=" ")


def market_open_fill_time(trade_date: str, order_id: str) -> str:
    """为模拟盘按开盘价撮合的成交生成 ``YYYY-MM-DD 09:30:SS`` 形式的时间字符串。

    A 股模拟撮合按当日开盘价成交，对应真实场景就是 09:30 连续竞价开始后若干秒。
    秒数 SS 由 ``order_id`` 的 md5 哈希分散到 0-59，使同日多笔成交的秒数互不重复，
    在 dashboard 的「成交记录」里看起来更接近真实交易流水。
    """
    base = (trade_date or "")[:10] or datetime.now().strftime("%Y-%m-%d")
    second = hashlib.md5((order_id or "").encode("utf-8")).digest()[0] % 60
    return f"{base} 09:30:{second:02d}"
