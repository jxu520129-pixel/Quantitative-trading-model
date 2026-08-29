"""Shared domain models used by strategies, brokers, and services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TradeMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class Signal:
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
    code: str
    name: str
    quantity: int
    sellable_quantity: int
    avg_cost: float
    latest_price: float
    updated_at: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.latest_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.latest_price - self.avg_cost) * self.quantity


@dataclass
class Account:
    account_id: str
    cash: float
    market_value: float
    total_equity: float
    updated_at: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(sep=" ")
