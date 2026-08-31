"""Translate signals into lot-sized paper orders and run the morning fill cycle."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..brokers.paper import PaperBroker
from ..config import Settings
from ..data.service import DataService
from ..database import Database
from ..market_rules import round_to_lot
from ..models import OrderRequest, OrderSide
from ..notifications import NotificationHub
from ..utils import normalize_date, today_text


def next_weekday(value: str) -> str:
    """返回给定日期的下一交易日（跳过周六周日，不含节假日判断）。"""
    day = pd.Timestamp(normalize_date(value)) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.date().isoformat()


class TradingService:
    """把策略信号转换为整手委托，并执行晨间模拟成交流程。"""

    def __init__(
        self, database: Database, data: DataService, broker: PaperBroker,
        settings: Settings, notifications: NotificationHub,
    ):
        self.database = database
        self.data = data
        self.broker = broker
        self.settings = settings
        self.notifications = notifications

    def queue_new_signals(self, as_of_date: str | None = None) -> dict[str, int]:
        """把状态为 NEW 的信号转换为委托排队（卖出优先、按评分降序），返回排队/跳过计数。"""
        clauses = ["status='NEW'"]
        params: list[str] = []
        if as_of_date:
            clauses.append("as_of_date=?")
            params.append(normalize_date(as_of_date))
        rows = self.database.query_all(
            f"SELECT * FROM signals WHERE {' AND '.join(clauses)} ORDER BY CASE action WHEN 'SELL' THEN 0 ELSE 1 END,score DESC",
            params,
        )
        outcome = {"queued": 0, "skipped": 0}
        new_buys = 0
        max_new = int(self.settings.risk["max_daily_new_positions"])
        account = self.broker.get_account()
        positions = {position.code: position for position in self.broker.get_positions()}
        for signal in rows:
            execution_date = next_weekday(signal["as_of_date"])
            bar = self.data.latest_bar(signal["code"])
            if not bar or float(bar["close"]) <= 0:
                self._set_signal_status(signal["id"], "SKIPPED")
                outcome["skipped"] += 1
                continue
            price = float(bar["close"])
            if signal["action"] == "SELL":
                position = positions.get(signal["code"])
                quantity = round_to_lot(position.sellable_quantity if position else 0)
                side = OrderSide.SELL
            else:
                if new_buys >= max_new:
                    self._set_signal_status(signal["id"], "DEFERRED")
                    outcome["skipped"] += 1
                    continue
                current_value = positions.get(signal["code"]).market_value if signal["code"] in positions else 0.0
                target_value = account.total_equity * min(
                    float(signal["target_weight"]), float(self.settings.risk["max_single_position_weight"])
                ) * 0.995  # Leave room for open-price movement, slippage, and minimum commission.
                quantity = round_to_lot(max(0.0, target_value - current_value) / price)
                side = OrderSide.BUY
                new_buys += 1
            if quantity <= 0:
                self._set_signal_status(signal["id"], "SKIPPED")
                outcome["skipped"] += 1
                continue
            self.broker.submit_order(OrderRequest(
                code=signal["code"], name=signal["name"], side=side, quantity=quantity,
                strategy=signal["strategy"], trade_date=execution_date,
                requested_price=price, reason=signal["reason"],
            ))
            self._set_signal_status(signal["id"], "QUEUED")
            outcome["queued"] += 1
        return outcome

    def execute_morning(self, trade_date: str | None = None, refresh_quotes: bool = True) -> dict[str, int]:
        """执行晨间成交流程：刷新待执行标的最新价 → 滚动交易日 → 撮合委托，返回成交统计。"""
        trade_date = normalize_date(trade_date or today_text())
        pending = self.database.query_all("SELECT DISTINCT code FROM orders WHERE status='PENDING' AND trade_date<=?", (trade_date,))
        if refresh_quotes and pending:
            try:
                self.data.refresh_quotes([item["code"] for item in pending])
            except Exception as error:
                self.notifications.send("实时行情获取失败", str(error), "WARNING")
        self.broker.roll_to_new_day(trade_date)
        outcome = self.broker.execute_pending_orders(trade_date)
        if sum(outcome.values()):
            summary = f"成交 {outcome['filled']} 笔，拒绝 {outcome['rejected']} 笔，失败 {outcome['failed']} 笔"
            self.notifications.send("模拟盘成交报告", summary, "INFO" if not outcome["failed"] else "ERROR")
        return outcome

    def queue_manual_close(self, code: str, as_of_date: str | None = None) -> str:
        """看板手动全量平仓：对指定持仓提交一笔卖出委托，返回委托 id。"""
        position = next((item for item in self.broker.get_positions() if item.code == code), None)
        if not position:
            raise ValueError(f"未找到证券 {code} 的持仓")
        quantity = round_to_lot(position.sellable_quantity)
        if quantity <= 0:
            raise ValueError("该持仓当日不可卖出，不符合 T+1 规则")
        request = OrderRequest(
            code=code, name=position.name, side=OrderSide.SELL, quantity=quantity,
            strategy="manual", trade_date=next_weekday(as_of_date or today_text()), reason="看板手动全量平仓",
        )
        return self.broker.submit_order(request)

    def _set_signal_status(self, signal_id: str, status: str) -> None:
        """更新信号状态（NEW → QUEUED / SKIPPED / DEFERRED 等）。"""
        self.database.execute("UPDATE signals SET status=? WHERE id=?", (status, signal_id))
