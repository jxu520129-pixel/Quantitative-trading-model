"""Pre-trade and account-level risk control for paper and future live brokers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .database import Database
from .models import OrderRequest, OrderSide, utc_now_text


@dataclass(frozen=True)
class RiskDecision:
    """一次风控判定的结果：是否放行 + 拒绝原因。"""

    allowed: bool
    reason: str = ""


class RiskManager:
    """交易前与账户级风控：仓位、现金、T+1、日亏损、连续失败熔断。"""

    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings

    def reset_for_new_day(self, trade_date: str) -> None:
        """新交易日的幂等重置：清空失败计数、日内停开与暂停状态（跨日才执行）。"""
        state = self.database.query_one("SELECT last_reset_date FROM risk_state WHERE account_id='paper'")
        if state and state["last_reset_date"] == trade_date:
            return
        self.database.execute(
            "UPDATE risk_state SET failure_count=0,daily_open_blocked=0,paused=0,last_reset_date=?,updated_at=? WHERE account_id='paper'",
            (trade_date, utc_now_text()),
        )

    def pre_trade_check(self, request: OrderRequest, estimate_price: float) -> RiskDecision:
        """委托前逐项校验，任一不满足即拒绝并给出原因。"""
        state = self.database.query_one("SELECT * FROM risk_state WHERE account_id='paper'") or {}
        if state.get("paused"):
            return RiskDecision(False, "连续执行失败，交易已暂停")
        if request.side == OrderSide.SELL:
            position = self.database.query_one("SELECT sellable_quantity FROM positions WHERE code=?", (request.code,))
            if not position or int(position["sellable_quantity"]) < request.quantity:
                return RiskDecision(False, "可卖数量不足，不符合 A 股 T+1 规则")
            return RiskDecision(True)
        if state.get("daily_open_blocked"):
            return RiskDecision(False, "已触发日亏损限制，停止新开仓")

        account = self.database.query_one("SELECT * FROM account_state WHERE account_id='paper'") or {}
        equity = float(account.get("total_equity", 0))
        cash = float(account.get("cash", 0))
        if equity <= 0:
            return RiskDecision(False, "账户总资产异常")
        estimated_amount = request.quantity * estimate_price
        if estimated_amount > cash:
            return RiskDecision(False, "可用现金不足")
        max_weight = float(self.settings.risk["max_single_position_weight"])
        current = self.database.query_one("SELECT quantity,latest_price FROM positions WHERE code=?", (request.code,))
        current_value = float(current["quantity"] * current["latest_price"]) if current else 0.0
        if (current_value + estimated_amount) / equity > max_weight + 1e-9:
            return RiskDecision(False, f"单一标的仓位将超过总资产的 {max_weight:.0%}")
        if not current:
            holding_count = self.database.query_one("SELECT COUNT(*) AS count FROM positions WHERE quantity>0")
            if int(holding_count["count"]) >= int(self.settings.risk["max_positions"]):
                return RiskDecision(False, "已达到最大持仓数量")
        daily_new = self.database.query_one(
            "SELECT COUNT(*) AS count FROM orders WHERE side='BUY' AND trade_date=? AND status='FILLED'", (request.trade_date,)
        )
        if int(daily_new["count"]) >= int(self.settings.risk["max_daily_new_positions"]):
            return RiskDecision(False, "已达到单日最大新开仓数量")
        self.evaluate_daily_loss(request.trade_date)
        refreshed = self.database.query_one("SELECT daily_open_blocked FROM risk_state WHERE account_id='paper'") or {}
        if refreshed.get("daily_open_blocked"):
            return RiskDecision(False, "已触发日亏损限制，停止新开仓")
        return RiskDecision(True)

    def evaluate_daily_loss(self, trade_date: str) -> None:
        """Block purchases when equity is down beyond the configured daily stop level."""
        account = self.database.query_one("SELECT total_equity FROM account_state WHERE account_id='paper'")
        baseline = self.database.query_one(
            """SELECT total_equity FROM account_snapshots WHERE account_id='paper' AND snapshot_date<?
               ORDER BY snapshot_date DESC,id DESC LIMIT 1""", (trade_date,)
        )
        if not account or not baseline or float(baseline["total_equity"]) <= 0:
            return
        loss = float(account["total_equity"]) / float(baseline["total_equity"]) - 1
        if loss <= -float(self.settings.risk["daily_loss_stop"]):
            self.database.execute("UPDATE risk_state SET daily_open_blocked=1,updated_at=? WHERE account_id='paper'", (utc_now_text(),))
            self.record_event("WARNING", "DAILY_LOSS", f"当日亏损 {loss:.2%}，已停止新开仓")

    def record_failure(self, message: str, code: str = "") -> None:
        """累计一次执行失败；连续失败达到阈值时暂停交易并记录风险事件。"""
        state = self.database.query_one("SELECT failure_count FROM risk_state WHERE account_id='paper'") or {"failure_count": 0}
        failures = int(state["failure_count"]) + 1
        paused = int(failures >= int(self.settings.risk["max_consecutive_failures"]))
        self.database.execute(
            "UPDATE risk_state SET failure_count=?,paused=?,updated_at=? WHERE account_id='paper'",
            (failures, paused, utc_now_text()),
        )
        category = "EXECUTION_PAUSED" if paused else "EXECUTION_FAILURE"
        self.record_event("ERROR", category, message, code)

    def record_success(self) -> None:
        """成交成功后清零连续失败计数。"""
        self.database.execute("UPDATE risk_state SET failure_count=0,updated_at=? WHERE account_id='paper'", (utc_now_text(),))

    def record_event(self, level: str, category: str, message: str, code: str = "") -> None:
        """写一条风险事件到 risk_events 表（供看板/审计回溯）。"""
        self.database.execute(
            "INSERT INTO risk_events(event_time,level,category,message,code) VALUES(?,?,?,?,?)",
            (utc_now_text(), level, category, message, code),
        )
