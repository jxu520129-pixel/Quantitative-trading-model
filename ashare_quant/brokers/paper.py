"""SQLite-backed paper broker with A-share fees, T+1, lot size and price limits."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..database import Database
from ..market_rules import LOT_SIZE, TradingCosts, at_limit, board_price_limit, round_to_lot
from ..models import Account, OrderRequest, OrderSide, OrderStatus, Position, utc_now_text
from ..risk import RiskManager
from ..utils import normalize_date
from .base import Broker


class PaperBroker(Broker):
    """基于 SQLite 的模拟券商：落地 A 股费用、T+1、整手与涨跌停限制，资金与持仓全量记账。"""

    account_id = "paper"

    def __init__(self, database: Database, settings: Settings, risk_manager: RiskManager):
        self.database = database
        self.settings = settings
        self.risk = risk_manager
        self.costs = TradingCosts(
            commission_rate=float(settings.trading["commission_rate"]),
            min_commission=float(settings.trading["min_commission"]),
            stamp_duty_rate=float(settings.trading["stamp_duty_rate"]),
            slippage_rate=float(settings.trading["slippage_rate"]),
        )

    def get_account(self) -> Account:
        """读取模拟账户资金快照。"""
        row = self.database.query_one("SELECT * FROM account_state WHERE account_id=?", (self.account_id,))
        if not row:
            raise RuntimeError("模拟账户尚未初始化")
        return Account(**row)

    def get_positions(self) -> list[Position]:
        """读取全部非零持仓，按市值降序返回。"""
        rows = self.database.query_all("SELECT * FROM positions WHERE quantity>0 ORDER BY latest_price*quantity DESC")
        return [Position(**row) for row in rows]

    def submit_order(self, request: OrderRequest) -> str:
        """校验整手后把委托以 PENDING 状态落库，返回委托 id。"""
        if request.quantity <= 0 or request.quantity % LOT_SIZE != 0:
            raise ValueError(f"A 股委托数量必须是大于 0 的 {LOT_SIZE} 股整数倍")
        self.database.execute(
            """INSERT INTO orders(id,code,name,side,quantity,requested_price,status,strategy,trade_date,reason,error,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?, ?,?,?,?, ?,?)""",
            (request.id, request.code, request.name, request.side.value, request.quantity, request.requested_price,
             OrderStatus.PENDING.value, request.strategy, normalize_date(request.trade_date), request.reason, "", utc_now_text(), utc_now_text()),
        )
        return request.id

    def cancel_order(self, order_id: str) -> None:
        """撤销一笔仍处于 PENDING 状态的委托。"""
        self.database.execute(
            "UPDATE orders SET status=?,updated_at=? WHERE id=? AND status='PENDING'",
            (OrderStatus.CANCELLED.value, utc_now_text(), order_id),
        )

    def roll_to_new_day(self, trade_date: str) -> None:
        """Purchased shares become sellable on the next trade day; daily stops reset."""
        trade_date = normalize_date(trade_date)
        state = self.database.query_one("SELECT last_reset_date FROM risk_state WHERE account_id='paper'")
        if not state or state["last_reset_date"] != trade_date:
            self.database.execute("UPDATE positions SET sellable_quantity=quantity,updated_at=?", (utc_now_text(),))
            self.risk.reset_for_new_day(trade_date)

    def execute_pending_orders(self, trade_date: str) -> dict[str, int]:
        """撮合当日待执行委托（卖出优先）：涨跌停拦截、逐笔成交/拒绝/失败，返回统计。"""
        trade_date = normalize_date(trade_date)
        rows = self.database.query_all(
            "SELECT * FROM orders WHERE status='PENDING' AND trade_date<=? ORDER BY CASE side WHEN 'SELL' THEN 0 ELSE 1 END,created_at",
            (trade_date,),
        )
        outcome = {"filled": 0, "rejected": 0, "failed": 0}
        for order in rows:
            bar = self.database.query_one(
                "SELECT * FROM daily_bars WHERE code=? AND trade_date=?", (order["code"], trade_date)
            )
            quote = self.database.query_one(
                "SELECT price AS open,pre_close,quote_time FROM market_quotes WHERE code=? AND substr(quote_time,1,10)=?",
                (order["code"], trade_date),
            )
            if quote:
                bar = quote
            if not bar and self.settings.trading.get("allow_stale_demo_prices", False):
                bar = self.database.query_one(
                    "SELECT close AS open,pre_close,trade_date FROM daily_bars WHERE code=? ORDER BY trade_date DESC LIMIT 1",
                    (order["code"],),
                )
            if not bar:
                self._reject(order, "没有可用于成交的行情数据")
                outcome["rejected"] += 1
                continue
            try:
                if at_limit(float(bar["open"]), bar["pre_close"], order["code"], order["side"]):
                    pre_close = float(bar["pre_close"])
                    open_price = float(bar["open"])
                    limit_pct = board_price_limit(order["code"], is_st=False)
                    if order["side"] == "BUY":
                        limit_price = round(pre_close * (1 + limit_pct), 2)
                        detail = (f"涨停拦截 {order['code']}：开盘价{open_price:.2f} ≥ 涨停价{limit_price:.2f}（昨收{pre_close:.2f}，{limit_pct:.0%}涨停）")
                    else:
                        limit_price = round(pre_close * (1 - limit_pct), 2)
                        detail = (f"跌停拦截 {order['code']}：开盘价{open_price:.2f} ≤ 跌停价{limit_price:.2f}（昨收{pre_close:.2f}，{limit_pct:.0%}跌停）")
                    self._reject(order, detail)
                    outcome["rejected"] += 1
                    continue
                self._fill(order, bar, trade_date)
                self.risk.record_success()
                outcome["filled"] += 1
            except ValueError as error:
                self._reject(order, str(error))
                outcome["rejected"] += 1
            except Exception as error:  # Preserve order history and trip failure circuit if execution itself breaks.
                self.database.execute(
                    "UPDATE orders SET status='FAILED',error=?,updated_at=? WHERE id=?", (str(error), utc_now_text(), order["id"])
                )
                self.risk.record_failure(str(error), order["code"])
                outcome["failed"] += 1
        self.mark_to_market(trade_date)
        return outcome

    def _fill(self, order: dict[str, Any], bar: dict[str, Any], trade_date: str) -> None:
        """以含滑点的开盘价成交一笔委托：更新资金、持仓、成交记录（买/卖分别记账）。"""
        side = order["side"]
        raw_price = float(bar["open"])
        price = raw_price * (1 + self.costs.slippage_rate if side == "BUY" else 1 - self.costs.slippage_rate)
        price = round(price, 2)
        requested = int(order["quantity"])
        if side == "BUY":
            decision = self.risk.pre_trade_check(
                OrderRequest(code=order["code"], side=OrderSide.BUY,
                             quantity=requested, strategy=order["strategy"], trade_date=trade_date, requested_price=price,
                             reason=order["reason"], name=order["name"], id=order["id"]),
                price,
            )
            if not decision.allowed:
                raise ValueError(decision.reason)
        with self.database.transaction() as conn:
            account = dict(conn.execute("SELECT * FROM account_state WHERE account_id='paper'").fetchone())
            position_row = conn.execute("SELECT * FROM positions WHERE code=?", (order["code"],)).fetchone()
            position = dict(position_row) if position_row else None
            quantity = requested
            if side == "BUY":
                # Recalculate affordable lots with current cash to keep a queued order from overdrawing the account.
                while quantity > 0:
                    amount = quantity * price
                    if amount + self.costs.commission(amount) <= float(account["cash"]):
                        break
                    quantity -= LOT_SIZE
                if quantity <= 0:
                    raise ValueError("按成交价格计算后可用现金不足")
            else:
                if not position or int(position["sellable_quantity"]) <= 0:
                    raise ValueError("当日无可卖数量，不符合 T+1 规则")
                quantity = min(quantity, int(position["sellable_quantity"]))
                quantity = round_to_lot(quantity)
                if quantity <= 0:
                    raise ValueError("卖出数量不足一手")
            amount = quantity * price
            commission = self.costs.commission(amount)
            stamp_duty = self.costs.stamp_duty(amount, side == "SELL")
            if side == "BUY":
                cash = float(account["cash"]) - amount - commission
                old_quantity = int(position["quantity"]) if position else 0
                old_cost = float(position["avg_cost"]) if position else 0.0
                new_quantity = old_quantity + quantity
                average_cost = (old_quantity * old_cost + amount + commission) / new_quantity
                if position:
                    conn.execute(
                        "UPDATE positions SET quantity=?,sellable_quantity=?,avg_cost=?,latest_price=?,updated_at=? WHERE code=?",
                        (new_quantity, int(position["sellable_quantity"]), average_cost, price, utc_now_text(), order["code"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO positions(code,name,quantity,sellable_quantity,avg_cost,latest_price,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (order["code"], order["name"] or order["code"], new_quantity, 0, average_cost, price, utc_now_text()),
                    )
            else:
                cash = float(account["cash"]) + amount - commission - stamp_duty
                remaining = int(position["quantity"]) - quantity
                sellable = int(position["sellable_quantity"]) - quantity
                if remaining <= 0:
                    conn.execute("DELETE FROM positions WHERE code=?", (order["code"],))
                else:
                    conn.execute(
                        "UPDATE positions SET quantity=?,sellable_quantity=?,latest_price=?,updated_at=? WHERE code=?",
                        (remaining, max(0, sellable), price, utc_now_text(), order["code"]),
                    )
            conn.execute(
                "UPDATE account_state SET cash=?,updated_at=? WHERE account_id='paper'", (cash, utc_now_text())
            )
            conn.execute(
                """UPDATE orders SET fill_price=?,filled_quantity=?,status='FILLED',error='',updated_at=? WHERE id=?""",
                (price, quantity, utc_now_text(), order["id"]),
            )
            conn.execute(
                """INSERT INTO fills(order_id,code,side,quantity,price,gross_amount,commission,stamp_duty,filled_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (order["id"], order["code"], side, quantity, price, amount, commission, stamp_duty, utc_now_text()),
            )

    def _reject(self, order: dict[str, Any], message: str) -> None:
        """把委托标记为 REJECTED 并累计一次失败（用于熔断计数）。"""
        self.database.execute(
            "UPDATE orders SET status='REJECTED',error=?,updated_at=? WHERE id=?", (message, utc_now_text(), order["id"])
        )
        self.risk.record_failure(message, order["code"])

    def mark_to_market(self, trade_date: str) -> Account:
        """Mark current positions at close, update account state and upsert a daily snapshot."""
        trade_date = normalize_date(trade_date)
        with self.database.transaction() as conn:
            positions = conn.execute("SELECT * FROM positions").fetchall()
            market_value = 0.0
            for raw in positions:
                position = dict(raw)
                bar = conn.execute("SELECT close FROM daily_bars WHERE code=? AND trade_date=?", (position["code"], trade_date)).fetchone()
                price = float(bar["close"]) if bar else float(position["latest_price"])
                market_value += int(position["quantity"]) * price
                conn.execute("UPDATE positions SET latest_price=?,updated_at=? WHERE code=?", (price, utc_now_text(), position["code"]))
            account = dict(conn.execute("SELECT * FROM account_state WHERE account_id='paper'").fetchone())
            total = float(account["cash"]) + market_value
            # 当日盈亏以「上一个交易日」的快照为基线，避免同日多次估值互相污染口径。
            previous = conn.execute(
                "SELECT total_equity FROM account_snapshots WHERE account_id='paper' AND snapshot_date<? "
                "ORDER BY snapshot_date DESC, id DESC LIMIT 1", (trade_date,)
            ).fetchone()
            daily_pnl = total - float(previous["total_equity"]) if previous else 0.0
            now = utc_now_text()
            existing = conn.execute(
                "SELECT id FROM account_snapshots WHERE account_id='paper' AND snapshot_date=?", (trade_date,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE account_snapshots SET cash=?,market_value=?,total_equity=?,daily_pnl=?,created_at=? WHERE id=?",
                    (float(account["cash"]), market_value, total, daily_pnl, now, existing["id"]),
                )
            else:
                conn.execute(
                    """INSERT INTO account_snapshots(account_id,snapshot_date,cash,market_value,total_equity,daily_pnl,created_at)
                       VALUES('paper',?,?,?,?,?,?)""",
                    (trade_date, float(account["cash"]), market_value, total, daily_pnl, now),
                )
            conn.execute(
                "UPDATE account_state SET market_value=?,total_equity=?,updated_at=? WHERE account_id='paper'",
                (market_value, total, now),
            )
        return self.get_account()
