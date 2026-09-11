"""Translate signals into lot-sized paper orders and run the morning fill cycle."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pandas as pd

from ..brokers.paper import PaperBroker
from ..config import Settings
from ..data.service import DataService
from ..database import Database
from ..market_rules import round_to_lot
from ..models import OrderRequest, OrderSide, Signal, intraday_fill_time
from ..notifications import NotificationHub
from ..utils import normalize_date, today_text


LOG = logging.getLogger(__name__)


def _quote_price(value: Any) -> float:
    """把行情值（``{"price":x}`` 或 ``x``）统一取成价格浮点数。"""
    if isinstance(value, dict):
        value = value.get("price")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


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

    # ---------------------------------------------------------------- 盘中即时撮合
    def execute_intraday(
        self,
        quotes: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
        exit_signals: list[Signal] | None = None,
        trade_date: str | None = None,
    ) -> dict[str, int]:
        """盘中即时撮合：以**实时价**先卖后买，当日即时成交，不再排队到次日开盘。

        这是模拟盘买点的唯一入口——旧的「15:30 生成信号 → 次日 09:26 按开盘价成交」已废弃。
        ``quotes`` 为实时行情（``{code: {"price","volume"}}`` 或 ``{code: price}``）；
        ``exit_signals`` 为持仓离场信号（``SignalService.generate(codes=持仓, current_quotes=...)``
        用实时价评估，含止损/止盈/移动止损/跌破箱体/离场条件）；
        ``candidates`` 为买点扫描候选（``lab.scan_buy_candidates`` 的结果，与邮件报告同一口径）。

        **实时行情不可用时直接返回、不做任何成交**：宁可当天不出手，也不拿陈旧价格成交。
        """
        trade_date = normalize_date(trade_date or today_text())
        outcome = {"buy_filled": 0, "buy_rejected": 0, "sell_filled": 0, "sell_rejected": 0,
                   "failed": 0, "skipped": 0}
        prices = {
            code: price
            for code, price in ((str(c), _quote_price(v)) for c, v in (quotes or {}).items())
            if price > 0
        }
        if not prices:
            outcome["skipped"] = 1
            LOG.warning("实时行情不可用，跳过本次盘中即时撮合（不做任何成交）")
            return outcome

        positions = {item.code: item for item in self.broker.get_positions()}
        submitted: list[tuple[str, str, Signal | None]] = []  # (side, order_id, 离场信号)
        # ① 卖出：实时价触发离场 → 立即提交（先卖，释放名额与资金）
        for signal in exit_signals or []:
            order_id = self._submit_exit(signal, prices, positions, trade_date)
            if order_id:
                submitted.append(("SELL", order_id, signal))
        # ② 买入：买点扫描候选 → 名额/去重过滤 → 立即提交
        for pick in self._pick_buys(candidates, prices, positions, trade_date):
            order_id = self._submit_buy(pick, prices[pick["code"]], trade_date)
            if order_id:
                submitted.append(("BUY", order_id, None))
        if not submitted:
            return outcome

        self.broker.execute_pending_orders(
            trade_date, price_overrides=prices, filled_at=intraday_fill_time(trade_date),
        )
        # 回填结果：成交时间已是真实时点（不再写死 09:30），信号状态同步落库
        for side, order_id, exit_signal in submitted:
            row = self.database.query_one("SELECT status, error FROM orders WHERE id=?", (order_id,))
            status = str(row["status"]) if row else "FAILED"
            if status == "FILLED":
                outcome["sell_filled" if side == "SELL" else "buy_filled"] += 1
            elif status == "FAILED":
                outcome["failed"] += 1
            else:
                outcome["sell_rejected" if side == "SELL" else "buy_rejected"] += 1
            if exit_signal is not None:
                self._mark_signal(exit_signal, status)
            if status != "FILLED":
                LOG.info("盘中%s未成交（%s）：%s", "卖出" if side == "SELL" else "买入", order_id,
                         str(row["error"]) if row else "")
        return outcome

    def _submit_exit(self, signal: Signal, prices: dict[str, float],
                     positions: dict[str, Any], trade_date: str) -> str:
        """提交一笔持仓离场卖出委托；无可卖数量（T+1 或已无仓位）时返回空串。"""
        position = positions.get(signal.code)
        price = prices.get(signal.code)
        if not position or price is None:
            return ""
        quantity = round_to_lot(position.sellable_quantity)
        if quantity <= 0:
            LOG.info("持仓 %s 当日无可卖数量（T+1），跳过离场", signal.code)
            return ""
        return self.broker.submit_order(OrderRequest(
            code=signal.code, name=position.name, side=OrderSide.SELL, quantity=quantity,
            strategy=signal.strategy, trade_date=trade_date,
            requested_price=price, reason=signal.reason or "盘中离场",
        ))

    def _submit_buy(self, pick: dict[str, Any], price: float, trade_date: str) -> str:
        """按目标权重提交一笔买入委托（单票不超风险上限，与既有调仓口径一致）。"""
        account = self.broker.get_account()
        max_positions = max(1, int(self.settings.risk["max_positions"]))
        target_weight = min(
            1.0 / max_positions, float(self.settings.risk["max_single_position_weight"])
        )
        # 留出成交价波动、滑点与最低佣金的余量
        target_value = account.total_equity * target_weight * 0.995
        quantity = round_to_lot(target_value / price)
        if quantity <= 0:
            return ""
        return self.broker.submit_order(OrderRequest(
            code=pick["code"], name=pick["name"], side=OrderSide.BUY, quantity=quantity,
            strategy=pick["strategy"], trade_date=trade_date,
            requested_price=price, reason=pick["reason"],
        ))

    def _pick_buys(self, candidates: list[dict[str, Any]] | None, prices: dict[str, float],
                   positions: dict[str, Any], trade_date: str) -> list[dict[str, Any]]:
        """从买点扫描候选里挑出本次要买的标的（去重 + 名额约束 + 排序）。

        过滤规则（与既有「多策略并行」口径一致）：
        - 已在持仓中 → 不再加仓（同一标的只买一次）；
        - 当日已有买单（含未成交）→ 跳过，避免同一个交易日多个盘中时点反复买同一只；
        - 总持仓 ≤ ``risk.max_positions``、当日新开 ≤ ``risk.max_daily_new_positions``。
        排序：**命中策略数降序 → 当日成交额降序**（更确定、流动性更好的先买）。
        """
        if not candidates:
            return []
        hits: dict[str, dict[str, Any]] = {}
        for item in candidates:
            strategy = str(item.get("strategy") or "")
            for match in item.get("matches") or []:
                code = str(match.get("code") or "")
                if not code or code not in prices:
                    continue
                row = hits.setdefault(code, {
                    "code": code, "name": str(match.get("name") or code),
                    "amount": float(match.get("amount") or 0), "strategies": [],
                })
                if strategy and strategy not in row["strategies"]:
                    row["strategies"].append(strategy)
        if not hits:
            return []
        held = set(positions)
        ordered_today = {
            row["code"] for row in self.database.query_all(
                "SELECT DISTINCT code FROM orders WHERE side='BUY' AND trade_date=? "
                "AND status IN ('PENDING','FILLED')",
                (trade_date,),
            )
        }
        budget = min(
            max(0, int(self.settings.risk["max_positions"]) - len(held)),
            max(0, int(self.settings.risk["max_daily_new_positions"]) - len(ordered_today)),
        )
        if budget <= 0:
            LOG.info("盘中买入名额已用尽（持仓 %s 只 / 当日已下单 %s 只）", len(held), len(ordered_today))
            return []
        picks: list[dict[str, Any]] = []
        for row in sorted(hits.values(), key=lambda r: (-len(r["strategies"]), -r["amount"])):
            if row["code"] in held or row["code"] in ordered_today:
                continue
            row["strategy"] = row["strategies"][0] if row["strategies"] else ""
            row["reason"] = f"盘中扫描命中：{'、'.join(row['strategies']) or '自定义策略'}"
            picks.append(row)
            if len(picks) >= budget:
                break
        return picks

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

    def _mark_signal(self, signal: Signal, status: str) -> None:
        """把信号标记为本次实际执行结果，按**自然键**更新而不是按 id。

        ``generate()`` 用 ``INSERT OR IGNORE`` 落库，唯一键是
        ``(code, action, as_of_date, strategy)``：同一个交易日内多次评估产生的信号会带着
        **新的 uuid** 被忽略，此时按 id 更新会命中 0 行，看板里就会留下一条永远 `NEW` 的记录
        （而它其实已经成交或被拒）。按自然键更新可以覆盖两种情况。
        """
        self.database.execute(
            "UPDATE signals SET status=? WHERE code=? AND action=? AND as_of_date=? AND strategy=?",
            (status, signal.code, signal.action.value, normalize_date(signal.as_of_date), signal.strategy),
        )
