"""单个因子实验室自定义策略适配器：一个策略 = ``signal_strategies`` 表里的一条记录。

与 :class:`LabSignalStrategy`（把表内所有已启用策略**合并打分**）不同，本适配器只运行
**指定的一条**策略，并同时产出买入与卖出信号：

- **买入**：候选池内命中该策略 ``entry`` 条件、且当前未被持有的标的，
  按「条件因子横截面 z-score 加权」排序（方向由运算符决定）。
- **卖出**：当前持仓中**归属本策略**（``positions.strategy == 策略名``）的标的，
  按其 ``exit`` 配置评估离场 —— 条件离场 / 固定止损 / 固定止盈 / 移动止损 / 跌破箱体，
  与回测共用 :func:`ashare_quant.lab.evaluate_exit`，保证「回测怎么写、模拟盘就怎么执行」。

这样每个策略各自独立买卖、独立止损止盈，多策略并行时不会互相错乱。
"""

from __future__ import annotations

import json

import pandas as pd

from ..lab import evaluate_conditions, evaluate_exit, evaluate_factor, resolve_factor_expressions
from ..models import Signal, SignalAction
from .base import BaseStrategy, StrategyContext


class LabCustomStrategy(BaseStrategy):
    """运行单条自定义策略，独立生成买卖信号。"""

    name = "lab_custom"

    def __init__(self, parameters: dict[str, object]):
        super().__init__(parameters)
        self.database = parameters.get("_database")
        self.lab_name = str(parameters.get("_lab_name") or "")

    # ------------------------------------------------------------------ 入口
    def generate(self, context: StrategyContext) -> list[Signal]:
        if self.database is None or not self.lab_name:
            return []
        row = self.database.query_one(
            "SELECT name, entry_json, exit_json FROM signal_strategies WHERE name=? AND enabled=1",
            (self.lab_name,),
        )
        if not row:
            return []
        try:
            entry = json.loads(row["entry_json"])
            exit_config = json.loads(row["exit_json"]) if row["exit_json"] else {}
        except (ValueError, TypeError):
            return []

        signals = self._exit_signals(context, exit_config)
        signals.extend(self._entry_signals(context, entry))
        return signals

    # -------------------------------------------------------------- 卖出信号
    def _exit_signals(self, context: StrategyContext, exit_config: dict) -> list[Signal]:
        """只对归属本策略的持仓评估离场，命中则生成 SELL。"""
        conditions = exit_config.get("conditions") or []
        combine = str(exit_config.get("combine", "OR"))
        expressions: dict[str, str] = {}
        if conditions:
            try:
                expressions = resolve_factor_expressions(
                    self.database, [str(c["factor"]) for c in conditions]
                )
            except ValueError:
                expressions = {}

        signals: list[Signal] = []
        for code, position in context.positions.items():
            if int(position.get("quantity") or 0) <= 0:
                continue
            # 多策略并行时各管各的：只处理归属本策略的持仓，避免策略间互相平仓
            if str(position.get("strategy") or "") != self.lab_name:
                continue
            frame = context.bars_by_code.get(code)
            if frame is None or frame.empty:
                continue
            close = float(frame["close"].iloc[-1])
            factor_exit = False
            if conditions and expressions:
                try:
                    values = {fn: evaluate_factor(expr, frame) for fn, expr in expressions.items()}
                    mask = evaluate_conditions(values, conditions, combine)
                    factor_exit = bool(mask.iloc[-1])
                except (ValueError, KeyError):
                    factor_exit = False
            reason = evaluate_exit(
                exit_config,
                entry_price=float(position.get("avg_cost") or 0),
                close=close,
                trail_peak=float(position.get("trail_peak") or 0),
                entry_breakout=float(position.get("entry_breakout") or 0),
                factor_exit=factor_exit,
            )
            if reason:
                signals.append(Signal(
                    code=code, name=str(position.get("name") or code), action=SignalAction.SELL,
                    as_of_date=context.as_of_date, strategy=self.lab_name,
                    reason=f"{reason}",
                ))
        return signals

    # -------------------------------------------------------------- 买入信号
    def _entry_signals(self, context: StrategyContext, entry: dict) -> list[Signal]:
        """产出本策略的买入候选（不做名额裁剪，由信号服务跨策略统一排序取 Top）。"""
        engine = str(entry.get("engine", "")).strip()
        if engine:
            return self._engine_signals(context, engine, entry)
        conditions = entry.get("conditions") or []
        if not conditions:
            return []
        try:
            expressions = resolve_factor_expressions(
                self.database, [str(c["factor"]) for c in conditions]
            )
        except ValueError:
            return []
        combine = str(entry.get("combine", "AND"))

        latest: dict[str, dict[str, float]] = {fn: {} for fn in expressions}
        hit_codes: list[str] = []
        for code, frame in context.bars_by_code.items():
            if code in context.held_codes:
                continue  # 同一标的只买一次（已被任一策略持有即不重复建仓）
            try:
                values = {fn: evaluate_factor(expr, frame) for fn, expr in expressions.items()}
                mask = evaluate_conditions(values, conditions, combine)
            except (ValueError, KeyError):
                continue
            for fn in expressions:
                latest[fn][code] = float(values[fn].iloc[-1])
            if bool(mask.iloc[-1]):
                hit_codes.append(code)
        if not hit_codes:
            return []

        # 候选池内横截面 z-score（方向由运算符决定：>/>= 越高越好，</<= 越低越好）
        zscore: dict[str, pd.Series] = {}
        for fn in expressions:
            series = pd.Series(latest[fn], dtype=float)
            std = float(series.std())
            zscore[fn] = (series - series.mean()) / std if std and std > 0 else series * 0.0

        scored: list[tuple[str, float]] = []
        for code in hit_codes:
            score = 0.0
            for condition in conditions:
                fn, op = str(condition["factor"]), str(condition["op"])
                sign = 1.0 if op in (">", ">=") else (-1.0 if op in ("<", "<=") else 0.0)
                score += sign * float(zscore[fn].get(code, 0.0))
            scored.append((code, score))
        scored.sort(key=lambda item: item[1], reverse=True)

        names = dict(zip(context.universe["code"], context.universe["name"], strict=False))
        target_weight = 1.0 / context.max_positions if context.max_positions else 0.0
        return [
            Signal(
                code=code, name=names.get(code, code), action=SignalAction.BUY,
                as_of_date=context.as_of_date, strategy=self.lab_name,
                target_weight=target_weight, score=round(score, 6),
                reason=f"命中「{self.lab_name}」",
            )
            for code, score in scored
        ]

    # ------------------------------------------------------- engine 类策略
    def _engine_signals(self, context: StrategyContext, engine: str, entry: dict) -> list[Signal]:
        """内置引擎类策略（涨停回马枪）的买入信号。"""
        if engine != "limit_pullback_score":
            return []
        from ..limit_pullback_strategy import scan_limit_pullback_signals

        threshold = float(entry.get("threshold", 70.0))
        names = dict(zip(context.universe["code"], context.universe["name"], strict=False))
        target_weight = 1.0 / context.max_positions if context.max_positions else 0.0
        signals: list[Signal] = []
        for hit in scan_limit_pullback_signals(self.database):
            code = hit["code"]
            if float(hit["score"]) < threshold or code in context.held_codes:
                continue
            signals.append(Signal(
                code=code, name=names.get(code, hit.get("name") or code), action=SignalAction.BUY,
                as_of_date=context.as_of_date, strategy=self.lab_name,
                target_weight=target_weight, score=round(float(hit["score"]), 6),
                reason=f"命中「{self.lab_name}」",
            ))
        signals.sort(key=lambda item: item.score, reverse=True)
        return signals
