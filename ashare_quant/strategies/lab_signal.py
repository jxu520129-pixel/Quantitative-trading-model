"""因子实验室自定义策略适配器：把 signal_strategies 表里的条件策略接入模拟盘。

因子实验室的「策略定义」保存的策略（entry_json 为因子条件、exit_json 为离场条件）
原本只用于因子实验室内的回测与盘中买点扫描。本适配器把它们接入模拟盘信号生成：
对候选池逐股判断是否命中各已启用策略的买入条件，命中者按「条件因子的横截面
z-score 加权（方向由运算符决定）」打分，取 Top max_positions 生成调仓信号。
"""

from __future__ import annotations

import json

import pandas as pd

from ..lab import evaluate_conditions, evaluate_factor, resolve_factor_expressions
from .base import BaseStrategy, StrategyContext


class LabSignalStrategy(BaseStrategy):
    """运行 signal_strategies 表中所有已启用策略，合并打分生成调仓信号。"""

    name = "lab_signal"

    def __init__(self, parameters: dict[str, object]):
        super().__init__(parameters)
        self.database = parameters.get("_database")

    def generate(self, context: StrategyContext):
        if self.database is None:
            return []
        strategies = self.database.query_all(
            "SELECT name, entry_json FROM signal_strategies WHERE enabled=1 ORDER BY created_at DESC"
        )
        if not strategies:
            return []

        ranked: list[tuple[str, float, str]] = []
        for strat in strategies:
            try:
                entry = json.loads(strat["entry_json"])
            except (ValueError, TypeError):
                continue
            engine = str(entry.get("engine", "")).strip()
            if engine:
                # engine 类策略（涨停回马枪 / 人气热度）：内置引擎打分，单独处理
                self._rank_engine_signals(engine, entry, str(strat["name"]), ranked)
                continue
            conditions = entry.get("conditions", [])
            if not conditions:
                continue
            factor_names = [str(c["factor"]) for c in conditions]
            try:
                exprs = resolve_factor_expressions(self.database, factor_names)
            except ValueError:
                continue
            combine = str(entry.get("combine", "AND"))

            # 逐股计算因子值，记录命中股票
            latest: dict[str, dict[str, float]] = {fn: {} for fn in exprs}
            hit_codes: list[str] = []
            for code, frame in context.bars_by_code.items():
                try:
                    factor_values = {fn: evaluate_factor(expr, frame) for fn, expr in exprs.items()}
                    mask = evaluate_conditions(factor_values, conditions, combine)
                except (ValueError, KeyError):
                    continue
                for fn in exprs:
                    latest[fn][code] = float(factor_values[fn].iloc[-1])
                if bool(mask.iloc[-1]):
                    hit_codes.append(code)
            if not hit_codes:
                continue

            # 候选池内横截面 z-score（方向由运算符决定：>/>= 越高越好，</<= 越低越好）
            zscore: dict[str, pd.Series] = {}
            for fn in exprs:
                series = pd.Series(latest[fn], dtype=float)
                std = float(series.std())
                zscore[fn] = (series - series.mean()) / std if std and std > 0 else series * 0.0

            for code in hit_codes:
                score = 0.0
                for cond in conditions:
                    fn = str(cond["factor"])
                    op = str(cond["op"])
                    sign = 1.0 if op in (">", ">=") else (-1.0 if op in ("<", "<=") else 0.0)
                    score += sign * float(zscore[fn].get(code, 0.0))
                ranked.append((code, score, f"命中「{strat['name']}」"))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return self.rebalance_signals(context, ranked)

    def _rank_engine_signals(
        self, engine: str, entry: dict, name: str, ranked: list[tuple[str, float, str]]
    ) -> None:
        """处理 engine 类策略（内置引擎打分），把命中信号并入统一排序列表。

        涨停回马枪：复用 :func:`scan_limit_pullback_signals` 扫描最新交易日信号，
        命中（score ≥ threshold）的信号按 ``(score - threshold) / 10`` 归一化到
        与条件类 z-score 可比的量纲（约 0~3），正分越高越优先。
        """
        if engine == "limit_pullback_score":
            from ..limit_pullback_strategy import scan_limit_pullback_signals

            threshold = float(entry.get("threshold", 70.0))
            signals = scan_limit_pullback_signals(self.database)
            for sig in signals:
                if sig["score"] >= threshold:
                    strength = (sig["score"] - threshold) / 10.0
                    ranked.append((sig["code"], strength, f"命中「{name}」"))
        # engine == "hot_score"（人气热度）暂不接入模拟盘；如需接入再补充
