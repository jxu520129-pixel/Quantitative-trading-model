"""Strategy interface and standard rebalance signal construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..models import Signal, SignalAction


@dataclass(frozen=True)
class StrategyContext:
    """策略生成信号所需的上下文：交易日、候选池、各标的日线、当前持仓与持仓上限。

    ``positions`` 是 code -> 持仓行（含 ``strategy`` 归属、``avg_cost``、``trail_peak``、
    ``entry_breakout``），供自定义策略判断「哪些持仓归我管」以及各自的止损止盈基准。
    """

    as_of_date: str
    universe: pd.DataFrame
    bars_by_code: dict[str, pd.DataFrame]
    held_codes: set[str]
    max_positions: int
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略基类：约定 ``generate`` 接口，并提供标准的调仓信号构造逻辑。"""

    name: str

    def __init__(self, parameters: dict[str, object]):
        self.parameters = parameters

    @abstractmethod
    def generate(self, context: StrategyContext) -> list[Signal]:
        """Return actionable standardized signals for the next trading session."""

    def rebalance_signals(self, context: StrategyContext, ranked: list[tuple[str, float, str]]) -> list[Signal]:
        """把按评分降序的 ``(code, score, reason)`` 列表转换为调仓信号。

        不在目标组合内的持仓生成卖出信号，不在持仓中的选中标的生成买入信号，
        目标权重按持仓上限等分。
        """
        selected = ranked[: context.max_positions]
        selected_codes = {code for code, _score, _reason in selected}
        names = dict(zip(context.universe["code"], context.universe["name"], strict=False))
        target_weight = 1.0 / context.max_positions
        signals: list[Signal] = []
        for code in sorted(context.held_codes - selected_codes):
            signals.append(Signal(
                code=code, name=names.get(code, code), action=SignalAction.SELL, as_of_date=context.as_of_date,
                strategy=self.name, reason="不再属于目标组合",
            ))
        for code, score, reason in selected:
            if code not in context.held_codes:
                signals.append(Signal(
                    code=code, name=names.get(code, code), action=SignalAction.BUY, as_of_date=context.as_of_date,
                    strategy=self.name, target_weight=target_weight, score=round(score, 6), reason=reason,
                ))
        return signals


def usable_bars(frame: pd.DataFrame, as_of_date: str, minimum: int) -> pd.DataFrame | None:
    """截取截至 ``as_of_date`` 的日线并校验最少样本数，不足则返回 None（该标的不可用）。"""
    if frame.empty:
        return None
    data = frame[frame["trade_date"].astype(str).str[:10] <= as_of_date].copy()
    if len(data) < minimum:
        return None
    return data.sort_index()
