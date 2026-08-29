"""Strategy interface and standard rebalance signal construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..models import Signal, SignalAction


@dataclass(frozen=True)
class StrategyContext:
    as_of_date: str
    universe: pd.DataFrame
    bars_by_code: dict[str, pd.DataFrame]
    held_codes: set[str]
    max_positions: int


class BaseStrategy(ABC):
    name: str

    def __init__(self, parameters: dict[str, object]):
        self.parameters = parameters

    @abstractmethod
    def generate(self, context: StrategyContext) -> list[Signal]:
        """Return actionable standardized signals for the next trading session."""

    def rebalance_signals(self, context: StrategyContext, ranked: list[tuple[str, float, str]]) -> list[Signal]:
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
    if frame.empty:
        return None
    data = frame[frame["trade_date"].astype(str).str[:10] <= as_of_date].copy()
    if len(data) < minimum:
        return None
    return data.sort_index()
