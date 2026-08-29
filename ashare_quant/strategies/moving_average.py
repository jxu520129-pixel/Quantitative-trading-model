"""Trend-following dual moving-average selection strategy."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, StrategyContext, usable_bars


class DualMovingAverageStrategy(BaseStrategy):
    name = "dual_moving_average"

    def generate(self, context: StrategyContext):
        fast = int(self.parameters.get("fast_window", 10))
        slow = int(self.parameters.get("slow_window", 30))
        minimum = max(fast, slow)
        ranked: list[tuple[str, float, str]] = []
        for row in context.universe.itertuples(index=False):
            data = usable_bars(context.bars_by_code.get(row.code, pd.DataFrame()), context.as_of_date, minimum)
            if data is None:
                continue
            fast_ma = float(data["close"].tail(fast).mean())
            slow_ma = float(data["close"].tail(slow).mean())
            if fast_ma <= slow_ma:
                continue
            score = fast_ma / slow_ma - 1
            ranked.append((row.code, score, f"MA{fast}={fast_ma:.2f} 高于 MA{slow}={slow_ma:.2f}"))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return self.rebalance_signals(context, ranked)
