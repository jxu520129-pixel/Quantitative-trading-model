"""Weekly cross-sectional momentum rotation."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, StrategyContext, usable_bars


class MomentumRotationStrategy(BaseStrategy):
    name = "momentum_rotation"

    def generate(self, context: StrategyContext):
        lookback = int(self.parameters.get("lookback_days", 60))
        min_amount = float(self.parameters.get("min_average_amount", 20_000_000))
        ranked: list[tuple[str, float, str]] = []
        for row in context.universe.itertuples(index=False):
            data = usable_bars(context.bars_by_code.get(row.code, pd.DataFrame()), context.as_of_date, lookback + 1)
            if data is None:
                continue
            momentum = data["close"].iloc[-1] / data["close"].iloc[-lookback - 1] - 1
            average_amount = float(data["amount"].tail(20).mean())
            if average_amount < min_amount:
                continue
            ranked.append((row.code, float(momentum), f"{lookback} 日动量 {momentum:.2%}，近 20 日平均成交额 {average_amount:,.0f} 元"))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return self.rebalance_signals(context, ranked)
