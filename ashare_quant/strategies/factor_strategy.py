"""主流量化因子策略：基于共享因子库的横截面打分与调仓。"""

from __future__ import annotations

from .base import BaseStrategy
from .factors import cross_sectional_rank_score


class FactorStrategy(BaseStrategy):
    """通用因子策略：对候选池做带符号权重合成评分并生成调仓信号。"""

    name = "factor"
    factors: tuple[tuple[str, float], ...] = ()

    def generate(self, context):
        ranked = cross_sectional_rank_score(context.bars_by_code, self.factors, self.parameters)
        return self.rebalance_signals(context, ranked)


class ReversalStrategy(FactorStrategy):
    name = "reversal"
    factors = (("reversal", 1.0),)


class LowVolatilityStrategy(FactorStrategy):
    name = "low_volatility"
    factors = (("volatility", -1.0), ("atr", -0.5))


class TrendStrategy(FactorStrategy):
    name = "trend"
    factors = (("high_52w", 1.0), ("ma_trend", 0.6), ("macd", 0.4))


class LiquidityStrategy(FactorStrategy):
    name = "liquidity"
    factors = (("liquidity", 1.0), ("amihud", -0.5))


class RsiMeanReversionStrategy(FactorStrategy):
    name = "rsi_mean_reversion"
    factors = (("rsi", -1.0),)


class EnhancedMultiFactorStrategy(FactorStrategy):
    name = "enhanced_multifactor"
    factors = (
        ("momentum", 0.5), ("reversal", 0.2), ("volatility", -0.4),
        ("liquidity", 0.25), ("high_52w", 0.3),
    )


class SimpleMultiFactorStrategy(FactorStrategy):
    """简易多因子：动量、低波动、流动性。回测与实盘共用同一套横截面打分。"""

    name = "simple_multifactor"
    factors = (("momentum", 1.0), ("volatility", -1.0), ("liquidity", 0.25))


FACTOR_STRATEGIES = {
    ReversalStrategy.name: ReversalStrategy,
    LowVolatilityStrategy.name: LowVolatilityStrategy,
    TrendStrategy.name: TrendStrategy,
    LiquidityStrategy.name: LiquidityStrategy,
    RsiMeanReversionStrategy.name: RsiMeanReversionStrategy,
    EnhancedMultiFactorStrategy.name: EnhancedMultiFactorStrategy,
    SimpleMultiFactorStrategy.name: SimpleMultiFactorStrategy,
}

FACTOR_STRATEGY_SPECS = {name: cls.factors for name, cls in FACTOR_STRATEGIES.items()}
