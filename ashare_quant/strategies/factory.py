"""Central strategy registry so new strategies have one integration point."""

from __future__ import annotations

from .base import BaseStrategy
from .factor_strategy import FACTOR_STRATEGIES
from .moving_average import DualMovingAverageStrategy
from .momentum import MomentumRotationStrategy


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "momentum_rotation": MomentumRotationStrategy,
    "dual_moving_average": DualMovingAverageStrategy,
    **FACTOR_STRATEGIES,
}


def build_strategy(name: str, parameters: dict[str, object]) -> BaseStrategy:
    try:
        return STRATEGIES[name](parameters)
    except KeyError as error:
        raise ValueError(f"未知策略：{name}。可选策略：{', '.join(STRATEGIES)}") from error
