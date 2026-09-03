"""Central strategy registry so new strategies have one integration point."""

from __future__ import annotations

from .base import BaseStrategy
from .factor_strategy import FACTOR_STRATEGIES
from .lab_signal import LabSignalStrategy
from .moving_average import DualMovingAverageStrategy
from .momentum import MomentumRotationStrategy


STRATEGIES: dict[str, type[BaseStrategy]] = {
    "momentum_rotation": MomentumRotationStrategy,
    "dual_moving_average": DualMovingAverageStrategy,
    "lab_signal": LabSignalStrategy,
    **FACTOR_STRATEGIES,
}


def build_strategy(name: str, parameters: dict[str, object]) -> BaseStrategy:
    """按策略名从注册表实例化策略，未知名称抛出含可用选项的 ValueError。"""
    try:
        return STRATEGIES[name](parameters)
    except KeyError as error:
        raise ValueError(f"未知策略：{name}。可选策略：{', '.join(STRATEGIES)}") from error
