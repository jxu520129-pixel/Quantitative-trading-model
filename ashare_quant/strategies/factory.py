"""Central strategy registry so new strategies have one integration point."""

from __future__ import annotations

from .base import BaseStrategy
from .factor_strategy import FACTOR_STRATEGIES
from .lab_custom import LabCustomStrategy
from .lab_signal import LabSignalStrategy
from .moving_average import DualMovingAverageStrategy
from .momentum import MomentumRotationStrategy


# 因子实验室自定义策略的标识前缀：``lab:<策略名>`` 表示运行**单个**自定义策略。
LAB_PREFIX = "lab:"

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "momentum_rotation": MomentumRotationStrategy,
    "dual_moving_average": DualMovingAverageStrategy,
    "lab_signal": LabSignalStrategy,
    **FACTOR_STRATEGIES,
}


def lab_strategy_key(lab_name: str) -> str:
    """把自定义策略名转换为策略标识（看板/CLI 统一使用）。"""
    return f"{LAB_PREFIX}{lab_name}"


def is_lab_strategy(name: str) -> bool:
    """判断策略标识是否为「单个自定义策略」。"""
    return str(name).startswith(LAB_PREFIX) and bool(str(name)[len(LAB_PREFIX):].strip())


def build_strategy(name: str, parameters: dict[str, object]) -> BaseStrategy:
    """按策略名从注册表实例化策略，未知名称抛出含可用选项的 ValueError。

    ``lab:<策略名>`` 形式表示运行因子实验室里的**单个**自定义策略——该策略独立生成
    买卖信号、独立执行止损止盈；与 ``lab_signal``（把表内全部已启用策略合并打分）不同。
    """
    if is_lab_strategy(name):
        lab_name = str(name)[len(LAB_PREFIX):].strip()
        return LabCustomStrategy({**parameters, "_lab_name": lab_name})
    try:
        return STRATEGIES[name](parameters)
    except KeyError as error:
        raise ValueError(f"未知策略：{name}。可选策略：{', '.join(STRATEGIES)}") from error
