"""中文展示层：内部标识保持稳定，所有面向用户的文本在此本地化。"""

from __future__ import annotations

import dataclasses
import re
from enum import Enum
from typing import Any

import pandas as pd


STRATEGY_LABELS = {
    "momentum_rotation": "周度动量轮动",
    "dual_moving_average": "双均线趋势",
    "simple_multifactor": "简易多因子",
    "reversal": "短期反转",
    "low_volatility": "低波动",
    "trend": "趋势跟踪",
    "liquidity": "流动性",
    "rsi_mean_reversion": "RSI 超卖反转",
    "enhanced_multifactor": "增强多因子",
    "manual": "手动操作",
}

VALUE_LABELS = {
    "PAPER": "模拟盘",
    "LIVE": "实盘",
    "BUY": "买入",
    "SELL": "卖出",
    "HOLD": "持有",
    "NEW": "新信号",
    "QUEUED": "已排队",
    "PENDING": "待成交",
    "FILLED": "已成交",
    "REJECTED": "已拒绝",
    "CANCELLED": "已撤销",
    "FAILED": "执行失败",
    "SKIPPED": "已跳过",
    "DEFERRED": "已延后",
    "INFO": "提示",
    "WARNING": "警告",
    "ERROR": "错误",
    "DAILY_LOSS": "日亏损限制",
    "EXECUTION_FAILURE": "执行失败",
    "EXECUTION_PAUSED": "执行暂停",
    "initialized": "已初始化",
    "synthetic-demo": "内置演示行情",
    "paper": "模拟账户",
}

COLUMN_LABELS = {
    "id": "编号",
    "run_id": "运行编号",
    "account_id": "账户标识",
    "mode": "运行模式",
    "status": "状态",
    "source": "数据来源",
    "database": "数据库文件",
    "bars": "日线数量",
    "universe": "证券池数量",
    "updated_symbols": "更新标的数",
    "signals": "信号数量",
    "queued": "已排队数量",
    "skipped": "跳过数量",
    "executed": "执行结果",
    "filled": "成交数量",
    "rejected": "拒绝数量",
    "failed": "失败数量",
    "account": "账户",
    "positions": "持仓",
    "risk": "风控",
    "trade_date": "交易日期",
    "as_of_date": "信号日期",
    "snapshot_date": "净值日期",
    "filled_at": "成交时间",
    "created_at": "创建时间",
    "updated_at": "更新时间",
    "event_time": "事件时间",
    "code": "证券代码",
    "name": "证券名称",
    "side": "方向",
    "action": "操作",
    "quantity": "数量",
    "filled_quantity": "成交数量",
    "sellable_quantity": "可卖数量",
    "requested_price": "委托价格",
    "fill_price": "成交价格",
    "price": "成交价格",
    "avg_cost": "持仓成本",
    "latest_price": "最新价格",
    "market_value": "持仓市值",
    "unrealized_pnl": "浮动盈亏",
    "gross_amount": "成交金额",
    "commission": "佣金",
    "stamp_duty": "印花税",
    "cash": "可用现金",
    "total_equity": "总资产",
    "equity": "资产净值",
    "daily_pnl": "当日盈亏",
    "strategy": "策略",
    "target_weight": "目标仓位",
    "score": "评分",
    "reason": "信号依据",
    "error": "异常信息",
    "level": "级别",
    "category": "风险类别",
    "failure_count": "连续失败次数",
    "paused": "交易暂停",
    "daily_open_blocked": "停止开仓",
    "last_reset_date": "风控重置日期",
    "initial_cash": "初始资金",
    "final_equity": "期末资产",
    "total_return": "累计收益",
    "annual_return": "年化收益",
    "benchmark": "基准代码",
    "benchmark_return": "基准收益",
    "benchmark_annual_return": "基准年化收益",
    "excess_annual_return": "超额年化收益",
    "max_drawdown": "最大回撤",
    "sharpe": "夏普比率",
    "win_rate": "胜率",
    "total_trades": "总交易次数",
    "equity_curve": "资金曲线",
    "expression": "因子公式",
    "strategy_name": "策略名称",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "entry_date": "买入日期",
    "entry_price": "买入价格",
    "exit_date": "卖出日期",
    "exit_price": "卖出价格",
    "shares": "股数",
    "pnl": "盈亏金额",
    "pnl_pct": "盈亏率",
    "holding_days": "持仓天数",
    "entry_reason": "买入依据",
    "exit_reason": "离场依据",
    "total_pnl": "总盈亏",
    "profit_factor": "盈亏比",
    "avg_pnl": "平均每笔盈亏",
    "avg_holding_days": "平均持仓天数",
    "metrics_json": "回测指标",
}

_BOOLEAN_COLUMNS = {"paused", "daily_open_blocked"}
_MESSAGE_LABELS = {
    "Trading is paused after consecutive execution failures": "连续执行失败，交易已暂停",
    "Insufficient sellable shares under A-share T+1 rule": "可卖数量不足，不符合 A 股 T+1 规则",
    "New positions are stopped by daily loss limit": "已触发日亏损限制，停止新开仓",
    "Account equity is invalid": "账户总资产异常",
    "Insufficient available cash": "可用现金不足",
    "Maximum holdings reached": "已达到最大持仓数量",
    "Maximum daily new positions reached": "已达到单日最大新开仓数量",
    "Paper account is not initialized": "模拟账户尚未初始化",
    "No market bar available": "没有可用于成交的行情数据",
    "Order blocked at daily price limit": "委托被涨跌停限制拦截",
    "Insufficient cash at execution price": "按成交价格计算后可用现金不足",
    "No sellable quantity under T+1": "当日无可卖数量，不符合 T+1 规则",
    "Sell quantity is below one lot": "卖出数量不足一手",
    "No daily bars are available for signal generation": "没有可用于生成信号的日线数据",
}


def label_strategy(value: object) -> str:
    """返回策略的中文名称，未知策略仍展示原标识便于排障。

    ``lab:<策略名>`` 表示因子实验室里的单个自定义策略，直接展示策略名本身。
    """
    text = str(value)
    if text.startswith("lab:") and text[len("lab:"):].strip():
        return text[len("lab:"):].strip()
    return STRATEGY_LABELS.get(text, text)


def label_value(value: object, field: str | None = None) -> object:
    """按字段将内部枚举或布尔值转换成中文展示文本。"""
    if isinstance(value, Enum):
        value = value.value
    if field == "strategy":
        return label_strategy(value)
    if field in _BOOLEAN_COLUMNS and isinstance(value, (bool, int)):
        return "是" if bool(value) else "否"
    if isinstance(value, str):
        return VALUE_LABELS.get(value, value)
    return value


def label_message(value: object) -> object:
    """翻译新旧记录中的固定异常与策略依据文本。"""
    if not isinstance(value, str):
        return value
    if value in _MESSAGE_LABELS:
        return _MESSAGE_LABELS[value]
    if value == "No longer in selected portfolio":
        return "不再属于目标组合"
    if match := re.fullmatch(r"(\d+)d momentum=([-+]?\d+(?:\.\d+)?)%, avg_amount=([\d,]+)", value):
        return f"{match.group(1)} 日动量 {match.group(2)}%，近 20 日平均成交额 {match.group(3)} 元"
    if match := re.fullmatch(r"MA(\d+)=([\d.]+) above MA(\d+)=([\d.]+)", value):
        return f"MA{match.group(1)}={match.group(2)} 高于 MA{match.group(3)}={match.group(4)}"
    if match := re.fullmatch(r"factor score=([-+]?\d+(?:\.\d+)?); momentum=([-+]?\d+(?:\.\d+)?)%", value):
        return f"因子评分 {match.group(1)}；动量 {match.group(2)}%"
    if match := re.fullmatch(r"Daily loss ([-+]?\d+(?:\.\d+)?)%; new positions are blocked", value):
        return f"当日亏损 {match.group(1)}%，已停止新开仓"
    return value


def localize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """复制数据表并翻译列名与可见枚举值，不修改原始数据。"""
    result = frame.copy()
    for column in result.columns:
        if column in {"reason", "error", "message"}:
            result[column] = result[column].map(label_message)
        elif column in {"strategy", "side", "action", "status", "level", "category", "mode", "source"} | _BOOLEAN_COLUMNS:
            result[column] = result[column].map(lambda value: label_value(value, column))
    return result.rename(columns=COLUMN_LABELS)


def localize_payload(value: Any, field: str | None = None) -> Any:
    """递归转换 CLI JSON 的键名、状态值与数据类对象。"""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        return label_value(value.value, field)
    if isinstance(value, dict):
        return {
            COLUMN_LABELS.get(str(key), str(key)): localize_payload(item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [localize_payload(item, field) for item in value]
    if field in {"reason", "error", "message"}:
        return label_message(value)
    return label_value(value, field)
