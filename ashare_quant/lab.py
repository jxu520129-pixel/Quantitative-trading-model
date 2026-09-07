"""因子实验室：公式因子求值、买卖条件、信号回测、逐笔交易记录与分析。

与 Backtrader 的周度调仓回测不同，这里做的是**逐日信号回测**：对候选池内每只标的独立判断，
第 T 日收盘后的因子信号在第 T+1 日开盘成交；符合入场条件建仓、符合离场条件平仓。
"""

from __future__ import annotations

import ast
import html as _html
import json
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from .hot_strategy import HOT_STRATEGY_NAME
from .limit_pullback_strategy import LIMIT_STRATEGY_NAME
from .market_rules import TradingCosts, round_to_lot


LOG = logging.getLogger(__name__)


# 内置示例因子：名称 -> (中文说明, pandas 公式表达式)
BUILTIN_FACTOR_FORMULAS: dict[str, tuple[str, str]] = {
    "momentum_60": ("60日动量", "close/close.shift(60)-1"),
    "reversal_5": ("5日反转", "-(close/close.shift(5)-1)"),
    "volatility_20": ("20日波动率", "close.pct_change().rolling(20).std()"),
    "liquidity_20": ("20日成交额", "amount.rolling(20).mean()"),
    "rsi_14": (
        "RSI(14)",
        "100-100/(1+close.diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean()"
        "/close.diff().clip(upper=0).abs().ewm(alpha=1/14,adjust=False).mean())",
    ),
    "macd_hist": (
        "MACD柱",
        "(close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean())"
        "-(close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean())"
        ".ewm(span=9,adjust=False).mean()",
    ),
    "ma_trend": (
        "均线多头度",
        "(close.rolling(5).mean()>close.rolling(10).mean()).astype(float)"
        "+(close.rolling(10).mean()>close.rolling(20).mean()).astype(float)"
        "+(close.rolling(20).mean()>close.rolling(60).mean()).astype(float)",
    ),
    "high_52w": ("52周新高接近度", "close/close.rolling(250,min_periods=1).max()-1"),
    "atr_14": ("ATR近似", "(high-low).rolling(14).mean()/close"),
    "amihud_20": ("Amihud非流动性", "(close.pct_change().abs()/amount).rolling(20).mean()"),
    # 箱体突破与趋势确认因子（配合箱体突破策略，减少出手、提高胜率）
    "ma20_dev": ("20日线偏离", "close/close.rolling(20).mean()-1"),
    "ma60_dev": ("60日线偏离", "close/close.rolling(60).mean()-1"),
    "box_range_4": ("4日箱体振幅", "(high.rolling(4).max()-low.rolling(4).min())/close"),
    "box_range_5": ("5日箱体振幅", "(high.rolling(5).max()-low.rolling(5).min())/close"),
    "breakout_4": ("突破前4日高点", "close/high.shift(1).rolling(4).max()-1"),
    "breakout_5": ("突破前5日高点", "close/high.shift(1).rolling(5).max()-1"),
    "lower_high": ("未破前高", "close/high.shift(1).rolling(20).max()-1"),
    "rally_5": ("5日涨幅", "close/close.shift(5)-1"),
    "rally_10": ("10日涨幅", "close/close.shift(10)-1"),
    "pullback_1": ("当日涨跌", "close/close.shift(1)-1"),
    "vol_ratio": ("量比", "volume/volume.shift(1).rolling(5).mean()"),
    "near_low_20": ("距20日低点", "close/low.rolling(20).min()-1"),
    "up_days_5": ("5日上涨比例", "(close.diff()>0).rolling(5).mean()"),
    "v_recovery": ("深V回收", "close/low.rolling(10).min()-1"),
    # 涨跌幅/竞价/形态因子
    "rally_3": ("三日涨跌幅", "close/close.shift(3)-1"),
    "gap_up": ("集合竞价高开", "open/pre_close-1"),
    "morning_star": ("早晨之星近似", "(close-open)/close-(close.shift(1)-open.shift(1))/close.shift(1)"),
    "breakout_20": ("突破前20日高点", "close/high.shift(1).rolling(20).max()-1"),
    "breakout_60": ("突破前60日高点", "close/high.shift(1).rolling(60).max()-1"),
}


# 内置策略模板：名称 -> (说明, entry_json, exit_json)。可在看板「因子实验室」一键导入。
# 止损/止盈等比例均为小数（0.08 = 8%），与看板保存策略口径一致。
BUILTIN_STRATEGY_TEMPLATES: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    # 名称与 hot_strategy.HOT_STRATEGY_NAME 保持一致；entry_json 带 engine=hot_score，
    # 因子条件扫描会自动跳过，回测与运行由 ③/打分引擎按引擎标记分派。
    "箱体突破·放量确认": (
        "收盘突破前4日高点 + 量比>2 + 站上20日线 + 箱体振幅<15%；移动止损12%/止盈30%（全A股回测年化18.5%）",
        {"combine": "AND", "conditions": [
            {"factor": "breakout_4", "op": ">", "value": 0.005},
            {"factor": "vol_ratio", "op": ">", "value": 2.0},
            {"factor": "ma20_dev", "op": ">", "value": 0.0},
            {"factor": "box_range_5", "op": "<", "value": 0.15},
        ]},
        {"combine": "OR", "conditions": [],
         "stop_loss_pct": 0.0, "take_profit_pct": 0.30, "trailing_stop_pct": 0.12, "breakout_exit": False},
    ),
    "超跌反弹·量比确认": (
        "超跌反弹 + 量比>1.2（放量确认，回撤更低至约13%）；止损6%/止盈15%",
        {"combine": "AND", "conditions": [
            {"factor": "near_low_20", "op": "<", "value": 0.15},
            {"factor": "up_days_5", "op": ">", "value": 0.6},
            {"factor": "ma20_dev", "op": ">", "value": -0.02},
            {"factor": "vol_ratio", "op": ">", "value": 1.2},
        ]},
        {"combine": "OR", "conditions": [],
         "stop_loss_pct": 0.06, "take_profit_pct": 0.15, "trailing_stop_pct": 0.0, "breakout_exit": False},
    ),
    "主升浪·启动突破": (
        "突破前20日高点 + 放量 + 站上20日线 + 均线多头（近似「启动→主升」阶段）；移动止损15%/止盈50%",
        {"combine": "AND", "conditions": [
            {"factor": "breakout_20", "op": ">", "value": 0.003},
            {"factor": "vol_ratio", "op": ">", "value": 1.5},
            {"factor": "ma20_dev", "op": ">", "value": 0.0},
            {"factor": "ma_trend", "op": ">=", "value": 0.5},
        ]},
        {"combine": "OR", "conditions": [],
         "stop_loss_pct": 0.0, "take_profit_pct": 0.50, "trailing_stop_pct": 0.15, "breakout_exit": False},
    ),
    HOT_STRATEGY_NAME: (
        "人气×涨停×题材×资金四维共振打分（横截面模型）：题材≥次主线且量比≥1.5才出手，接力需次日高开2%~6%，"
        "仅情绪活跃期（涨停≥50家且连板高度≥3）参与，冰点空仓、每周≤3笔；-6%止损/3日时间止损/浮盈回撤保护（日线近似回测）",
        {"engine": "hot_score", "threshold": 75, "sentiment_scope": "main", "conditions": [],
         "description": "四维共振打分模型，引擎在 ashare_quant/hot_strategy.py"},
        {"combine": "OR", "conditions": []},
    ),
    LIMIT_STRATEGY_NAME: (
        "涨停回马枪·冲高回调低吸（形态模型）：收盘封死涨停(首板~2板，剔一字) → 次日冲高≥5%创20日新高 → "
        "缩量回调5%~15%不破涨停日最低/MA20 → 缩量十字星+放量阳线收复5日线双确认 → 尾盘低吸；"
        "前高减半+移动止盈+10日强平，大盘闸门（上证≥MA20/涨停≥50家/跌停≤10家/大跌熔断）；仅沪深主板",
        {"engine": "limit_pullback_score", "threshold": 70, "conditions": [],
         "description": "涨停回马枪形态引擎，代码在 ashare_quant/limit_pullback_strategy.py"},
        {"combine": "OR", "conditions": []},
    ),
}

OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _to_pct(value: Any) -> float | None:
    """把止损/止盈配置解析为正百分比小数；为空或 <=0 视为不启用。"""
    if value is None:
        return None
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return None
    return pct if pct > 0 else None


_FACTOR_COLUMNS = {"open", "high", "low", "close", "volume", "amount", "pre_close"}
_FACTOR_FUNCTIONS = {"abs", "min", "max", "round", "sum", "float", "int", "bool"}
_FACTOR_METHODS = {"shift", "pct_change", "rolling", "std", "mean", "diff", "clip", "ewm", "min", "max", "abs", "astype"}
_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_UNARY_OPS = (ast.UAdd, ast.USub)
_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _validate_factor_node(node: ast.AST) -> None:
    """白名单校验因子公式 AST，仅允许列名、数字字面量、有限方法与运算符。

    拒绝任意属性访问、函数调用与下标，杜绝 ``__import__``/``__class__`` 逃逸。
    """
    if isinstance(node, ast.Expression):
        _validate_factor_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return
        raise ValueError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _FACTOR_COLUMNS or node.id in _FACTOR_FUNCTIONS:
            return
        raise ValueError(f"未知名称：{node.id}")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        _validate_factor_node(node.left)
        _validate_factor_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        _validate_factor_node(node.operand)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE_OPS:
            raise ValueError("因子公式仅支持单一比较运算符")
        _validate_factor_node(node.left)
        for comp in node.comparators:
            _validate_factor_node(comp)
        return
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _FACTOR_FUNCTIONS:
                raise ValueError(f"未知函数：{node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in _FACTOR_METHODS:
                raise ValueError(f"不支持的方法：{node.func.attr}")
            _validate_factor_node(node.func.value)
        else:
            raise ValueError("不支持调用该表达式")
        for arg in node.args:
            _validate_factor_node(arg)
        for kw in node.keywords:
            _validate_factor_node(kw.value)
        return
    if isinstance(node, ast.Attribute):
        if node.attr not in _FACTOR_METHODS:
            raise ValueError(f"不支持的方法：{node.attr}")
        _validate_factor_node(node.value)
        return
    raise ValueError(f"不支持的表达式：{type(node).__name__}")


def evaluate_factor(expression: str, frame: pd.DataFrame) -> pd.Series:
    """在 AST 白名单内安全求值因子公式，返回对齐 frame.index 的 float Series。

    仅允许 OHLCV 列名与 shift/rolling/pct_change/ewm/diff/clip 等有限方法，
    不允许任意属性访问、函数调用或下标，杜绝代码注入。
    """
    if not expression or not expression.strip():
        raise ValueError("因子公式不能为空")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"公式解析失败：{error}") from error
    try:
        _validate_factor_node(tree)
    except ValueError as error:
        raise ValueError(f"公式解析失败：{error}") from error
    namespace = {column: frame[column].astype(float) for column in _FACTOR_COLUMNS}
    namespace.update({
        "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
        "float": float, "int": int, "bool": bool,
    })
    try:
        result = eval(compile(tree, "<factor>", "eval"), {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as error:
        raise ValueError(f"公式求值失败：{error}") from error
    if isinstance(result, pd.Series):
        return result.astype(float)
    if np.isscalar(result):
        return pd.Series(float(result), index=frame.index)
    return pd.Series(np.asarray(result, dtype=float), index=frame.index)


def evaluate_conditions(
    factor_values: dict[str, pd.Series],
    conditions: list[dict[str, Any]],
    combine: str = "AND",
) -> pd.Series:
    """计算条件组合的逐日布尔掩码，NaN 视为不满足。"""
    if not conditions:
        raise ValueError("条件不能为空")
    masks: list[pd.Series] = []
    for cond in conditions:
        factor = str(cond["factor"])
        if factor not in factor_values:
            raise ValueError(f"未知因子：{factor}")
        op = OPS.get(str(cond["op"]))
        if op is None:
            raise ValueError(f"不支持的运算符：{cond['op']}")
        masks.append(op(factor_values[factor], float(cond["value"])).fillna(False))
    result = masks[0]
    for mask in masks[1:]:
        result = result & mask if combine == "AND" else result | mask
    return result.astype(bool)


def resolve_factor_expressions(database: Any, names: list[str]) -> dict[str, str]:
    """将因子名解析为公式：优先查自定义因子表，其次内置公式。"""
    exprs: dict[str, str] = {}
    for name in names:
        row = database.query_one("SELECT expression FROM custom_factors WHERE name=?", (name,))
        if row:
            exprs[name] = str(row["expression"])
        elif name in BUILTIN_FACTOR_FORMULAS:
            exprs[name] = BUILTIN_FACTOR_FORMULAS[name][1]
        else:
            raise ValueError(f"未知因子：{name}")
    return exprs


def run_signal_backtest(
    data_service: Any,
    settings: Any,
    factor_exprs: dict[str, str],
    entry: dict[str, Any],
    exit_: dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 5,
    exposure: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[str, float]]]:
    """逐日信号回测。返回 (trades, metrics, equity_curve)。

    entry/exit 结构：{"combine": "AND"/"OR", "conditions": [{"factor","op","value"}, ...]}
    exposure 控制总仓位暴露比例（0~1），用于留出现金缓冲、压低回撤。
    """
    universe = data_service.eligible_universe(limit=int(settings.data["backtest_max_symbols"]))
    names = dict(zip(universe["code"], universe["name"], strict=False))
    costs = TradingCosts(
        commission_rate=float(settings.trading["commission_rate"]),
        min_commission=float(settings.trading["min_commission"]),
        stamp_duty_rate=float(settings.trading["stamp_duty_rate"]),
        slippage_rate=float(settings.trading["slippage_rate"]),
    )
    entry_conditions = entry["conditions"]
    entry_combine = entry.get("combine", "AND")
    exit_conditions = exit_["conditions"]
    exit_combine = exit_.get("combine", "OR")
    stop_loss = _to_pct(exit_.get("stop_loss_pct"))
    take_profit = _to_pct(exit_.get("take_profit_pct"))
    trailing_stop = _to_pct(exit_.get("trailing_stop_pct"))
    breakout_exit = bool(exit_.get("breakout_exit"))

    # 价格与流动性过滤（与盘中扫描口径一致，限制回测股票池）
    scan_cfg = settings.raw.get("scan", {})
    min_price = float(scan_cfg.get("min_price", 0) or 0) or None
    max_price = float(scan_cfg.get("max_price", 0) or 0) or None
    min_avg_amount = float(scan_cfg.get("min_average_amount", 0) or 0) or None
    min_avg_volume = float(scan_cfg.get("min_average_volume", 0) or 0) or None

    bars_by_code = data_service.load_bars_many(universe["code"].tolist(), start_date, end_date)
    symbols: dict[str, dict[str, Any]] = {}
    calendar: set[pd.Timestamp] = set()
    for row in universe.itertuples(index=False):
        frame = bars_by_code.get(row.code)
        if frame is None or frame.empty or len(frame) < 2:
            continue
        # 价格与流动性过滤（只做一次，与盘中扫描口径一致）
        effective_price = float(frame["close"].iloc[-1])
        if min_price is not None and effective_price < min_price:
            continue
        if max_price is not None and effective_price > max_price:
            continue
        if min_avg_amount is not None and float(frame["amount"].tail(20).mean()) < min_avg_amount:
            continue
        if min_avg_volume is not None and float(frame["volume"].tail(20).mean()) < min_avg_volume:
            continue
        factor_values = {name: evaluate_factor(expr, frame) for name, expr in factor_exprs.items()}
        entry_mask = evaluate_conditions(factor_values, entry_conditions, entry_combine)
        # 离场条件可为空（仅依赖止损/止盈/移动止损/跌破箱体离场）
        exit_mask = evaluate_conditions(factor_values, exit_conditions, exit_combine) if exit_conditions else pd.Series(False, index=frame.index)
        symbols[row.code] = {
            "name": row.name,
            "frame": frame,
            "entry_shift": entry_mask.shift(1, fill_value=False),
            "exit_shift": exit_mask.shift(1, fill_value=False),
            "prev_close": frame["close"].shift(1),
            "breakout_ref": frame["close"].shift(1).rolling(4).max(),
        }
        calendar.update(frame.index)

    if not symbols:
        raise RuntimeError("没有可用于信号回测的日线数据，请先 seed-demo 或 update-data")

    ordered_dates = sorted(calendar)
    cash = float(initial_cash)
    position_cash = initial_cash / max_positions * exposure
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[tuple[str, float]] = []

    for day in ordered_dates:
        # 先平仓释放资金，再开新仓
        for code in list(positions):
            sym = symbols[code]
            if day not in sym["frame"].index:
                continue
            pos = positions[code]
            prev_close = float(sym["prev_close"].loc[day])
            if not math.isnan(prev_close):
                pos["trail_peak"] = max(pos["trail_peak"], prev_close)
            factor_exit = bool(sym["exit_shift"].loc[day])
            stop_hit = stop_loss is not None and prev_close <= pos["entry_price"] * (1 - stop_loss)
            take_hit = take_profit is not None and prev_close >= pos["entry_price"] * (1 + take_profit)
            trail_hit = trailing_stop is not None and prev_close <= pos["trail_peak"] * (1 - trailing_stop)
            breakout_hit = breakout_exit and prev_close < pos["entry_breakout"]
            if not (factor_exit or stop_hit or take_hit or trail_hit or breakout_hit):
                continue
            price = costs.slipped_price(float(sym["frame"].loc[day, "open"]), False)
            proceeds = pos["shares"] * price
            commission = costs.commission(proceeds)
            stamp = costs.stamp_duty(proceeds, True)
            cash += proceeds - commission - stamp
            pnl = proceeds - commission - stamp - pos["cost"]
            if stop_hit:
                reason = f"止损（-{stop_loss:.0%}）"
            elif take_hit:
                reason = f"止盈（+{take_profit:.0%}）"
            elif trail_hit:
                reason = f"移动止损（-{trailing_stop:.0%}）"
            elif breakout_hit:
                reason = "跌破箱体高点"
            else:
                reason = "满足离场条件"
            trades.append(_trade_record(
                code=code, name=sym["name"], pos=pos, exit_date=day, exit_price=price,
                pnl=pnl, exit_reason=reason,
            ))
            del positions[code]

        if len(positions) < max_positions:
            for row in universe.itertuples(index=False):
                if row.code in positions or len(positions) >= max_positions:
                    continue
                sym = symbols.get(row.code)
                if sym is None or day not in sym["frame"].index or not bool(sym["entry_shift"].loc[day]):
                    continue
                price = costs.slipped_price(float(sym["frame"].loc[day, "open"]), True)
                shares = round_to_lot(position_cash / price)
                if shares <= 0:
                    continue
                cost = shares * price
                commission = costs.commission(cost)
                if cash < cost + commission:
                    continue
                cash -= cost + commission
                positions[row.code] = {
                    "name": sym["name"], "entry_date": day, "entry_price": price,
                    "shares": shares, "cost": cost + commission,
                    "entry_breakout": float(sym["breakout_ref"].loc[day]),
                    "trail_peak": price,
                }

        equity = cash
        for code, pos in positions.items():
            sym = symbols[code]
            equity += pos["shares"] * (float(sym["frame"].loc[day, "close"]) if day in sym["frame"].index else pos["entry_price"])
        equity_curve.append((day.date().isoformat(), equity))

    # 期末强制平仓未了结持仓
    for code in list(positions):
        sym = symbols[code]
        pos = positions[code]
        last_price = costs.slipped_price(float(sym["frame"]["close"].iloc[-1]), False)
        proceeds = pos["shares"] * last_price
        commission = costs.commission(proceeds)
        stamp = costs.stamp_duty(proceeds, True)
        pnl = proceeds - commission - stamp - pos["cost"]
        trades.append(_trade_record(
            code=code, name=sym["name"], pos=pos, exit_date=sym["frame"].index[-1], exit_price=last_price,
            pnl=pnl, exit_reason="回测期末强制平仓",
        ))

    metrics = compute_metrics(trades, equity_curve, initial_cash)
    return trades, metrics, equity_curve


def _trade_record(code, name, pos, exit_date, exit_price, pnl, exit_reason) -> dict[str, Any]:
    pnl_pct = pnl / pos["cost"] if pos["cost"] else 0.0
    return {
        "code": code, "name": name,
        "entry_date": pos["entry_date"].date().isoformat(), "entry_price": pos["entry_price"],
        "shares": pos["shares"],
        "exit_date": exit_date.date().isoformat(), "exit_price": exit_price,
        "pnl": round(pnl, 2), "pnl_pct": pnl_pct,
        "holding_days": max(0, (exit_date - pos["entry_date"]).days),
        "status": "CLOSED", "entry_reason": "满足买入条件", "exit_reason": exit_reason,
    }


def compute_metrics(trades: list[dict[str, Any]], equity_curve: list[tuple[str, float]], initial_cash: float) -> dict[str, Any]:
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    total_pnl = sum(t["pnl"] for t in closed)
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)

    max_drawdown = 0.0
    peak = -math.inf
    for _date, equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    annual_return = 0.0
    if len(equity_curve) >= 2 and equity_curve[-1][1] > 0:
        days = max(1, (pd.Timestamp(equity_curve[-1][0]) - pd.Timestamp(equity_curve[0][0])).days)
        total_return = equity_curve[-1][1] / initial_cash - 1 if initial_cash else 0.0
        annual_return = (equity_curve[-1][1] / initial_cash) ** (365.25 / days) - 1 if equity_curve[-1][1] / initial_cash > 0 else -1.0

    return {
        "total_trades": len(closed),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "total_pnl": round(total_pnl, 2),
        "total_return": (equity_curve[-1][1] / initial_cash - 1) if equity_curve and initial_cash else 0.0,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "avg_pnl": total_pnl / len(closed) if closed else 0.0,
        "avg_holding_days": sum(t["holding_days"] for t in closed) / len(closed) if closed else 0.0,
    }


FACTOR_LABELS = {
    "ma20_dev": "20日线偏离", "ma60_dev": "60日线偏离",
    "box_range_4": "4日箱体振幅", "box_range_5": "5日箱体振幅",
    "breakout_4": "突破前4日高点", "breakout_5": "突破前5日高点",
    "lower_high": "未破前高", "rally_5": "5日涨幅", "rally_10": "10日涨幅",
    "pullback_1": "当日涨跌", "vol_ratio": "量比", "near_low_20": "距20日低点",
    "up_days_5": "5日连涨天数", "v_recovery": "深V回收",
    "rally_3": "三日涨跌幅", "gap_up": "集合竞价高开", "morning_star": "早晨之星",
    "breakout_20": "突破前20日高点", "breakout_60": "突破前60日高点", "ma_trend": "均线多头度",
}


def describe_conditions(conditions: list[dict[str, Any]], combine: str = "AND") -> str:
    joiner = " 且 " if combine == "AND" else " 或 "
    return joiner.join(
        f"{FACTOR_LABELS.get(c['factor'], c['factor'])}{c['op']}{c['value']}" for c in conditions
    )


def describe_exit(exit_config: dict[str, Any]) -> str:
    parts: list[str] = []
    conditions = exit_config.get("conditions", [])
    if conditions:
        parts.append(describe_conditions(conditions, exit_config.get("combine", "AND")))
    if exit_config.get("stop_loss_pct"):
        parts.append(f"止损{exit_config['stop_loss_pct']:.0%}")
    if exit_config.get("take_profit_pct"):
        parts.append(f"止盈{exit_config['take_profit_pct']:.0%}")
    if exit_config.get("trailing_stop_pct"):
        parts.append(f"移动止损{exit_config['trailing_stop_pct']:.0%}")
    if exit_config.get("breakout_exit"):
        parts.append("跌破箱体高点")
    return " / ".join(parts) if parts else "无"


def fetch_scan_quotes(data_service: Any, database: Any) -> dict[str, dict[str, float]]:
    """刷新全市场实时快照并读取 {code: {price, volume}}，供盘中买点扫描覆盖最新价与当日成交量。"""
    universe = data_service.eligible_universe(limit=int(data_service.settings.data["strategy_scan_symbols"]))
    try:
        data_service.refresh_quotes(universe["code"].tolist())
    except Exception as error:
        LOG.warning("盘中实时行情刷新失败：%s", error)
    quotes: dict[str, dict[str, float]] = {}
    for q in database.query_all("SELECT code, price, volume FROM market_quotes WHERE price>0"):
        price = float(q["price"] or 0)
        if price <= 0:
            continue
        volume = float(q["volume"] or 0)
        quotes[str(q["code"])] = {"price": price, "volume": volume if volume > 0 else 0.0}
    return quotes


def find_buy_candidates(
    data_service: Any,
    database: Any,
    as_of_date: str | None = None,
    limit: int | None = None,
    current_quotes: dict[str, float] | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    min_average_amount: float | None = None,
    min_average_volume: float | None = None,
    strategy_names: list[str] | None = None,
    exclude_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """扫描已保存策略，找出最新日线上符合买入条件的股票。

    每个策略结果为 {"strategy", "entry_desc", "exit_desc", "matches"}，
    matches 为 [{"code","name","price","factors"}, ...]。
    current_quotes 为 {code: 实时价} 或 {code: {"price": 实时价, "volume": 当日累计成交量}}，
    传入时用实时价覆盖最新 close、当日成交量覆盖最新 volume，以便盘中扫描价格与量能条件都实时生效。
    min_average_amount / min_average_volume 按近 20 日均值过滤流动性。
    """
    strategies = database.query_all("SELECT name, entry_json, exit_json FROM signal_strategies WHERE enabled=1 ORDER BY created_at DESC")
    if strategy_names:
        strategies = [s for s in strategies if s["name"] in strategy_names]
    if exclude_prefix:
        strategies = [s for s in strategies if not s["name"].startswith(exclude_prefix)]
    if not strategies:
        return []
    universe = data_service.eligible_universe(limit=limit or int(data_service.settings.data["strategy_scan_symbols"]))

    # 只加载最近 260 个交易日（覆盖 52 周新高等因子所需最长窗口），
    # 全市场扫描时避免加载 2020 至今的全历史（1600+ 天）导致过慢。
    lookback_start = data_service.lookback_start(260)
    # 预加载所有标的的日线（只加载一次，供所有策略复用），并用实时价覆盖最新 close
    loaded = data_service.load_bars_many(universe["code"].tolist(), start_date=lookback_start, end_date=as_of_date)
    bars_cache: dict[str, tuple[str, pd.DataFrame, str]] = {}
    for row in universe.itertuples(index=False):
        frame = loaded.get(row.code)
        if frame is None or frame.empty or len(frame) < 20:
            continue
        realtime = False
        quote = current_quotes.get(row.code) if current_quotes else None
        if quote:
            if isinstance(quote, dict):
                price = float(quote.get("price") or 0)
                intraday_volume = float(quote.get("volume") or 0)
            else:
                price, intraday_volume = float(quote), 0.0
            if price > 0:
                frame = frame.copy()
                frame.iloc[-1, frame.columns.get_loc("close")] = price
                if intraday_volume > 0:
                    frame.iloc[-1, frame.columns.get_loc("volume")] = intraday_volume
                realtime = True
        # 价格与流动性过滤（只做一次，供所有策略复用）
        effective_price = float(frame["close"].iloc[-1])
        if max_price is not None and effective_price > max_price:
            continue
        if min_price is not None and effective_price < min_price:
            continue
        if min_average_amount is not None and float(frame["amount"].tail(20).mean()) < min_average_amount:
            continue
        if min_average_volume is not None and float(frame["volume"].tail(20).mean()) < min_average_volume:
            continue
        bar_date = str(frame["trade_date"].iloc[-1].date())
        bars_cache[row.code] = (row.name, frame, "实时" if realtime else bar_date)

    results: list[dict[str, Any]] = []
    for strat in strategies:
        try:
            entry = json.loads(strat["entry_json"])
            exit_cfg = json.loads(strat["exit_json"]) if strat.get("exit_json") else {}
        except (ValueError, TypeError):
            continue
        conditions = entry.get("conditions", [])
        if not conditions:
            continue
        factor_names = [c["factor"] for c in conditions]
        try:
            exprs = resolve_factor_expressions(database, factor_names)
        except ValueError:
            continue
        matches: list[dict[str, Any]] = []
        for code, (name, frame, price_date) in bars_cache.items():
            try:
                factor_values = {fn: evaluate_factor(expr, frame) for fn, expr in exprs.items()}
                mask = evaluate_conditions(factor_values, conditions, entry.get("combine", "AND"))
            except (ValueError, KeyError):
                continue
            if bool(mask.iloc[-1]):
                price = float(frame["close"].iloc[-1])
                matches.append({
                    "code": code, "name": name, "price": round(price, 2),
                    "price_date": price_date,
                    "factors": {fn: round(float(factor_values[fn].iloc[-1]), 4) for fn in exprs},
                })
        results.append({
            "strategy": strat["name"],
            "entry_desc": describe_conditions(conditions, entry.get("combine", "AND")),
            "exit_desc": describe_exit(exit_cfg),
            "matches": matches,
        })
    return results


def fetch_recent_notices(days: int = 5) -> dict[str, list[str]]:
    """获取最近 N 个自然日的公司公告，返回 {code: [公告标题, ...]}。"""
    import akshare as ak

    notices: dict[str, list[str]] = {}
    today = pd.Timestamp.now()
    for i in range(days):
        date = (today - pd.Timedelta(days=i)).strftime("%Y%m%d")
        try:
            frame = ak.stock_notice_report(symbol="全部", date=date)
            for row in frame.to_dict("records"):
                code = str(row.get("代码", "")).zfill(6)
                title = str(row.get("公告标题", "")).strip()
                if code and title and any(k in title for k in _IMPORTANT_NOTICE_KEYWORDS):
                    notices.setdefault(code, []).append(title)
        except Exception:
            continue
    return notices


def fetch_hot_stocks(limit: int = 200) -> set[str]:
    """获取东方财富人气榜（热股排名）前 limit 只代码。失败返回空集。"""
    import akshare as ak

    try:
        frame = ak.stock_hot_rank_em()
    except Exception as error:
        LOG.warning("热股排名获取失败：%s", error)
        return set()
    if "代码" not in frame.columns:
        LOG.warning("热股排名数据缺少代码列：%s", list(frame.columns))
        return set()
    return {str(code)[-6:].zfill(6) for code in frame["代码"].head(limit)}


# 巨潮行业关键词 -> 同花顺板块（近似映射，用于展示候选所属板块当日涨跌）
_INDUSTRY_BOARD = {
    "货币金融": "银行", "资本市场": "证券", "保险": "保险",
    "医药": "医药", "房地产": "房地产开发", "零售": "零售",
    "软件": "软件", "计算机": "计算机设备", "通信": "通信设备",
    "电子": "元件", "半导体": "半导体", "食品": "食品",
    "饮料": "白酒", "汽车": "汽车", "钢铁": "钢铁",
    "煤炭": "煤炭", "电力": "电力", "有色": "有色金属",
    "化学": "化学制品", "石油": "石油", "航空": "航空机场",
    "航运": "航运港口", "建筑": "建筑装饰", "建材": "建筑材料",
    "农业": "农业", "环保": "环保", "教育": "教育",
    "文化": "文化传媒", "旅游": "旅游", "纺织": "纺织制造",
    "服装": "服装家纺", "造纸": "造纸", "家具": "家居用品",
    "通用设备": "通用设备", "专用设备": "专用设备", "仪器": "仪器仪表",
    "电气": "电气设备", "机械": "机械", "金属制品": "金属制品",
    "橡胶": "橡胶", "塑料": "塑料", "玻璃": "玻璃", "水泥": "水泥",
}

# 重要公告关键词（只保留这些类型的公告）
_IMPORTANT_NOTICE_KEYWORDS = (
    "年报", "半年报", "季报", "季度报告",
    "重大合同", "中标", "重组", "收购", "并购", "定增", "增发", "重大资产",
    "减持", "增持", "回购", "质押", "解禁",
    "分红", "送转", "派息", "权益分派",
    "诉讼", "仲裁", "处罚", "立案", "风险警示", "退市",
    "股权激励", "股权转让", "担保",
)

# 负面公告关键词（用于质量过滤，排除有这些公告的候选，降低政策/基本面风险）
_NEGATIVE_NOTICE_KEYWORDS = ("减持", "处罚", "立案", "退市", "风险警示", "诉讼", "仲裁")


def match_board_for_industry(industry: str, sector_changes: dict[str, float]) -> tuple[str, float] | None:
    """把巨潮行业名近似映射到同花顺板块，返回 (板块名, 涨跌幅) 或 None。"""
    if not industry:
        return None
    for keyword, board in _INDUSTRY_BOARD.items():
        if keyword in industry:
            pct = sector_changes.get(board)
            if pct is not None:
                return board, pct
    return None


def fetch_sector_summary(top_n: int = 3) -> tuple[str, dict[str, float]]:
    """获取同花顺行业板块涨跌榜，返回 (表现描述文本, {板块名: 涨跌幅})。单次拉取供两用。"""
    import akshare as ak

    try:
        frame = ak.stock_board_industry_summary_ths()
        frame = frame.sort_values("涨跌幅", ascending=False)
        changes = {str(r["板块"]): float(r["涨跌幅"]) for r in frame.to_dict("records")}
        top = frame.head(top_n).to_dict("records")
        bottom = frame.tail(top_n).to_dict("records")
        strong: list[str] = []
        for r in top:
            leader = str(r.get("领涨股", "")).strip()
            leader_pct = float(r.get("领涨股-涨跌幅", 0) or 0)
            if leader:
                strong.append(f"{r['板块']} {r['涨跌幅']:+.2f}%（领涨 {leader} {leader_pct:+.2f}%）")
            else:
                strong.append(f"{r['板块']} {r['涨跌幅']:+.2f}%")
        weak = "、".join(f"{r['板块']}{r['涨跌幅']:+.2f}%" for r in bottom)
        text = "今日板块（同花顺）\n  强势：" + "；".join(strong) + "\n  弱势：" + weak
        return text, changes
    except Exception:
        return "", {}


def fetch_industries(database: Any, codes: list[str]) -> dict[str, str]:
    """获取股票所属行业（巨潮资讯），并缓存到 stock_industry 表。"""
    import akshare as ak
    from .models import utc_now_text

    cached = {r["code"]: r["industry"] for r in database.query_all("SELECT code, industry FROM stock_industry")}
    industries = {c: cached[c] for c in codes if c in cached}
    missing = [c for c in codes if c not in cached]
    if not missing:
        return industries
    now = utc_now_text()
    for code in missing:
        try:
            frame = ak.stock_profile_cninfo(symbol=code)
            row = frame.iloc[0].to_dict() if len(frame) else {}
            industry = str(row.get("所属行业", "")).strip()
            if industry:
                industries[code] = industry
                database.execute(
                    "INSERT INTO stock_industry(code, industry, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(code) DO UPDATE SET industry=excluded.industry, updated_at=excluded.updated_at",
                    (code, industry, now),
                )
        except Exception:
            continue
    return industries


def _price_source_note(results: list[dict[str, Any]]) -> str:
    """根据候选的价格日期，给出行情数据来源与新鲜度说明，避免把陈旧收盘价误当实时价。"""
    price_dates = {m.get("price_date", "") for item in results for m in item["matches"]}
    if price_dates == {"实时"}:
        return "行情：盘中实时价"
    dates = sorted(d for d in price_dates if d and d != "实时")
    if not dates:
        return "行情：日线收盘价"
    latest = max(dates)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    note = f"行情：日线收盘价（数据截至 {latest}）"
    if latest < today:
        note += "　⚠️ 行情数据未更新至最新交易日，现价可能过期"
    return note


def format_buy_report(
    results: list[dict[str, Any]],
    notices: dict[str, list[str]] | None = None,
    industries: dict[str, str] | None = None,
    max_per_strategy: int = 10,
    sector_changes: dict[str, float] | None = None,
) -> str:
    """把买点扫描结果格式化为中文报告文本（按现价降序，每策略最多报 max_per_strategy 只）。"""
    if not results or all(not item["matches"] for item in results):
        return "今日无符合条件的买入候选"
    notices = notices or {}
    industries = industries or {}
    sector_changes = sector_changes or {}
    sep = "-" * 36
    total_all = sum(len(item["matches"]) for item in results)
    unique_codes = {m["code"] for item in results for m in item["matches"]}
    lines: list[str] = []
    lines.append(f"A股量化买点扫描 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"今日总览：{len(results)} 个策略命中，{total_all} 只候选（去重 {len(unique_codes)} 只）")
    lines.append(_price_source_note(results))
    lines.append(sep)

    for si, item in enumerate(results, 1):
        all_matches = sorted(item["matches"], key=lambda m: m.get("price", 0), reverse=True)
        matches = all_matches[:max_per_strategy]
        if len(all_matches) > len(matches):
            title = f"策略{si}｜{item['strategy']}｜候选 {len(all_matches)} 只，精选前 {len(matches)}（按现价降序）"
        else:
            title = f"策略{si}｜{item['strategy']}｜候选 {len(all_matches)} 只（按现价降序）"
        lines.append("")
        lines.append(title)
        lines.append(f"  买：{item['entry_desc']}")
        lines.append(f"  卖：{item['exit_desc']}")
        for i, m in enumerate(matches, 1):
            industry_display = ""
            ind = industries.get(m["code"])
            if ind:
                board_match = match_board_for_industry(ind, sector_changes)
                if board_match:
                    board, pct = board_match
                    industry_display = f"｜{board} {pct:+.2f}%"
                else:
                    industry_display = f"｜{ind}"
            date_tag = ""
            price_date = m.get("price_date", "")
            if price_date and price_date != "实时":
                date_tag = f"（{price_date}）"
            lines.append(f"  {i:>2}. ¥{m['price']:>7.2f}  {m['code']}  {m['name']}{industry_display}{date_tag}")
            factor_txt = "，".join(
                f"{FACTOR_LABELS.get(k, k)} {v:+.2f}" for k, v in m.get("factors", {}).items()
            )
            if factor_txt:
                lines.append(f"       因子：{factor_txt}")
            for title in notices.get(m["code"], [])[:2]:
                lines.append(f"       公告：{title}")
    lines.append("")
    lines.append(sep)
    lines.append("系统自动扫描，仅供研究参考，不构成投资建议。")
    return "\n".join(lines)


def format_buy_report_html(
    results: list[dict[str, Any]],
    notices: dict[str, list[str]] | None = None,
    industries: dict[str, str] | None = None,
    max_per_strategy: int = 10,
    sector_changes: dict[str, float] | None = None,
    sector_text: str = "",
) -> str:
    """生成 HTML 表格版买点扫描报告（邮件用）。"""
    if not results or all(not item["matches"] for item in results):
        return "<p>今日无符合条件的买入候选</p>"
    notices = notices or {}
    industries = industries or {}
    sector_changes = sector_changes or {}
    esc = _html.escape
    total_all = sum(len(item["matches"]) for item in results)
    unique_codes = {m["code"] for item in results for m in item["matches"]}

    parts: list[str] = []
    parts.append('<div style="font-family:Microsoft YaHei,Arial,sans-serif;max-width:860px;">')
    parts.append('<div style="background:#34495e;color:#ffffff;padding:12px 16px;border-radius:6px;">')
    parts.append(f'<h3 style="margin:0;font-size:17px;">A股量化买点扫描 · {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</h3>')
    parts.append(f'<p style="margin:5px 0 0;font-size:13px;opacity:0.92;">今日总览：{len(results)} 个策略命中，{total_all} 只候选（去重 {len(unique_codes)} 只）</p>')
    source_note = _price_source_note(results)
    if "⚠️" in source_note:
        parts.append(f'<p style="margin:5px 0 0;font-size:13px;color:#c0392b;font-weight:600;">{esc(source_note)}</p>')
    else:
        parts.append(f'<p style="margin:5px 0 0;font-size:13px;opacity:0.9;">{esc(source_note)}</p>')
    parts.append('</div>')
    if sector_text:
        parts.append(f'<div style="margin:10px 0;padding:10px 14px;background:#f4f6f8;border-left:4px solid #3498db;font-size:13px;color:#333;">{esc(sector_text).replace(chr(10), "<br>")}</div>')

    for si, item in enumerate(results, 1):
        all_matches = sorted(item["matches"], key=lambda m: m.get("price", 0), reverse=True)
        matches = all_matches[:max_per_strategy]
        if len(all_matches) > len(matches):
            title = f"策略{si}｜{item['strategy']}｜候选 {len(all_matches)} 只，精选前 {len(matches)}（按现价降序）"
        else:
            title = f"策略{si}｜{item['strategy']}｜候选 {len(all_matches)} 只（按现价降序）"
        parts.append(f'<h4 style="margin:18px 0 6px;padding-left:9px;border-left:4px solid #3498db;font-size:15px;color:#2c3e50;">{esc(title)}</h4>')
        parts.append(f'<p style="margin:3px 0 8px;font-size:12px;color:#666;">'
                     f'<span style="color:#e74c3c;font-weight:600;">买</span> {esc(item["entry_desc"])}　'
                     f'<span style="color:#27ae60;font-weight:600;">卖</span> {esc(item["exit_desc"])}</p>')
        parts.append('<table border="0" cellspacing="0" cellpadding="7" style="border-collapse:collapse;width:100%;font-size:13px;">')
        parts.append('<tr style="background:#34495e;">'
                     '<th style="color:#fff;text-align:left;font-weight:600;">#</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">现价</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">代码</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">名称</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">所属板块</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">因子</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;">公告</th>'
                     '</tr>')
        for i, m in enumerate(matches, 1):
            bg = ' style="background:#f8f9fa;"' if i % 2 == 0 else ''
            ind = industries.get(m["code"])
            if ind:
                board_match = match_board_for_industry(ind, sector_changes)
                if board_match:
                    board, pct = board_match
                    color = "#e74c3c" if pct >= 0 else "#27ae60"
                    industry_html = f'<span style="color:{color};font-weight:600;">{esc(board)} {pct:+.2f}%</span>'
                else:
                    industry_html = esc(ind)
            else:
                industry_html = ""
            factor_txt = "，".join(f"{FACTOR_LABELS.get(k, k)} {v:+.2f}" for k, v in m.get("factors", {}).items())
            notice_txt = "；".join(notices.get(m["code"], [])[:2])
            price_cell = f'¥{m["price"]:.2f}'
            price_date = m.get("price_date", "")
            if price_date and price_date != "实时":
                price_cell += f' <span style="font-weight:400;color:#999;font-size:11px;">（{esc(price_date)}）</span>'
            parts.append(
                f'<tr{bg}><td style="color:#999;">{i}</td>'
                f'<td style="font-weight:700;color:#d35400;">{price_cell}</td>'
                f'<td>{m["code"]}</td><td>{esc(m["name"])}</td>'
                f'<td>{industry_html}</td>'
                f'<td style="color:#666;font-size:12px;">{esc(factor_txt)}</td>'
                f'<td style="color:#666;font-size:12px;">{esc(notice_txt)}</td></tr>'
            )
        parts.append('</table>')
    parts.append('<p style="margin:16px 0 0;color:#999;font-size:12px;">系统自动扫描，仅供研究参考，不构成投资建议。</p>')
    parts.append('</div>')
    return "".join(parts)


def filter_candidates_by_quality(
    candidates: list[dict[str, Any]],
    sector_changes: dict[str, float],
    industries: dict[str, str],
    notices: dict[str, list[str]],
    settings: Any,
) -> list[dict[str, Any]]:
    """用板块强弱、热股排名、负面公告过滤候选，减少出手次数、提高胜率。

    返回与 ``candidates`` 同构的过滤结果；被整体排除的策略项不会出现在结果里。
    """
    scan_cfg = settings.raw.get("scan", {})
    min_sector_change = scan_cfg.get("min_sector_change", 0.0)
    min_sector_change = float(min_sector_change) if min_sector_change is not None else None
    require_hot_stock = bool(scan_cfg.get("require_hot_stock", False))
    hot_limit = int(scan_cfg.get("hot_rank_limit", 200) or 200)
    exclude_negative = bool(scan_cfg.get("exclude_negative_notice", True))

    negative_codes: set[str] = set()
    if exclude_negative:
        negative_codes = {
            code for code, titles in notices.items()
            if any(any(keyword in title for keyword in _NEGATIVE_NOTICE_KEYWORDS) for title in titles)
        }
    hot_codes = fetch_hot_stocks(hot_limit) if require_hot_stock else set()

    filtered: list[dict[str, Any]] = []
    for item in candidates:
        matches = []
        for match in item["matches"]:
            code = match["code"]
            if negative_codes and code in negative_codes:
                continue
            if require_hot_stock and hot_codes and code not in hot_codes:
                continue
            if min_sector_change is not None and code in industries:
                board_info = match_board_for_industry(industries[code], sector_changes)
                if board_info is not None and board_info[1] < min_sector_change:
                    continue
            matches.append(match)
        if matches:
            filtered.append({**item, "matches": matches})
    return filtered


def fetch_stock_metrics() -> dict[str, dict[str, float]]:
    """获取全 A 股的换手率/动态市盈率/总市值（东方财富）。失败回退 Tushare daily_basic，再失败返回空 dict。"""
    import akshare as ak

    try:
        frame = ak.stock_zh_a_spot_em()
    except Exception as error:
        LOG.warning("股票指标获取失败，尝试 Tushare 回退：%s", error)
        return _stock_metrics_from_tushare()

    def num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    metrics: dict[str, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        code = str(row.get("代码", "")).zfill(6)
        if not code or code == "000000":
            continue
        metrics[code] = {
            "turnover": num(row.get("换手率")),
            "pe": num(row.get("市盈率-动态")),
            "market_cap": num(row.get("总市值")),
        }
    if not metrics:
        return _stock_metrics_from_tushare()
    return metrics


def _stock_metrics_from_tushare() -> dict[str, dict[str, float]]:
    """Tushare daily_basic 全市场快照回退。日线指标为收盘后数据，取最近一个有数据的交易日。"""
    from datetime import datetime, timedelta

    from .data.providers import tushare_pro_from_env

    pro = tushare_pro_from_env()
    if pro is None:
        return {}
    frame = None
    for days_back in range(0, 8):
        trade_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            frame = pro.daily_basic(trade_date=trade_date, fields="ts_code,turnover_rate,pe_ttm,total_mv")
        except Exception as error:
            LOG.warning("Tushare daily_basic 获取失败 %s：%s", trade_date, error)
            frame = None
        if frame is not None and not frame.empty:
            break
    if frame is None or frame.empty:
        return {}

    def num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    metrics: dict[str, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        code = str(row.get("ts_code", ""))[:6]
        if not code:
            continue
        metrics[code] = {
            "turnover": num(row.get("turnover_rate")),
            "pe": num(row.get("pe_ttm")),
            "market_cap": num(row.get("total_mv")) * 1e4,  # Tushare 总市值单位为万元
        }
    LOG.info("Tushare daily_basic 回退提供 %s 只标的的质量指标", len(metrics))
    return metrics


def filter_candidates_by_metrics(
    candidates: list[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
    settings: Any,
) -> list[dict[str, Any]]:
    """用换手率/市盈率/总市值过滤候选（仅盘中扫描，指标无历史不可回测）。"""
    scan_cfg = settings.raw.get("scan", {})
    min_turnover = float(scan_cfg.get("min_turnover", 0) or 0) or None
    max_pe = float(scan_cfg.get("max_pe", 0) or 0) or None
    min_market_cap = float(scan_cfg.get("min_market_cap", 0) or 0) or None
    if not metrics or (min_turnover is None and max_pe is None and min_market_cap is None):
        return candidates

    filtered: list[dict[str, Any]] = []
    for item in candidates:
        matches = []
        for match in item["matches"]:
            meta = metrics.get(match["code"])
            if meta is None:
                matches.append(match)
                continue
            if min_turnover is not None and 0 < meta["turnover"] < min_turnover:
                continue
            if max_pe is not None and meta["pe"] > max_pe:
                continue
            if min_market_cap is not None and 0 < meta["market_cap"] < min_market_cap:
                continue
            matches.append(match)
        if matches:
            filtered.append({**item, "matches": matches})
    return filtered


def build_buy_report(data_service: Any, database: Any, current_quotes: dict[str, float] | None = None, strategy_names: list[str] | None = None, exclude_prefix: str | None = None) -> tuple[bool, str, str]:
    """完整买点扫描报告：扫描 + 板块涨跌 + 公司公告 + 所属行业 + 质量过滤。

    返回 (是否有候选, 纯文本报告, HTML 报告)。无候选时不拉取外部公告/板块数据。
    """
    scan_cfg = data_service.settings.raw.get("scan", {})
    max_price = float(scan_cfg.get("max_price", 0) or 0) or None
    min_price = float(scan_cfg.get("min_price", 0) or 0) or None
    min_average_amount = float(scan_cfg.get("min_average_amount", 0) or 0) or None
    min_average_volume = float(scan_cfg.get("min_average_volume", 0) or 0) or None
    max_per_strategy = int(scan_cfg.get("max_per_strategy", 10))
    candidates = find_buy_candidates(
        data_service, database, current_quotes=current_quotes, max_price=max_price,
        min_price=min_price, min_average_amount=min_average_amount, min_average_volume=min_average_volume,
        strategy_names=strategy_names, exclude_prefix=exclude_prefix,
    )
    if not any(item["matches"] for item in candidates):
        return False, "今日无符合条件的买入候选（或没有买得起价格的候选）", ""
    codes = sorted({m["code"] for item in candidates for m in item["matches"]})
    notices = fetch_recent_notices()
    sector, sector_changes = fetch_sector_summary()
    industries = fetch_industries(database, codes)
    # 质量过滤：板块强弱、热股排名、负面公告（减少出手、提高胜率）
    candidates = filter_candidates_by_quality(candidates, sector_changes, industries, notices, data_service.settings)
    # 财务/流动性指标过滤：换手率、市盈率、总市值（配置了阈值才拉取）
    metric_keys = ("min_turnover", "max_pe", "min_market_cap")
    if any(float(scan_cfg.get(k, 0) or 0) > 0 for k in metric_keys):
        candidates = filter_candidates_by_metrics(candidates, fetch_stock_metrics(), data_service.settings)
    if not any(item["matches"] for item in candidates):
        return False, "候选均被质量过滤排除（板块弱势 / 负面公告 / 非热股 / 指标不符）", ""
    text = format_buy_report(candidates, notices=notices, industries=industries, sector_changes=sector_changes, max_per_strategy=max_per_strategy)
    if sector:
        text = sector + "\n\n" + text
    html = format_buy_report_html(candidates, notices=notices, industries=industries, max_per_strategy=max_per_strategy, sector_changes=sector_changes, sector_text=sector)
    return True, text, html
