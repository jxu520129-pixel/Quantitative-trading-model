"""多策略组合回测：横截面因子策略回测 + 风险平价/逆波动率组合合并。

与 Backtrader 的周度调仓回测不同，本模块直接基于全市场历史库
（``data/ashare_quant_hist.db``）做**宽表向量化**的横截面回测，产出每个策略的
日度收益序列，再把多个低相关策略合并成组合，用于评估「分散持有」对年化/回撤的影响。

严格无未来函数：第 T 日收盘打分、从第 T+1 日才开始计收益。
停牌日以 ``ffill`` 延续前收（收益记为 0），成本按每次调仓的换手率近似收取。
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_hist_bars(db_path: str | Path, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从历史库读取全市场日线，返回 ``(close, amount, high, low)`` 四张宽表。

    宽表以交易日为索引、证券代码为列；停牌日以前收前向填充（``ffill``），
    保证 ``pct_change`` 在停牌日收益为 0。
    """
    clauses = ["1=1"]
    params: list[str] = []
    if start:
        clauses.append("trade_date>=?")
        params.append(start.replace("-", ""))
    if end:
        clauses.append("trade_date<=?")
        params.append(end.replace("-", ""))
    con = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            f"SELECT code,trade_date,open,high,low,close,amount FROM daily_bars WHERE {' AND '.join(clauses)}",
            con, params=params,
        )
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError("历史库没有可用的日线数据，请先执行 scripts/fetch_hist_data.py")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates(["code", "trade_date"])
    close = frame.pivot(index="trade_date", columns="code", values="close").ffill()
    amount = frame.pivot(index="trade_date", columns="code", values="amount").ffill()
    high = frame.pivot(index="trade_date", columns="code", values="high").ffill()
    low = frame.pivot(index="trade_date", columns="code", values="low").ffill()

    # 数据清洗：过滤复权因子跳变的股票。次新股（创业板/科创板）在前复权时
    # 可能因 adj_factor 缺失/跳变出现单日 ±30% 以上的假收益，污染动量等因子的回测。
    # 以单日 |涨跌幅| > 30% 作为异常标志（超过任何板块的合法涨跌停上限）。
    ret = close.pct_change()
    bad_codes = (ret.abs() > 0.30).any(axis=0)
    bad_codes = bad_codes[bad_codes].index
    if len(bad_codes):
        close = close.drop(columns=bad_codes)
        amount = amount.drop(columns=bad_codes)
        high = high.drop(columns=bad_codes)
        low = low.drop(columns=bad_codes)
    return close, amount, high, low


# ---------------------------------------------------------------------------
# 宽表因子（与 strategies/factors.py 的单标的口径一致，但向量化到全市场）
# ---------------------------------------------------------------------------

def _momentum(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return close / close.shift(window) - 1.0


def _reversal(close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    return -(close.pct_change(window))


def _volatility(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return close.pct_change().rolling(window).std()


def _liquidity(amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return amount.rolling(window).mean()


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    hist = dif - dif.ewm(span=signal, adjust=False).mean()
    return hist / close


def _ma_trend(close: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)) -> pd.DataFrame:
    averages = [close.rolling(w).mean() for w in windows]
    score = sum((a > b).astype(float) for a, b in zip(averages, averages[1:]))
    return score / len(windows)


def _high_52w(close: pd.DataFrame, high: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    return close / high.rolling(window).max() - 1.0


def _atr(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()]).groupby(level=0).max()
    return tr.rolling(window).mean() / close


def _amihud(close: pd.DataFrame, amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ratio = close.pct_change().abs() / amount.replace(0.0, np.nan)
    return ratio.rolling(window).mean()


FACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "momentum": _momentum,
    "reversal": _reversal,
    "volatility": _volatility,
    "liquidity": _liquidity,
    "rsi": _rsi,
    "macd": _macd,
    "ma_trend": _ma_trend,
    "high_52w": _high_52w,
    "atr": _atr,
    "amihud": _amihud,
}


# ---------------------------------------------------------------------------
# 横截面策略回测
# ---------------------------------------------------------------------------

def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "W") -> list[pd.Timestamp]:
    """返回调仓日（打分日）：每周/每月最后一个交易日，升序。"""
    if freq == "W":
        key = index.isocalendar().week
        key = index.isocalendar().year.astype(str) + "-" + key.astype(str)
    elif freq == "M":
        key = index.to_period("M").astype(str)
    else:
        raise ValueError(f"不支持的调仓频率：{freq}")
    s = pd.Series(index, index=index)
    return sorted(s.groupby(key.values).last().tolist())


def _cross_section_zscore(values: pd.Series) -> pd.Series:
    """对某个时点的横截面因子值做 z-score 标准化（标准差为 0 时返回 0）。"""
    std = values.std()
    if not std or math.isnan(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def cross_sectional_returns(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    factor_specs: list[tuple[str, float]],
    factor_params: dict[str, dict] | None = None,
    top_n: int = 5,
    rebalance: str = "W",
    min_average_amount: float = 0.0,
    cost_rate: float = 0.0016,
) -> pd.Series:
    """横截面等权 top-N 组合的日度收益序列。

    参数
    ----
    factor_specs : ``[(因子名, 权重), ...]``，权重符号代表方向（正=越高越好）。
    factor_params : ``{因子名: {参数: 值}}`` 覆盖各因子默认窗口。
    top_n : 每个调仓日持有的股票数。
    rebalance : 调仓频率，``W`` 周度 / ``M`` 月度。
    min_average_amount : 近 20 日平均成交额下限（元），用于过滤低流动性标的。
    cost_rate : 单次调仓按换手率收取的近似成本率（佣金+印花税+滑点）。
    """
    factor_params = factor_params or {}
    # 预计算各因子宽表
    factor_frames: dict[str, pd.DataFrame] = {}
    for name, _weight in factor_specs:
        params = factor_params.get(name, {})
        if name == "high_52w":
            factor_frames[name] = _high_52w(close, high, **params)
        elif name == "atr":
            factor_frames[name] = _atr(close, high, low, **params)
        elif name == "amihud":
            factor_frames[name] = _amihud(close, amount, **params)
        elif name == "liquidity":
            factor_frames[name] = _liquidity(amount, **params)
        else:
            factor_frames[name] = FACTOR_REGISTRY[name](close, **params)

    rets = close.pct_change()
    avg_amount = amount.rolling(20).mean()

    dates = close.index
    rebalance_days = _rebalance_dates(dates, rebalance)
    portfolio_ret = pd.Series(0.0, index=dates, dtype=float)
    prev_holdings: list[str] | None = None

    for i, t in enumerate(rebalance_days):
        # 第 t 日收盘打分（因子值只用到 t 及以前，无未来函数）
        scores = pd.Series(0.0, index=close.columns, dtype=float)
        valid = pd.Series(True, index=close.columns)
        for name, weight in factor_specs:
            row = factor_frames[name].loc[t]
            scores = scores + weight * _cross_section_zscore(row)
            valid &= row.notna()
        if min_average_amount > 0:
            valid &= (avg_amount.loc[t] >= min_average_amount)
        candidates = scores[valid].dropna()
        if candidates.empty:
            continue
        holdings = candidates.sort_values(ascending=False).head(top_n).index.tolist()

        # 持仓期：第 t+1 日到下一个调仓日（含）
        next_t = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else dates[-1]
        period = dates[(dates > t) & (dates <= next_t)]
        if len(period) == 0:
            continue
        period_rets = rets.loc[period, holdings].mean(axis=1).fillna(0.0)
        portfolio_ret.loc[period] = period_rets.values

        # 换手成本近似
        if prev_holdings is not None:
            turnover = len(set(holdings) ^ set(prev_holdings)) / max(top_n, 1)
            portfolio_ret.loc[t] -= turnover * cost_rate
        prev_holdings = holdings

    return portfolio_ret


# ---------------------------------------------------------------------------
# 指标与组合合并
# ---------------------------------------------------------------------------

def annual_return(returns: pd.Series, periods: int = 252) -> float:
    """由日收益序列计算年化收益（几何）。"""
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod() - 1.0)
    years = len(returns) / periods
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else -1.0


def max_drawdown(returns: pd.Series) -> float:
    """由日收益序列计算最大回撤（负值）。"""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """年化夏普比率。"""
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(periods))


def summary(returns: pd.Series) -> dict[str, float]:
    """汇总一个收益序列的核心指标。"""
    return {
        "annual_return": annual_return(returns),
        "max_drawdown": max_drawdown(returns),
        "sharpe": sharpe_ratio(returns),
    }


def market_benchmark(close: pd.DataFrame) -> pd.Series:
    """全市场等权日收益，作为市场牛熊的代理基准（无需外部指数数据，离线可用）。"""
    return close.pct_change().mean(axis=1)


def apply_market_timing(
    returns: pd.Series,
    benchmark: pd.Series,
    ma_window: int = 60,
    bear_exposure: float = 0.3,
) -> pd.Series:
    """对收益序列应用大盘择时：基准净值跌破 N 日均线时降仓到 ``bear_exposure``。

    用 T-1 日及以前的信息决定 T 日仓位（``shift(1)``），严格无未来函数。
    ``returns`` 与 ``benchmark`` 需对齐到同一交易日索引。
    """
    equity = (1.0 + benchmark.fillna(0.0)).cumprod()
    ma = equity.rolling(ma_window).mean()
    exposure = pd.Series(1.0, index=benchmark.index)
    exposure[equity < ma] = bear_exposure
    exposure = exposure.shift(1).fillna(1.0)  # 用前一日信号，避免未来函数
    aligned = returns.reindex(exposure.index).fillna(0.0)
    return aligned * exposure


def combine_returns(returns: dict[str, pd.Series], method: str = "inverse_vol") -> pd.Series:
    """合并多个策略的日度收益序列。

    ``method``：
      - ``equal``：等权平均；
      - ``inverse_vol``：按历史波动率倒数加权（低波动多配，相关性不高时近似风险平价）；
      - ``min_drawdown``：按最大回撤倒数加权（回撤更小的策略占更高权重，偏防守）。
    """
    frame = pd.DataFrame(returns).dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float)
    if method == "equal":
        weights = pd.Series(1.0 / len(frame.columns), index=frame.columns)
    elif method == "inverse_vol":
        vol = frame.std()
        inv = 1.0 / vol.replace(0.0, np.nan)
        weights = inv / inv.sum()
    elif method == "min_drawdown":
        dd = frame.apply(lambda s: abs(max_drawdown(s)))
        inv = 1.0 / dd.replace(0.0, np.nan)
        weights = inv / inv.sum()
    else:
        raise ValueError(f"不支持的组合方式：{method}")
    combined = (frame.fillna(0.0) * weights).sum(axis=1)
    return combined
