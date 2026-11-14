"""主流技术面因子库：基于日线 OHLCV 的可复用计算与截面标准化。

每个因子函数接收一个日线 DataFrame（含 open/high/low/close/volume/amount/pre_close，
``.iloc[-1]`` 为最新交易日），返回 ``float | None``；历史不足时返回 ``None``。
因子原始值在横截面 z-score 后由策略按带符号权重合成，权重符号即代表方向
（正=原始值越高越好，负=越低越好）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _momentum(frame: pd.DataFrame, window: int = 60) -> float | None:
    """动量：最新收盘价相对 window 日前收盘价的涨跌幅（越大越强）。"""
    if len(frame) < window + 1:
        return None
    base = float(frame["close"].iloc[-window - 1])
    current = float(frame["close"].iloc[-1])
    if base <= 0:
        return None
    return current / base - 1.0


def _reversal(frame: pd.DataFrame, window: int = 5) -> float | None:
    """短期反转：取 window 日动量的相反数（超涨回落、超跌反弹）。"""
    momentum = _momentum(frame, window)
    return None if momentum is None else -momentum


def _volatility(frame: pd.DataFrame, window: int = 20) -> float | None:
    """波动率：近 window 日日收益率的标准差（越低越稳）。"""
    if len(frame) < window + 1:
        return None
    returns = frame["close"].astype(float).pct_change().dropna().tail(window)
    if len(returns) < window:
        return None
    return float(returns.std(ddof=0))


def _liquidity(frame: pd.DataFrame, window: int = 20) -> float | None:
    """流动性：近 window 日平均成交额（元）。"""
    if len(frame) < window:
        return None
    return float(np.nanmean(frame["amount"].astype(float).tail(window).to_numpy()))


def _rsi(frame: pd.DataFrame, window: int = 14) -> float | None:
    """RSI 相对强弱指标：0~100，超买偏高、超卖偏低。"""
    if len(frame) < window + 1:
        return None
    closes = frame["close"].astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + last_gain / last_loss))


def _macd(frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    """MACD 柱（DIF−DEA）相对现价归一化，衡量趋势动能强弱。"""
    if len(frame) < slow + signal:
        return None
    closes = frame["close"].astype(float)
    dif = closes.ewm(span=fast, adjust=False).mean() - closes.ewm(span=slow, adjust=False).mean()
    hist = dif - dif.ewm(span=signal, adjust=False).mean()
    current = float(closes.iloc[-1])
    return float(hist.iloc[-1]) / current if current else None


def _ma_trend(frame: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 60)) -> float | None:
    """均线多头排列比例：短均线高于长均线的对数占比（0~1）。"""
    if len(frame) < max(windows):
        return None
    averages = [float(frame["close"].astype(float).tail(w).mean()) for w in windows]
    pairs = list(zip(averages, averages[1:]))
    return sum(1 for short, long in pairs if short > long) / len(pairs)


def _high_52w(frame: pd.DataFrame, window: int = 250) -> float | None:
    """距 52 周（window 日）新高距离：最新价 / 区间最高价 − 1（越接近 0 越强）。"""
    if len(frame) < 2:
        return None
    highest = float(frame["high"].astype(float).tail(min(window, len(frame))).max())
    current = float(frame["close"].iloc[-1])
    if highest <= 0:
        return None
    return current / highest - 1.0


def _atr(frame: pd.DataFrame, window: int = 14) -> float | None:
    """ATR 真实波幅均值相对现价归一化，衡量波动幅度。"""
    if len(frame) < window + 1:
        return None
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    current = float(close.iloc[-1])
    return float(true_range.tail(window).mean()) / current if current else None


def _amihud(frame: pd.DataFrame, window: int = 20) -> float | None:
    """Amihud 非流动性：近 window 日 |收益率|/成交额 的均值（越大越不流动）。"""
    if len(frame) < window + 1:
        return None
    amount = frame["amount"].astype(float).replace(0.0, np.nan)
    ratio = frame["close"].astype(float).pct_change().abs() / amount
    ratio = ratio.dropna()
    if len(ratio) < window:
        return None
    return float(ratio.tail(window).mean())


# ── 振荡器 / 超买超卖 ──
def _kdj(frame: pd.DataFrame, n: int = 9, k: int = 3, d: int = 3) -> float | None:
    """KDJ 的 J 值 = 3K−2D，超买偏高（>100）、超卖偏低（<0）。"""
    if len(frame) < n + max(k, d):
        return None
    close = frame["close"].astype(float)
    low_min = frame["low"].astype(float).rolling(n).min()
    high_max = frame["high"].astype(float).rolling(n).max()
    denom = high_max - low_min
    rsv = (close - low_min) / denom.replace(0.0, np.nan) * 100.0
    k_line = rsv.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    last = 3.0 * float(k_line.iloc[-1]) - 2.0 * float(d_line.iloc[-1])
    return last if math.isfinite(last) else None


def _cci(frame: pd.DataFrame, window: int = 14) -> float | None:
    """CCI 顺势指标：典型价格对均值偏差 / 平均绝对偏差（0.015 缩放）。"""
    if len(frame) < window + 1:
        return None
    tp = (frame["high"].astype(float) + frame["low"].astype(float) + frame["close"].astype(float)) / 3.0
    tp_ma = tp.rolling(window).mean()
    mad = (tp - tp_ma).abs().rolling(window).mean()
    last_mad = float(mad.iloc[-1])
    if last_mad == 0:
        return None
    return (float(tp.iloc[-1]) - float(tp_ma.iloc[-1])) / (0.015 * last_mad)


def _williams_r(frame: pd.DataFrame, window: int = 14) -> float | None:
    """威廉指标 WR：-100~0，接近 0 超买、接近 -100 超卖。"""
    if len(frame) < window:
        return None
    high_max = float(frame["high"].astype(float).rolling(window).max().iloc[-1])
    low_min = float(frame["low"].astype(float).rolling(window).min().iloc[-1])
    if high_max == low_min:
        return None
    return (float(frame["close"].iloc[-1]) - high_max) / (high_max - low_min) * 100.0


def _boll_pctb(frame: pd.DataFrame, window: int = 20, k: float = 2.0) -> float | None:
    """布林带 %B 位置：0~1（>1 在上轨上方、<0 在下轨下方）。"""
    if len(frame) < window:
        return None
    close = frame["close"].astype(float)
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    last_std = float(std.iloc[-1])
    if last_std == 0:
        return None
    return (float(close.iloc[-1]) - float(ma.iloc[-1])) / (k * last_std)


def _boll_width(frame: pd.DataFrame, window: int = 20, k: float = 2.0) -> float | None:
    """布林带宽 = 2k·σ/中轨（越低=压缩越紧，常为变盘前兆）。"""
    if len(frame) < window:
        return None
    close = frame["close"].astype(float)
    ma = float(close.rolling(window).mean().iloc[-1])
    if ma == 0:
        return None
    return (2.0 * k * float(close.rolling(window).std().iloc[-1])) / ma


# ── 趋势 / 动量 ──
def _bias(frame: pd.DataFrame, window: int = 5) -> float | None:
    """乖离率：现价对 window 日均线的偏离（BIAS）。"""
    if len(frame) < window:
        return None
    close = frame["close"].astype(float)
    ma = float(close.rolling(window).mean().iloc[-1])
    if ma == 0:
        return None
    return float(close.iloc[-1]) / ma - 1.0


def _momentum_skip(frame: pd.DataFrame, window: int = 60, skip: int = 20) -> float | None:
    """动量（剔除近端）：跳过最近 skip 日，取更早 window 日的动量，规避短期反转噪声。"""
    if len(frame) < window + 1:
        return None
    current = float(frame["close"].iloc[-1 - skip])
    base = float(frame["close"].iloc[-1 - window])
    if base <= 0:
        return None
    return current / base - 1.0


# ── 量价配合 ──
def _obv_slope(frame: pd.DataFrame, window: int = 20) -> float | None:
    """OBV 净量比：近 window 日净流入量 / 总成交量（−1~1，量价方向确认）。"""
    if len(frame) < window + 1:
        return None
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    sign = (close.diff() > 0).astype(float) - (close.diff() < 0).astype(float)
    obv = (sign * volume).cumsum()
    avg_vol = float(volume.rolling(window).mean().iloc[-1])
    if avg_vol == 0:
        return None
    return float(obv.diff(window).iloc[-1]) / (window * avg_vol)


def _mfi(frame: pd.DataFrame, window: int = 14) -> float | None:
    """资金流量指标 MFI：正/负资金流之比的 0~100 映射，量价共振。"""
    if len(frame) < window + 1:
        return None
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    pos = (close.diff().clip(lower=0) * volume).rolling(window).sum()
    neg = ((-close.diff()).clip(lower=0) * volume).rolling(window).sum()
    last_pos, last_neg = float(pos.iloc[-1]), float(neg.iloc[-1])
    if last_neg == 0:
        return 100.0 if last_pos > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + last_pos / last_neg)


def _amount_ratio(frame: pd.DataFrame, window: int = 5) -> float | None:
    """成交额比：当日成交额 / 近 window 日均额（比量比更稳健的资金活跃代理）。"""
    if len(frame) < window:
        return None
    amount = frame["amount"].astype(float)
    ma = float(amount.rolling(window).mean().iloc[-1])
    if ma == 0:
        return None
    return float(amount.iloc[-1]) / ma


def _vol_stability(frame: pd.DataFrame, window: int = 20) -> float | None:
    """量稳度：近 window 日成交量变异系数（越低量能越稳，对应「量稳换手率」因子思路）。"""
    if len(frame) < window:
        return None
    volume = frame["volume"].astype(float)
    ma = float(volume.rolling(window).mean().iloc[-1])
    if ma == 0:
        return None
    return float(volume.rolling(window).std().iloc[-1]) / ma


# ── 风险调整 / 区间位置 ──
def _sharpe(frame: pd.DataFrame, window: int = 60) -> float | None:
    """风险调整动量：近 window 日收益均值 / 标准差（夏普代理）。"""
    if len(frame) < window + 1:
        return None
    returns = frame["close"].astype(float).pct_change()
    std = float(returns.rolling(window).std().iloc[-1])
    if std == 0:
        return None
    return float(returns.rolling(window).mean().iloc[-1]) / std


def _downside_vol(frame: pd.DataFrame, window: int = 20) -> float | None:
    """下行波动率：只对下跌收益取标准差（Sortino 式，越低越好）。"""
    if len(frame) < window + 1:
        return None
    downside = frame["close"].astype(float).pct_change().clip(upper=0)
    return float(downside.rolling(window).std().iloc[-1])


def _range_position(frame: pd.DataFrame, window: int = 20) -> float | None:
    """区间价格位置：现价在近 window 日高低区间中的相对位置（0~1）。"""
    if len(frame) < window:
        return None
    high_max = float(frame["high"].astype(float).rolling(window).max().iloc[-1])
    low_min = float(frame["low"].astype(float).rolling(window).min().iloc[-1])
    if high_max == low_min:
        return None
    return (float(frame["close"].iloc[-1]) - low_min) / (high_max - low_min)


def _updown_ratio(frame: pd.DataFrame, window: int = 20) -> float | None:
    """上涨力量占比：近 window 日上涨幅度合计 / 涨跌幅度合计（0~1）。"""
    if len(frame) < window + 1:
        return None
    diff = frame["close"].astype(float).diff()
    total = float(diff.abs().rolling(window).sum().iloc[-1])
    if total == 0:
        return None
    return float(diff.clip(lower=0).rolling(window).sum().iloc[-1]) / total


FACTORS: dict[str, dict[str, Any]] = {
    "momentum": {"func": _momentum, "label": "动量", "defaults": {"window": 60}},
    "reversal": {"func": _reversal, "label": "短期反转", "defaults": {"window": 5}},
    "volatility": {"func": _volatility, "label": "波动率", "defaults": {"window": 20}},
    "liquidity": {"func": _liquidity, "label": "流动性", "defaults": {"window": 20}},
    "rsi": {"func": _rsi, "label": "RSI", "defaults": {"window": 14}},
    "macd": {"func": _macd, "label": "MACD", "defaults": {"fast": 12, "slow": 26, "signal": 9}},
    "ma_trend": {"func": _ma_trend, "label": "均线多头", "defaults": {"windows": (5, 10, 20, 60)}},
    "high_52w": {"func": _high_52w, "label": "52周新高", "defaults": {"window": 250}},
    "atr": {"func": _atr, "label": "真实波幅", "defaults": {"window": 14}},
    "amihud": {"func": _amihud, "label": "非流动性", "defaults": {"window": 20}},
    "kdj": {"func": _kdj, "label": "KDJ·J值", "defaults": {"n": 9, "k": 3, "d": 3}},
    "cci": {"func": _cci, "label": "CCI", "defaults": {"window": 14}},
    "williams_r": {"func": _williams_r, "label": "威廉WR", "defaults": {"window": 14}},
    "boll_pctb": {"func": _boll_pctb, "label": "布林%B", "defaults": {"window": 20, "k": 2.0}},
    "boll_width": {"func": _boll_width, "label": "布林带宽", "defaults": {"window": 20, "k": 2.0}},
    "bias": {"func": _bias, "label": "乖离率", "defaults": {"window": 5}},
    "momentum_skip": {"func": _momentum_skip, "label": "动量(剔近端)", "defaults": {"window": 60, "skip": 20}},
    "obv_slope": {"func": _obv_slope, "label": "OBV净量比", "defaults": {"window": 20}},
    "mfi": {"func": _mfi, "label": "资金流量MFI", "defaults": {"window": 14}},
    "amount_ratio": {"func": _amount_ratio, "label": "成交额比", "defaults": {"window": 5}},
    "vol_stability": {"func": _vol_stability, "label": "量稳度", "defaults": {"window": 20}},
    "sharpe": {"func": _sharpe, "label": "风险调整动量", "defaults": {"window": 60}},
    "downside_vol": {"func": _downside_vol, "label": "下行波动", "defaults": {"window": 20}},
    "range_position": {"func": _range_position, "label": "区间位置", "defaults": {"window": 20}},
    "updown_ratio": {"func": _updown_ratio, "label": "上涨力量", "defaults": {"window": 20}},
}


def compute_factor(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> float | None:
    """计算单个因子，允许用 ``{因子名}_{参数名}`` 覆盖注册表默认值（如 ``momentum_window``）。"""
    spec = FACTORS[name]
    kwargs = dict(spec["defaults"])
    for key in spec["defaults"]:
        override = (params or {}).get(f"{name}_{key}")
        if override is not None:
            kwargs[key] = override
    return spec["func"](frame, **kwargs)


def cross_sectional_rank_score(
    frames_by_code: dict[str, pd.DataFrame],
    factor_specs: Sequence[tuple[str, float]],
    params: dict[str, Any] | None = None,
) -> list[tuple[str, float, str]]:
    """对全部标的计算因子、横截面 z-score 并按带符号权重合成评分。

    返回按评分降序的 ``[(code, score, reason), ...]``。实盘信号生成与回测共用此入口，
    保证两者对同一策略的排序一致。``min_average_amount`` 用于过滤成交额过低的标的。
    """
    params = params or {}
    factor_names = [name for name, _weight in factor_specs]
    min_amount = float(params.get("min_average_amount", 0.0))

    records: dict[str, dict[str, float]] = {}
    for code, frame in frames_by_code.items():
        if frame is None or frame.empty:
            continue
        average_amount = float(np.nanmean(frame["amount"].astype(float).tail(20).to_numpy())) if len(frame) else 0.0
        if average_amount < min_amount:
            continue
        values = {name: compute_factor(name, frame, params) for name in factor_names}
        if any(value is None or not math.isfinite(value) for value in values.values()):
            continue
        values["_amount"] = average_amount
        records[code] = values

    if not records:
        return []

    raw = pd.DataFrame(records).T
    z = pd.DataFrame(index=raw.index)
    for name in factor_names:
        std = float(raw[name].std(ddof=0))
        z[name] = 0.0 if std == 0 else (raw[name] - raw[name].mean()) / std

    score = sum(weight * z[name] for name, weight in factor_specs)
    ranked: list[tuple[str, float, str]] = []
    for code in raw.index:
        reason = "，".join(
            f"{FACTORS[name]['label']} z={float(z.loc[code, name]):+.2f}"
            for name, _weight in factor_specs
        )
        ranked.append((str(code), float(score.loc[code]), reason))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked
