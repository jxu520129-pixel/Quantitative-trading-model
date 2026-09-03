"""涨停回马枪·冲高回调低吸策略的日线近似回测。

对应《涨停冲高回调低吸策略.md》。形态链路：涨停启动(T) → 次日冲高确认(H) →
缩量回调(H+1~H+8) → 双信号止跌(①缩量十字星/小阳 + ②放量阳线收复5日线) → 尾盘低吸。

日线无法回填的数据按文档 §15 口径近似或跳过，逐项标注：
- 首封时间/开板次数（§4.4）与换手率（§4.5）：需分时数据，不参与；排序中"封板强度"
  以连板数近似（2板=30、首板=18 的中性替代）；
- 流通市值 30~300 亿（§3.6）：历史市值不可得，不参与；
- 炸板率 ≤40%（§11.1）：需盘中统计，不参与；大盘闸门其余四项（上证≥MA20、
  前日涨停≥50家、前日跌停≤10家且≤涨停家数、上证单日跌≥2%熔断）全部落地，
  指数日线由 Tushare 代理 index_daily 提供，缺失时降级为仅用市场涨跌停统计；
- 买入价用信号②日收盘（14:45~14:57 尾盘买入的近似）并计滑点；
- 回测池仅沪深主板（60x/000/001/002/003），与文档 §3 一致。
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .hot_strategy import _limit_streak, _load_panels
from .market_rules import TradingCosts, board_price_limit
from .models import utc_now_text

LOG = logging.getLogger(__name__)

LIMIT_STRATEGY_NAME = "涨停回马枪·冲高回调低吸"

_MAIN_BOARD_PREFIXES = ("60", "000", "001", "002", "003")  # 文档 §3.1 已确认
_MIN_HISTORY_BARS = 60  # §3.3 次新剔除
_MIN_PRICE = 3.0  # §3.5 已确认
_MAX_STREAK = 2  # §4.2 首板为主、允许 2 连板


def _load_index_daily(lookback_start: str, end_date: str) -> pd.DataFrame | None:
    """上证指数日线（大盘闸门用）。Tushare 代理 index_daily，失败返回 None。"""
    from .data.providers import tushare_pro_from_env

    pro = tushare_pro_from_env()
    if pro is None:
        return None
    try:
        frame = pro.index_daily(
            ts_code="000001.SH",
            start_date=lookback_start.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
    except Exception as error:
        LOG.warning("上证指数日线获取失败，大盘闸门趋势项降级：%s", error)
        return None
    if frame is None or frame.empty:
        return None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").set_index("trade_date")[["close", "pct_chg"]]


def _detect_pullback_events(panels: dict[str, pd.DataFrame], min_surge: float = 1.05) -> list[dict[str, Any]]:
    """向量检测 (T, H) 形态事件：T 收盘封死涨停(连板≤2、非一字) → H 冲高≥min_surge 且创20日新高、未再涨停。"""
    close, open_, high, low, volume = panels["close"], panels["open"], panels["high"], panels["low"], panels["volume"]
    pre_close = close.shift(1)
    limit_pct = pd.Series({code: board_price_limit(code, is_st=False) for code in close.columns})
    limit_up_price = (pre_close * (1 + limit_pct)).round(2)
    is_limit = close >= (limit_up_price - 1e-6)
    streak = _limit_streak(is_limit)
    not_one_word = low < (limit_up_price - 1e-6)  # §4.3 一字板：最低价=涨停价
    bars = close.notna().cumsum()

    h_high = high.shift(-1)
    h_vol = volume.shift(-1)
    h_is_limit = is_limit.shift(-1, fill_value=False)
    high20 = high.rolling(20, min_periods=20).max()

    mask = (
        is_limit
        & (streak >= 1) & (streak <= _MAX_STREAK)
        & not_one_word
        & (close >= _MIN_PRICE)
        & (bars >= _MIN_HISTORY_BARS)
        & (h_high >= close * min_surge)  # §5.2 冲高≥min_surge（默认 5%）
        & (h_high > high20)  # §5.3 创20日新高
        & ~h_is_limit  # §5.4 H 再涨停则形态升级，放弃
    )

    events: list[dict[str, Any]] = []
    dates = list(close.index)
    codes = list(close.columns)
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    vol_arr = volume.to_numpy()
    streak_arr = streak.to_numpy()
    for code_idx in range(len(codes)):
        code = codes[code_idx]
        col = mask[code].to_numpy()
        for t_idx in np.nonzero(col)[0]:
            h_idx = t_idx + 1
            if h_idx >= len(dates) or np.isnan(close_arr[h_idx, code_idx]):
                continue  # H 停牌，形态中断
            events.append({
                "code": code, "code_idx": code_idx, "t_di": int(t_idx), "h_di": int(h_idx),
                "s1_low": float(low_arr[t_idx, code_idx]),  # 支撑 S1 = 涨停日最低价
                "t_close": float(close_arr[t_idx, code_idx]),
                "h_high": float(high_arr[h_idx, code_idx]),
                "h_vol": float(vol_arr[h_idx, code_idx]),
                "streak": int(streak_arr[t_idx, code_idx]),
            })
    return events


def _scan_entry_signals(
    panels: dict[str, pd.DataFrame],
    events: list[dict[str, Any]],
    industry_map: dict[str, str],
    industry_rank: pd.DataFrame,
    sig2_vol_ratio: float = 2.0,
    sig2_gain_pct: float = 0.02,
    sig1_shrink: float = 0.4,
) -> dict[int, list[dict[str, Any]]]:
    """逐事件走回调观察窗（H 后 8 根有效 K 线），产出信号②买入日信号与打分。

    sig2_vol_ratio：信号②放量倍数（当日成交量 ≥ 前一日 × 该值）；sig2_gain_pct：信号②最低涨幅；
    sig1_shrink：信号①缩量门槛（当日成交量 ≤ H 日成交量 × 该值）。
    """
    close_a = panels["close"].to_numpy()
    open_a = panels["open"].to_numpy()
    high_a = panels["high"].to_numpy()
    low_a = panels["low"].to_numpy()
    vol_a = panels["volume"].to_numpy()
    ma5 = panels["close"].rolling(5, min_periods=5).mean().to_numpy()
    ma20 = panels["close"].rolling(20, min_periods=20).mean().to_numpy()
    pre_close = panels["close"].shift(1).to_numpy()
    dates = list(panels["close"].index)

    signals: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        ci, h_di = ev["code_idx"], ev["h_di"]
        s1, t_close, h_high, h_vol = ev["s1_low"], ev["t_close"], ev["h_high"], ev["h_vol"]
        n_days = close_a.shape[0]
        pullback_vols: list[float] = []
        sig1_di, sig1_high = None, None
        min_low = h_high  # 回撤以 H 日最高为基准
        k = 0
        d = h_di + 1
        while d < n_days and k < 8:
            c = close_a[d, ci]
            if np.isnan(c):
                d += 1
                continue  # 停牌不计入回调天数（文档 §13）
            k += 1
            o, hi, lo, v = open_a[d, ci], high_a[d, ci], low_a[d, ci], vol_a[d, ci]
            pc = pre_close[d, ci]
            pct = c / pc - 1 if pc and not np.isnan(pc) else 0.0
            limit_down_price = round(pc * 0.9, 2) if not np.isnan(pc) else 0.0
            # §6.3/§6.5 支撑破位与恐慌下跌 → 放弃
            if c < s1 or c < ma20[d, ci] or pct <= -0.07 or (limit_down_price > 0 and c <= limit_down_price + 1e-6):
                break
            min_low = min(min_low, lo)
            pullback = (h_high - min_low) / h_high if h_high > 0 else 0.0
            if pullback > 0.15:
                break  # §6.2 深回调按出货处理
            # §7 信号①：H+2 起的缩量十字星/小阳
            if sig1_di is None and k >= 2:
                cross = abs(c - o) / pc <= 0.005 if pc else False
                small_yang = 0 < pct <= 0.03
                shrink = v <= h_vol * sig1_shrink
                if (cross or small_yang) and shrink:
                    sig1_di, sig1_high = d, hi
            # §7 信号②：与①同日或其后 1~2 个有效交易日
            if sig1_di is not None and d - sig1_di <= 2 and c / pre_close[d, ci] - 1 >= sig2_gain_pct and c > o:
                if v >= vol_a[d - 1, ci] * sig2_vol_ratio and c >= ma5[d, ci] and c >= high_a[sig1_di, ci]:
                    # §6.4 回调期日均缩量（信号日之前的回调 K 线）
                    if pullback_vols and float(np.mean(pullback_vols)) > h_vol * 0.8:
                        break
                    if pullback < 0.05:
                        # §6.2 回调不充分：不出手，继续观察
                        pullback_vols.append(v)
                        d += 1
                        continue
                    # §8 打分：封板强度(连板近似) + 缩量程度 + 板块热度 + 回撤质量
                    strength = 30.0 if ev["streak"] >= 2 else 18.0
                    ratio = vol_a[sig1_di, ci] / h_vol if h_vol > 0 else 1.0
                    shrink_score = 30.0 if ratio <= 0.25 else 25.0 if ratio <= 0.35 else 20.0 if ratio <= 0.45 else 12.0
                    sector = industry_map.get(ev["code"], "")
                    heat = 8.0
                    if sector and sector in industry_rank.columns and d < len(industry_rank):
                        rank = industry_rank[sector].iloc[d]
                        total = industry_rank.shape[1]
                        if not np.isnan(rank) and total > 0 and rank <= total * 0.3:
                            heat = 30.0
                        elif not np.isnan(rank) and total > 0 and rank <= total * 0.6:
                            heat = 18.0
                    quality = 10.0 if 0.08 <= pullback <= 0.12 else 6.0
                    score = strength + shrink_score + heat + quality
                    signals.setdefault(dates[d], []).append({
                        "code": ev["code"], "score": score, "h_high": h_high,
                        "pullback": pullback, "streak": ev["streak"],
                        "sig1_low": float(low_a[sig1_di, ci]) if sig1_di is not None else float(min_low),
                    })
                    break  # 该形态只交易一次
            pullback_vols.append(v)
            d += 1
    return signals


def scan_limit_pullback_signals(
    database: Any,
    end_date: str | None = None,
    min_limit_up: int = 50,
    sig2_vol_ratio: float = 2.0,
    sig2_gain_pct: float = 0.02,
    sig1_shrink: float = 0.4,
    min_surge: float = 1.05,
) -> list[dict[str, Any]]:
    """扫描最新交易日涨停回马枪买入信号（供模拟盘 lab_signal 引擎复用）。

    与 :func:`run_limit_pullback_backtest` 同源复用形态检测与打分，但只返回
    「最新交易日」的买入信号（含 code/name/score/h_high/pullback/streak），
    不涉及资金与持仓管理。大盘闸门仅保留「前日涨停家数 ≥ min_limit_up」一项，
    弱势时返回空列表。
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    industry_rows = database.query_all("SELECT code,industry FROM stock_industry")
    industry_map = {r["code"]: r["industry"] for r in industry_rows}

    universe = [
        item for item in database.query_all(
            "SELECT code,name FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        )
        if item["code"].startswith(_MAIN_BOARD_PREFIXES)
    ]
    if not universe:
        return []
    name_map = {item["code"]: item["name"] for item in universe}
    lookback_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    panels = _load_panels(database, universe, lookback_start, end_date)
    if not panels:
        return []
    close = panels["close"]

    # 行业当日涨幅排名（板块热度打分用，等权均值近似板块涨幅）
    pct = close / close.shift(1) - 1
    sectors = sorted({v for v in industry_map.values() if v})
    industry_rank = pd.DataFrame(index=close.index)
    if sectors:
        ind_pos = {name: i for i, name in enumerate(sectors)}
        onehot = np.zeros((len(close.columns), len(sectors)), dtype=np.float32)
        for j, code in enumerate(close.columns):
            pos_ = ind_pos.get(industry_map.get(code, ""))
            if pos_ is not None:
                onehot[j, pos_] = 1.0
        den = pct.notna().astype(np.float32).to_numpy() @ onehot
        num = pct.fillna(0.0).to_numpy(dtype=np.float32) @ onehot
        ind_mean = pd.DataFrame(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), index=close.index, columns=sectors)
        industry_rank = ind_mean.rank(axis=1, ascending=False)

    # 大盘闸门：前日涨停家数（沪深全市场口径）
    sentiment_universe = [
        item for item in database.query_all(
            "SELECT code FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        )
        if not item["code"].startswith(("43", "83", "87", "88", "920"))
    ]
    sent_close = _load_panels(database, sentiment_universe, lookback_start, end_date).get("close")
    if sent_close is None or sent_close.empty:
        sent_close = close
    sent_pre = sent_close.shift(1)
    sent_limit_pct = pd.Series({code: board_price_limit(code, is_st=False) for code in sent_close.columns})
    limit_up_n = (sent_close >= (sent_pre * (1 + sent_limit_pct)).round(2) - 1e-6).sum(axis=1)

    events = _detect_pullback_events(panels, min_surge)
    signals = _scan_entry_signals(panels, events, industry_map, industry_rank, sig2_vol_ratio, sig2_gain_pct, sig1_shrink)
    if not signals:
        return []

    # 取最新交易日，并应用大盘闸门（用前一日涨停家数）
    latest_date = max(signals.keys())
    prev_limit_up = 0
    if latest_date in sent_close.index:
        pos = sent_close.index.get_loc(latest_date)
        if pos > 0:
            prev_limit_up = int(limit_up_n.iloc[pos - 1])
    if prev_limit_up < min_limit_up:
        return []

    result: list[dict[str, Any]] = []
    for sig in signals[latest_date]:
        result.append({
            "code": sig["code"],
            "name": name_map.get(sig["code"], sig["code"]),
            "score": sig["score"],
            "h_high": sig["h_high"],
            "pullback": sig["pullback"],
            "streak": sig["streak"],
        })
    return result


def run_limit_pullback_backtest(
    database: Any,
    data_service: Any,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    threshold: float = 70.0,  # 打分阈值；回测扫描 68~70 为甜点区（年化最优），50 偏松会混入诱多
    top_n: int = 3,
    max_positions: int = 3,
    initial_cash: float = 1_000_000.0,
    daily_budget: float = 0.4,  # §11.3 单日新增仓位 ≤40%
    persist: bool = True,
    use_sig1_stop: bool = False,  # True=止损用信号①止跌低点破位（贴合低吸形态）；False=固定比例
    stop_loss: float = 0.06,  # 硬止损比例
    trailing_stop: float = 0.05,  # 移动止盈回撤比例（仅在浮盈后生效）
    sig2_vol_ratio: float = 2.0,  # 信号②放量倍数（≥前一日成交量 × 该值）；2.0 为回测最优，1.5 偏松会混入诱多
    sig2_gain_pct: float = 0.02,  # 信号②最低涨幅
    min_limit_up: int = 50,  # 大盘闸门：前一日两市涨停家数下限（扫描 60/70/80 回撤降但年化更差，50 最优）
    entry_mode: str = "close",  # close=信号②日尾盘收盘买入；next_open=次日开盘买入（回测更差，仅保留对比）
    limit_break_exit: bool = True,  # True=涨停次日断板即了结；False=断板后继续持有用移动止盈追踪
    min_surge: float = 1.05,  # H 日冲高确认阈值（≥ T 收盘 × 该值）
    sig1_shrink: float = 0.4,  # 信号①缩量门槛（当日成交量 ≤ H 日 × 该值）；0.4 为回测最优（缩量越充分洗盘越彻底），0.5 偏松
) -> dict[str, Any]:
    """涨停回马枪日线近似回测。返回指标 dict 并默认落库至因子实验室表。"""
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    industry_rows = database.query_all("SELECT code,industry FROM stock_industry")
    industry_map = {r["code"]: r["industry"] for r in industry_rows}

    universe = [
        item for item in database.query_all(
            "SELECT code,name FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        )
        if item["code"].startswith(_MAIN_BOARD_PREFIXES)
    ]
    lookback_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    panels = _load_panels(database, universe, lookback_start, end_date)
    if not panels:
        raise ValueError("日线数据为空，请先执行 update-data")
    close, open_, high, low = panels["close"], panels["open"], panels["high"], panels["low"]

    # 行业当日涨幅排名（板块热度打分用，等权均值近似板块涨幅）
    pct = close / close.shift(1) - 1
    sectors = sorted({v for v in industry_map.values() if v})
    industry_rank = pd.DataFrame(index=close.index)
    if sectors:
        ind_pos = {name: i for i, name in enumerate(sectors)}
        onehot = np.zeros((len(close.columns), len(sectors)), dtype=np.float32)
        for j, code in enumerate(close.columns):
            pos_ = ind_pos.get(industry_map.get(code, ""))
            if pos_ is not None:
                onehot[j, pos_] = 1.0
        den = pct.notna().astype(np.float32).to_numpy() @ onehot
        num = pct.fillna(0.0).to_numpy(dtype=np.float32) @ onehot
        ind_mean = pd.DataFrame(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), index=close.index, columns=sectors)
        industry_rank = ind_mean.rank(axis=1, ascending=False)

    # 大盘闸门（§11.1）：指数项由 Tushare 代理提供，市场涨跌停统计用沪深全市场口径
    index_daily = _load_index_daily(lookback_start, end_date)
    if index_daily is None:
        LOG.warning("上证指数数据缺失，闸门的趋势与熔断项不生效（仅涨跌停统计项生效）")
    sentiment_universe = [
        item for item in database.query_all(
            "SELECT code FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        )
        if not item["code"].startswith(("43", "83", "87", "88", "920"))
    ]
    sent_close = _load_panels(database, sentiment_universe, lookback_start, end_date).get("close")
    if sent_close is None or sent_close.empty:
        sent_close = close
    sent_pre = sent_close.shift(1)
    sent_limit_pct = pd.Series({code: board_price_limit(code, is_st=False) for code in sent_close.columns})
    limit_up_n = (sent_close >= (sent_pre * (1 + sent_limit_pct)).round(2) - 1e-6).sum(axis=1)
    limit_down_n = (sent_close <= (sent_pre * (1 - sent_limit_pct)).round(2) + 1e-6).sum(axis=1)

    events = _detect_pullback_events(panels, min_surge)
    LOG.info("涨停回马枪：检测到 (T,H) 形态事件 %s 个", len(events))
    signals = _scan_entry_signals(panels, events, industry_map, industry_rank, sig2_vol_ratio, sig2_gain_pct, sig1_shrink)

    dates = [d for d in close.index if start_date <= d.strftime("%Y-%m-%d") <= end_date]
    costs = TradingCosts()
    cash = initial_cash
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[tuple[str, float]] = []
    name_map = {item["code"]: item["name"] for item in universe}
    cooldown_until: pd.Timestamp | None = None
    consecutive_stops = 0
    week_peak: dict[tuple[int, int], float] = {}
    week_blocked: set[tuple[int, int]] = set()

    def _equity_on(date) -> float:
        total = cash
        for code, pos in positions.items():
            price = close.at[date, code]
            if pd.isna(price):
                series = close[code].loc[:date].dropna()
                price = float(series.iloc[-1]) if not series.empty else pos["entry_price"]
            total += pos["shares"] * float(price)
        return total

    def _sell(code: str, pos: dict[str, Any], date, shares_to_sell: int, price: float, reason: str, held_days: int) -> None:
        nonlocal cash, consecutive_stops
        fill = costs.slipped_price(price, is_buy=False)
        amount = fill * shares_to_sell
        cash += amount - costs.commission(amount) - costs.stamp_duty(amount, is_sell=True)
        sold_cost = pos["cost"] * (shares_to_sell / pos["shares"])
        pnl = fill * shares_to_sell - sold_cost
        trades.append({
            "code": code, "name": name_map.get(code, code), "mode": "低吸", "score": pos["score"],
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"), "entry_price": round(pos["entry_price"], 3),
            "shares": shares_to_sell, "exit_date": date.strftime("%Y-%m-%d"), "exit_price": round(fill, 3),
            "pnl": round(pnl, 2), "pnl_pct": round(fill / pos["entry_price"] - 1, 4),
            "holding_days": held_days, "reason": reason,
        })
        if "止损" in reason and pnl < 0:
            consecutive_stops += 1
        else:
            consecutive_stops = 0
        pos["shares"] -= shares_to_sell
        pos["cost"] -= sold_cost

    for i, date in enumerate(dates):
        week_key = date.isocalendar()[:2]
        d_str = date.strftime("%Y-%m-%d")
        ma20_idx = float(index_daily["close"].rolling(20, min_periods=20).mean().loc[date]) if index_daily is not None and date in index_daily.index else None
        idx_close = float(index_daily["close"].loc[date]) if index_daily is not None and date in index_daily.index else None
        prev_date = dates[i - 1] if i > 0 else None

        # ---- 持仓退出 ----
        for code in list(positions):
            pos = positions[code]
            if date <= pos["entry_date"]:
                continue
            o, h, l, c = (open_.at[date, code], high.at[date, code], low.at[date, code], close.at[date, code])
            if pd.isna(c) or pd.isna(o):
                continue
            pos["peak"] = max(pos["peak"], float(c))
            valid_days = int(close[code].loc[pos["entry_date"]:date].dropna().shape[0])
            held_days = max(valid_days - 1, 0)  # 含买入日：买入日收盘后为第 1 天
            prev_c = float(pos["prev_close"] or c)
            limit_up_price = round(prev_c * 1.1, 2)

            if pos.get("pending_ma_exit"):
                _sell(code, pos, date, pos["shares"], float(o), "趋势止损破MA20", held_days)
                del positions[code]
                continue
            if c >= limit_up_price - 1e-6:  # S6 涨停日不卖
                pos["prev_close"], pos["kept_limit"] = float(c), True
                continue
            # S1 止损：可选信号①止跌低点破位（贴合低吸形态），否则固定比例
            if use_sig1_stop and pos.get("sig1_low"):
                stop_price = min(pos["sig1_low"] * 0.99, pos["entry_price"] * (1 - stop_loss))
            else:
                stop_price = pos["entry_price"] * (1 - stop_loss)
            if l <= stop_price:
                reason = "止损破位" if use_sig1_stop else f"硬止损-{int(stop_loss * 100)}%"
                _sell(code, pos, date, pos["shares"], float(min(o, stop_price)), reason, held_days)
                del positions[code]
                continue
            if pos.get("kept_limit"):  # S6 涨停次日不封板
                if limit_break_exit:
                    _sell(code, pos, date, pos["shares"], float(c), "涨停次日断板", held_days)
                    del positions[code]
                    continue
                pos["kept_limit"] = False  # 断板后继续持有，用后续移动止盈/前高减半追踪二次上攻
            if not pos.get("half_taken") and h >= pos["h_high"] and pos["shares"] >= 200:
                # S3 触及前高减半：仅当买入价未突破前高时有效（买入价已在前高上方则前高非压力位），
                # 且只在保本以上减半，避免「买入价>前高」时以低位前高卖出造成亏损。
                sell_price = max(float(pos["h_high"]), float(c))
                if sell_price >= pos["entry_price"]:
                    _sell(code, pos, date, (pos["shares"] // 200) * 100, sell_price, "前高减半", held_days)
                    pos["half_taken"] = True
                    if pos["shares"] < 100:
                        del positions[code]
                        continue
            # S4 移动止盈：仅在已有浮盈（peak 高于买入价）后才生效，亏损阶段交由硬止损处理，
            # 避免「买入即回撤」时把移动止盈当止损用。浮盈门槛（trailing_activate）经扫描为负贡献，已去除。
            if pos["peak"] > pos["entry_price"] and c <= pos["peak"] * (1 - trailing_stop):
                _sell(code, pos, date, pos["shares"], float(c), f"移动止盈回撤{int(trailing_stop * 100)}%", held_days)
                del positions[code]
                continue
            ma20_stock = float(close[code].rolling(20, min_periods=20).mean().loc[date])
            if held_days >= 10:  # S5 时间止损
                _sell(code, pos, date, pos["shares"], float(c), "时间止损10日", held_days)
                del positions[code]
                continue
            if not np.isnan(ma20_stock) and c < ma20_stock:  # S2 次日开盘卖出
                pos["pending_ma_exit"] = True
            pos["prev_close"], pos["kept_limit"] = float(c), False

        # ---- 开仓闸门（§11.1/§11.3）----
        equity_now = _equity_on(date)
        week_peak[week_key] = max(week_peak.get(week_key, equity_now), equity_now)
        if (
            len(positions) >= max_positions
            or i == 0
            or (cooldown_until is not None and date <= cooldown_until)
            or week_key in week_blocked
            or (week_peak.get(week_key, equity_now) > 0 and equity_now / week_peak[week_key] - 1 <= -0.05)
        ):
            if week_peak.get(week_key, equity_now) > 0 and equity_now / week_peak[week_key] - 1 <= -0.05:
                week_blocked.add(week_key)  # §11.3 周回撤≥5% 当周禁开仓
            equity_curve.append((d_str, equity_now))
            continue
        gate = True
        if prev_date is not None:
            if int(limit_up_n.get(prev_date, 0)) < min_limit_up:
                gate = False  # 前日涨停家数不足
            dn, up = int(limit_down_n.get(prev_date, 0)), int(limit_up_n.get(prev_date, 0))
            if dn > 10 or dn > up:
                gate = False  # 前日跌停家数超标
            if index_daily is not None and prev_date in index_daily.index and float(index_daily["pct_chg"].loc[prev_date]) <= -2.0:
                gate = False  # 大跌熔断
            if ma20_idx is not None and idx_close is not None and idx_close < ma20_idx:
                gate = False  # 上证跌破 MA20
        else:
            gate = False
        if not gate:
            equity_curve.append((d_str, equity_now))
            continue

        # ---- 买入（信号②确认：close=当日尾盘收盘买入 / next_open=次日开盘买入）----
        sig_date = prev_date if entry_mode == "next_open" else date
        day_signals = sorted(signals.get(sig_date, []), key=lambda s: -s["score"]) if sig_date is not None else []
        new_today = 0
        budget_used = 0.0
        for item in day_signals:
            if len(positions) >= max_positions or new_today >= 2 or budget_used >= 0.4:
                break
            code = item["code"]
            if code in positions or item["score"] < threshold:
                continue
            if entry_mode == "next_open":
                o = open_.at[date, code]
                pc = float(close.at[sig_date, code])
                if pd.isna(o) or o <= 0:
                    continue
                limit_up_open = round(pc * (1 + board_price_limit(code, is_st=False)), 2)
                if o >= limit_up_open - 1e-6:
                    continue  # 次日一字涨停买不进
                fill = costs.slipped_price(float(o), is_buy=True)
                prev_close = pc
            else:
                c = close.at[date, code]
                if pd.isna(c) or c <= 0:
                    continue
                fill = costs.slipped_price(float(c), is_buy=True)
                prev_close = float(c)
            pos_value = min(equity_now * 0.2, cash * 0.98)
            shares = math.floor(pos_value / fill / 100) * 100
            if shares < 100:
                continue
            amount = fill * shares
            fee = costs.commission(amount)
            if amount + fee > cash:
                continue
            cash -= amount + fee
            budget_used += amount / equity_now
            new_today += 1
            positions[code] = {
                "entry_date": date, "entry_price": fill, "shares": shares, "cost": amount + fee,
                "score": item["score"], "h_high": item["h_high"], "peak": fill,
                "prev_close": prev_close, "kept_limit": False, "half_taken": False, "pending_ma_exit": False,
                "sig1_low": item.get("sig1_low"),
            }
        equity_curve.append((d_str, _equity_on(date)))

    # ---- 指标与落库（仅因子实验室表）----
    equity = pd.Series({d: v for d, v in equity_curve}, dtype=float)
    final_equity = float(equity.iloc[-1]) if not equity.empty else initial_cash
    n_days = max(len(equity), 1)
    annual_return = (final_equity / initial_cash) ** (252 / n_days) - 1 if final_equity > 0 else -1.0
    returns = equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    running_max = equity.cummax()
    max_drawdown = float(((equity - running_max) / running_max).min()) if not equity.empty else 0.0
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    metrics = {
        "start_date": start_date, "end_date": end_date, "threshold": threshold, "top_n": top_n,
        "max_positions": max_positions, "initial_cash": initial_cash, "final_equity": round(final_equity, 2),
        "annual_return": round(annual_return, 4), "max_drawdown": round(max_drawdown, 4),
        "sharpe": round(sharpe, 2), "total_trades": len(trades), "win_rate": round(win_rate, 4),
        "avg_holding_days": round(sum(t["holding_days"] for t in trades) / len(trades), 1) if trades else 0.0,
        "events": len(events), "use_sig1_stop": use_sig1_stop, "stop_loss": stop_loss, "trailing_stop": trailing_stop,
        "sig2_vol_ratio": sig2_vol_ratio, "sig2_gain_pct": sig2_gain_pct,
        "min_limit_up": min_limit_up, "entry_mode": entry_mode, "limit_break_exit": limit_break_exit,
        "min_surge": min_surge, "sig1_shrink": sig1_shrink,
    }
    if persist:
        run_id = f"lp_{uuid4().hex[:12]}"
        database.execute(
            "INSERT INTO signal_backtest_runs(id,strategy_name,start_date,end_date,metrics_json,created_at) VALUES(?,?,?,?,?,?)",
            (run_id, LIMIT_STRATEGY_NAME, start_date, end_date, json.dumps(metrics, ensure_ascii=False), utc_now_text()),
        )
        database.executemany(
            """INSERT INTO signal_trades(run_id,code,name,entry_date,entry_price,shares,exit_date,exit_price,
               pnl,pnl_pct,holding_days,status,entry_reason,exit_reason)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(run_id, t["code"], t["name"], t["entry_date"], t["entry_price"], t["shares"],
              t["exit_date"], t["exit_price"], t["pnl"], t["pnl_pct"], t["holding_days"], "CLOSED",
              f"低吸/评分{t['score']:.0f}", t["reason"]) for t in trades],
        )
    metrics["trades"] = trades
    metrics["equity"] = equity_curve
    return metrics
