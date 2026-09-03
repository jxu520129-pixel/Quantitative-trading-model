"""短线人气题材策略（热度共振打分模型）的日线近似回测。

对应《短线人气题材选股策略.md》。该策略为盘中实时型，人气榜/涨停池/封板时间等
数据无法历史回填，本模块按文档框架用日线口径近似，差异逐项标注：

- A 人气分（0~30）：东财人气榜仅有实时前 100、无历史，用当日涨幅横截面分位作为
  关注度代理，分档与文档一致（30/26/20/14/8），排名跃升 +2 分保留；
- B 涨停连板分（-3~25）：按日线收盘判定涨停与连板高度，全档位照搬；首次封板
  时间加分与炸板扣分需要盘中数据，不参与；
- C 题材热度分（0~30）：人气聚集规则无法回填，用「板块涨停聚集 ≥3 家」+「板块
  涨幅领先前 10」两条规则判定，两条全中视作主线（30），任一条为一般热点（12）；
- D 资金量能分（0~15）：量比 1.5~3 (+3) / >3 (+4) 保留；主力净流入、封单市值比、
  龙虎榜席位需盘中或逐日 LHB 数据，回测不参与；
- 剔除规则：ST / 北交所 / 创业板 / 科创板 / 次新（<60 个交易日）/ 一字板买不进
  照常执行；利空公告与龙虎榜三日榜为公告级数据，实盘扫描启用、回测不启用；
- 交易执行：T 日评分 → T+1 开盘买入（开盘一字板跳过），退出按 §6.3 硬风控近似
  （-6% 止损、第 3 交易日时间止损、浮盈超 8% 后回撤过半止盈、当日涨停继续持有、
  接力单断板当日收盘离场），成本用 market_rules 的佣金/印花税/滑点口径。
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

from .market_rules import TradingCosts, board_price_limit
from .models import utc_now_text

LOG = logging.getLogger(__name__)

HOT_STRATEGY_NAME = "短线人气·热度共振"

# 剔除的代码前缀：创业板 300/301、科创板 688/689、北交所 43/83/87/88/920（文档 §3 规则 2/8）
_EXCLUDED_PREFIXES = ("300", "301", "688", "689", "43", "83", "87", "88", "920")
_MIN_HISTORY_BARS = 60  # 剔除上市不满 60 个交易日的次新股（文档 §3 规则 3）


def ensure_industry_mapping(database: Any, data_service: Any, min_coverage: float = 0.8) -> int:
    """确保 stock_industry 覆盖足够多 A 股；不足时用 Tushare 申万行业成分（SW2021）回填。"""
    total = database.query_one(
        "SELECT COUNT(*) n FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
    )["n"]
    mapped = database.query_one(
        """SELECT COUNT(*) n FROM stock_industry s JOIN stock_basic b ON s.code=b.code
           WHERE b.security_type='STOCK'"""
    )["n"]
    if total and mapped / total >= min_coverage:
        return mapped
    from .data.providers import tushare_pro_from_env

    pro = tushare_pro_from_env()
    if pro is None:
        return mapped
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    while True:
        frame = pro.index_member_all(src="SW2021", limit=2000, offset=offset)
        if frame is None or frame.empty:
            break
        keys = set((str(r.ts_code), str(r.l2_code)) for r in frame.itertuples())
        if not (keys - seen):
            break  # 代理不支持翻页时防死循环
        seen |= keys
        rows.extend(frame.to_dict("records"))
        if len(frame) < 2000:
            break
        offset += len(frame)
    now = utc_now_text()
    upserts = []
    for item in rows:
        if str(item.get("is_new") or "") != "Y":
            continue  # 已调出成分的历史记录不参与当前板块聚集
        code = str(item.get("ts_code") or "")[:6]
        industry = str(item.get("l2_name") or item.get("l1_name") or "").strip()
        if code and industry:
            upserts.append((code, industry, now))
    if upserts:
        database.executemany(
            "INSERT INTO stock_industry(code,industry,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(code) DO UPDATE SET industry=excluded.industry,updated_at=excluded.updated_at",
            upserts,
        )
        mapped = database.query_one(
            """SELECT COUNT(*) n FROM stock_industry s JOIN stock_basic b ON s.code=b.code
               WHERE b.security_type='STOCK'"""
        )["n"]
        LOG.info("申万行业映射已回填：覆盖 %s/%s 只 A 股", mapped, total)
    return mapped


def _load_panels(database: Any, universe: list[dict[str, Any]], lookback_start: str, end_date: str) -> dict[str, pd.DataFrame]:
    """按代码批量取日线并转成 date×code 宽表。"""
    # trade_date 有两种格式：hist 库为 YYYYMMDD（无连字符），默认演示/模拟盘库为 YYYY-MM-DD（带连字符）。
    # 字符串比较混用两种格式会导致 end_date 所在年份的数据被整体误滤（"20260102" > "2026-08-28"）。
    # 这里先探测库里实际格式，再把查询参数对齐到同一格式。
    sample = database.query_one("SELECT trade_date FROM daily_bars LIMIT 1")
    has_dash = bool(sample and sample.get("trade_date") and "-" in str(sample["trade_date"]))
    if not has_dash:
        lookback_start = str(lookback_start).replace("-", "")
        end_date = str(end_date).replace("-", "")
    frames: list[pd.DataFrame] = []
    chunk_size = 500
    for i in range(0, len(universe), chunk_size):
        chunk = [item["code"] for item in universe[i : i + chunk_size]]
        placeholders = ",".join("?" * len(chunk))
        rows = database.query_all(
            f"""SELECT code,trade_date,open,high,low,close,volume FROM daily_bars
                WHERE code IN ({placeholders}) AND trade_date>=? AND trade_date<=? ORDER BY trade_date""",
            [*chunk, lookback_start, end_date],
        )
        frames.append(pd.DataFrame(rows))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        return {}
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    panels = {}
    for column in ["open", "high", "low", "close", "volume"]:
        panels[column] = (
            data.pivot_table(index="trade_date", columns="code", values=column, aggfunc="last")
            .sort_index()
            .astype(np.float32)  # 长区间（2020~2026）控制内存
        )
    return panels


def _limit_streak(is_limit: pd.DataFrame) -> pd.DataFrame:
    """每列计算截至当日的连续涨停天数（含当日）。"""
    cum = is_limit.cumsum()
    reset = cum.where(~is_limit)
    return is_limit * (cum - reset.ffill().fillna(0))


def score_cross_section(panels: dict[str, pd.DataFrame], industry_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """按文档 §4 计算全历史逐日四维得分，返回各分项与涨停状态的宽表。"""
    close, open_, high, low, volume = panels["close"], panels["open"], panels["high"], panels["low"], panels["volume"]
    pre_close = close.shift(1)
    pct = close / pre_close - 1
    limit_pct = pd.Series({code: board_price_limit(code, is_st=False) for code in close.columns})
    limit_price = (pre_close * (1 + limit_pct)).round(2)
    is_limit = close >= (limit_price - 1e-6)
    streak = _limit_streak(is_limit)
    was_limit = is_limit.shift(1, fill_value=False)
    vol_mean = volume.rolling(20, min_periods=20).mean().shift(1)
    vol_ratio = volume / vol_mean

    # B 涨停/连板分：按文档优先级从高到低
    b = np.select(
        [
            streak >= 6,
            (streak >= 4) & (streak <= 5),
            streak == 3,
            streak == 2,
            streak == 1,
            was_limit & ~is_limit,
            (~is_limit) & (pct > 0.07),
        ],
        [16.0, 20.0, 18.0, 14.0, 10.0, 6.0, 4.0],
        default=0.0,
    )
    b = pd.DataFrame(b, index=close.index, columns=close.columns)

    # A 人气分代理：当日涨幅横截面分位分档 + 排名跃升 +2（封顶 30）
    pct_rank = pct.rank(axis=1, pct=True)
    a = np.select(
        [pct_rank >= 0.995, pct_rank >= 0.98, pct_rank >= 0.94, pct_rank >= 0.85, pct_rank >= 0.70],
        [30.0, 26.0, 20.0, 14.0, 8.0],
        default=0.0,
    )
    a = pd.DataFrame(a, index=close.index, columns=close.columns)
    rank_jump = (pct_rank - pct_rank.shift(1)) >= 0.004  # 约 ≥20 名/4500 只
    a = np.minimum(a + rank_jump * 2.0, 30.0)

    # C 题材热度分：板块涨停聚集（≥3 家）+ 板块涨幅领先（前 10），两条全中=主线 30，任一=12。
    # 用行业 one-hot 矩阵乘法聚合（等价于 stack+groupby，但长区间下内存与耗时低一个量级）。
    industry = pd.Series({code: industry_map.get(code, "") for code in close.columns})
    sectors = sorted({v for v in industry_map.values() if v})
    c = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if sectors:
        ind_pos = {name: i for i, name in enumerate(sectors)}
        onehot = np.zeros((len(close.columns), len(sectors)), dtype=np.float32)
        for j, code in enumerate(close.columns):
            pos_ = ind_pos.get(industry.get(code, ""))
            if pos_ is not None:
                onehot[j, pos_] = 1.0
        pct_den = pct.notna().astype(np.float32).to_numpy() @ onehot
        pct_num = pct.fillna(0.0).to_numpy(dtype=np.float32) @ onehot
        limit_num = is_limit.astype(np.float32).to_numpy() @ onehot
        ind_mean = pd.DataFrame(
            np.where(pct_den > 0, pct_num / np.maximum(pct_den, 1e-9), np.nan),
            index=close.index, columns=sectors,
        )
        ind_rank = ind_mean.rank(axis=1, ascending=False)
        cluster = pd.DataFrame(limit_num >= 3, index=close.index, columns=sectors)
        leading = ind_rank <= 10
        for code in close.columns:
            sector = industry.get(code, "")
            if not sector or sector not in cluster.columns:
                continue
            sector_cluster = cluster[sector].reindex(close.index, fill_value=False).astype(bool)
            sector_leading = leading[sector].reindex(close.index, fill_value=False).astype(bool)
            both = sector_cluster & sector_leading
            either = sector_cluster | sector_leading
            c[code] = np.where(both, 30.0, np.where(either, 12.0, 0.0))

    # D 资金量能分：量比 1.5~3 (+3) / >3 (+4)
    d = np.select([vol_ratio > 3, vol_ratio >= 1.5], [4.0, 3.0], default=0.0)
    d = pd.DataFrame(d, index=close.index, columns=close.columns)

    return {
        "score": a + b + c + d,
        "pct": pct,
        "is_limit": is_limit,
        "streak": streak,
        "vol_ratio": vol_ratio,
        "bars_count": close.notna().cumsum(),
        "part_a": a, "part_b": b, "part_c": c, "part_d": d,
    }


def _pick_entries(
    scores: dict[str, pd.DataFrame],
    date,
    threshold: float,
    top_n: int,
    min_sector_score: float = 12.0,
    min_vol_ratio: float = 1.5,
) -> list[dict[str, Any]]:
    """选出强候选清单，叠加文档的共振门槛：题材热度分 ≥ 次主线（C≥12）且量比 ≥1.5（资金确认）。"""
    row = scores["score"].loc[date].dropna()
    row = row[scores["bars_count"].loc[date] >= _MIN_HISTORY_BARS]
    row = row[row >= threshold].sort_values(ascending=False).head(top_n)
    entries = []
    for code, total in row.items():
        sector_score = float(scores["part_c"].loc[date, code])
        vol_ratio = float(scores["vol_ratio"].loc[date, code]) if pd.notna(scores["vol_ratio"].loc[date, code]) else 0.0
        if sector_score < min_sector_score or vol_ratio < min_vol_ratio:
            continue  # 文档核心思想：人气×题材×涨停×资金四维共振，缺题材或缺量能不出手
        streak = float(scores["streak"].loc[date, code])
        pct = float(scores["pct"].loc[date, code])
        mode = "连板接力" if streak >= 2 else ("首板打板" if streak == 1 else ("低吸埋伏" if pct < 0 else "半路买入"))
        entries.append({
            "code": code, "score": float(total), "mode": mode, "streak": int(streak),
            "vol_ratio": vol_ratio,
        })
    return entries


def _mode_gap_ok(mode: str, gap: float) -> bool:
    """按文档 §5 的入场口径过滤次日开盘缺口（竞价条件的日线近似）。"""
    if mode == "连板接力":
        return 0.02 <= gap <= 0.06  # 高开2%~6%可接力；高开>7%或平开低开不接
    if mode == "首板打板":
        return gap <= 0.05  # 打板本应盘中封板瞬间参与，日线近似下拒绝极端高开追价
    if mode == "低吸埋伏":
        return gap < 0  # 回调低开才低吸
    return 0.0 <= gap <= 0.06  # 半路买入


def run_hot_backtest(
    database: Any,
    data_service: Any,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    threshold: float = 75.0,
    top_n: int = 10,
    max_positions: int = 1,
    initial_cash: float = 10000.0,
    daily_budget: float = 0.9,
    min_sector_score: float = 12.0,
    min_vol_ratio: float = 1.5,
    weekly_entry_cap: int = 3,
    sentiment_scope: str = "hs",
    persist: bool = True,
    exit_mode: str = "legacy",  # legacy=原退出逻辑；optimized=分批止盈+移动止盈
    partial_take: float = 0.10,  # optimized：浮盈达到该比例时减半锁定利润
    trailing_stop: float = 0.06,  # optimized：剩余仓位从峰值回撤该比例止盈
    stop_loss: float = 0.06,  # 硬止损比例（两种模式通用）
) -> dict[str, Any]:
    """热度共振打分模型的日线近似回测。返回指标 dict 并默认落库供看板展示。

    叠加文档四项纪律：§5 各参与方式的竞价缺口过滤（日线用开盘缺口近似）、
    §1 四维共振门槛（题材分≥次主线 且 量比≥1.5）与适用行情门槛（涨停家数≥50 且
    连板高度≥3）、§7.1 每周开仓 ≤3 笔、§8 情绪周期风控——昨日涨停溢价/涨停家数/
    最高连板高度三项中两项落入冰点档（溢价<-1% / 家数<30 / 高度≤2）则当日空仓。
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    ensure_industry_mapping(database, data_service)
    industry_rows = database.query_all("SELECT code,industry FROM stock_industry")
    industry_map = {r["code"]: r["industry"] for r in industry_rows}

    universe = [
        item for item in database.query_all(
            "SELECT code,name FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        )
        if not item["code"].startswith(_EXCLUDED_PREFIXES)
    ]
    lookback_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    panels = _load_panels(database, universe, lookback_start, end_date)
    if not panels:
        raise ValueError("日线数据为空，请先执行 update-data")
    scores = score_cross_section(panels, industry_map)

    close, open_, high, low = panels["close"], panels["open"], panels["high"], panels["low"]
    dates = [d for d in close.index if start_date <= d.strftime("%Y-%m-%d") <= end_date]
    # §8 情绪周期风控 + §1 适用行情门槛。sentiment_scope："hs"=沪深两市（文档字面口径，
    # 默认）；"main"=仅沪深主板（对"活跃"要求更严，样本内表现更好但非文档字面口径）。
    # 情绪统计均剔除 ST；交易池始终仅沪深主板。
    if sentiment_scope == "main":
        sentiment_universe = [
            item for item in database.query_all(
                "SELECT code FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
            )
            if not item["code"].startswith(("300", "301", "688", "689", "43", "83", "87", "88", "920"))
        ]
    else:
        sentiment_universe = [
            item for item in database.query_all(
                "SELECT code FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
            )
            if not item["code"].startswith(("43", "83", "87", "88", "920"))
        ]
    sentiment_panels = _load_panels(database, sentiment_universe, lookback_start, end_date)
    all_close = sentiment_panels.get("close")
    if all_close is None or all_close.empty:
        all_close = close
    all_pre = all_close.shift(1)
    all_limit_pct = pd.Series({code: board_price_limit(code, is_st=False) for code in all_close.columns})
    all_limit_price = (all_pre * (1 + all_limit_pct)).round(2)
    all_is_limit = all_close >= (all_limit_price - 1e-6)
    limit_count = all_is_limit.sum(axis=1)
    max_streak = _limit_streak(all_is_limit).max(axis=1)
    premium = (all_close / all_pre - 1).where(all_is_limit.shift(1, fill_value=False)).mean(axis=1)  # 昨日涨停股今日平均涨幅
    frozen_votes = (
        (limit_count < 30).astype(np.int8)
        + (max_streak <= 2).astype(np.int8)
        + (premium < -0.01).astype(np.int8)
    )
    market_frozen = (frozen_votes >= 2).shift(1, fill_value=False)  # 用昨日情绪决定今日能否开仓
    market_active = ((limit_count >= 50) & (max_streak >= 3)).shift(1, fill_value=False)
    costs = TradingCosts()
    cash = initial_cash
    positions: dict[str, dict[str, Any]] = {}
    equity_curve: list[tuple[str, float]] = []
    trades: list[dict[str, Any]] = []
    entries_per_week: dict[tuple[int, int], int] = {}
    name_map = {item["code"]: item["name"] for item in universe}

    def _equity_on(date) -> float:
        total = cash
        for code, pos in positions.items():
            price = close.at[date, code]
            if pd.isna(price):
                series = close[code].loc[:date].dropna()
                price = float(series.iloc[-1]) if not series.empty else pos["entry_price"]
            total += pos["shares"] * float(price)
        return total

    for i, date in enumerate(dates):
        # ---- 先处理持仓退出（T+1 制度：买入当日不卖）----
        for code in list(positions):
            pos = positions[code]
            if date <= pos["entry_date"]:
                continue
            o, h, l, c = (open_.at[date, code], high.at[date, code], low.at[date, code], close.at[date, code])
            if pd.isna(c) or pd.isna(o):
                continue  # 停牌顺延
            pos["peak"] = max(pos["peak"], float(c))
            limit_pct = board_price_limit(code, is_st=False)
            limit_up_price = round(float(pos.get("prev_close") or 0) * (1 + limit_pct), 2) if pos.get("prev_close") else None
            held_days = sum(1 for d in dates if pos["entry_date"] < d <= date)

            def _sell(shares_to_sell: int, price: float, reason: str) -> None:
                nonlocal cash
                fill = costs.slipped_price(price, is_buy=False)
                amount = fill * shares_to_sell
                cash += amount - costs.commission(amount) - costs.stamp_duty(amount, is_sell=True)
                sold_cost = pos["cost"] * (shares_to_sell / pos["shares"])
                trades.append({
                    "code": code, "name": name_map.get(code, code), "mode": pos["mode"], "score": pos["score"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"), "entry_price": round(pos["entry_price"], 3),
                    "shares": shares_to_sell, "exit_date": date.strftime("%Y-%m-%d"), "exit_price": round(fill, 3),
                    "pnl": round(fill * shares_to_sell - sold_cost, 2),
                    "pnl_pct": round(fill / pos["entry_price"] - 1, 4), "holding_days": held_days, "reason": reason,
                })
                pos["shares"] -= shares_to_sell
                pos["cost"] -= sold_cost

            # §6.1 高开>7% 兑现一半 / 低开<-2% 立即走人：这两条的文档前提是"视承接/竞价卖压"
            # 的人工盘中判断，纯机械模拟经样本验证均为净负贡献（高开兑现砍掉连板大肉、
            # 无条件低开走错杀反包），故回测不落地，留给实盘人工执行环节。
            exit_price, reason = None, ""
            if limit_up_price and c >= limit_up_price - 1e-6:
                pos["prev_close"] = float(c)  # 当日涨停：继续持有（文档 §6.1）
                pos["kept_limit"] = True
            else:
                # §6.3 统一硬风控优先级最高：先查硬止损，再断板/止盈/时间止损
                stop_price = pos["entry_price"] * (1 - stop_loss)
                if l <= stop_price:
                    exit_price = float(min(o, stop_price))  # 跳空低开按开盘价止损
                    reason = f"硬止损-{int(stop_loss * 100)}%"
                if exit_price is None and pos.get("kept_limit") is False and pos["mode"] == "连板接力":
                    exit_price, reason = float(c), "断板卖出"
                if exit_mode == "optimized":
                    # 分批止盈：浮盈达到目标即减半锁定利润（降低回撤），剩余继续追踪
                    gain = c / pos["entry_price"] - 1
                    if exit_price is None and not pos.get("half_taken") and gain >= partial_take and pos["shares"] >= 200:
                        half = (pos["shares"] // 200) * 100
                        _sell(half, float(c), f"分批止盈+{int(partial_take * 100)}%")
                        pos["half_taken"] = True
                        if pos["shares"] < 100:
                            del positions[code]
                            continue
                    # 移动止盈：剩余仓位自持仓期峰值回撤 trailing_stop 即离场（仅在浮盈后生效，让利润奔跑）
                    if exit_price is None and pos["peak"] > pos["entry_price"] and c <= pos["peak"] * (1 - trailing_stop):
                        exit_price, reason = float(c), f"移动止盈-{int(trailing_stop * 100)}%"
                    if exit_price is None and held_days >= 3:
                        exit_price, reason = float(c), "时间止损3日"
                else:
                    peak_gain = pos["peak"] / pos["entry_price"] - 1
                    if exit_price is None and peak_gain >= 0.08 and (c / pos["entry_price"] - 1) <= peak_gain / 2:
                        exit_price, reason = float(c), "盈利回撤保护"
                    if exit_price is None and held_days >= 3:
                        exit_price, reason = float(c), "时间止损3日"
            if exit_price is None:
                pos["prev_close"] = float(c)
                pos["kept_limit"] = False
                continue
            _sell(pos["shares"], exit_price, reason)
            del positions[code]

        # ---- 再处理当日开仓（用 T-1 日评分清单，开盘买入）----
        if i == 0:
            equity_curve.append((date.strftime("%Y-%m-%d"), cash))
            continue
        prev = dates[i - 1]
        if len(positions) >= max_positions:
            equity_curve.append((date.strftime("%Y-%m-%d"), _equity_on(date)))
            continue
        week_key = date.isocalendar()[:2]
        if entries_per_week.get(week_key, 0) >= weekly_entry_cap:
            equity_curve.append((date.strftime("%Y-%m-%d"), _equity_on(date)))
            continue  # §7.1 开仓频率红线：每周开仓 ≤3 笔
        if bool(market_frozen.loc[date]) or not bool(market_active.loc[date]):
            equity_curve.append((date.strftime("%Y-%m-%d"), _equity_on(date)))
            continue  # §8 情绪冰点空仓；§1 仅在情绪活跃期（涨停≥50家且高度≥3）参与
        entries = _pick_entries(scores, prev, threshold, top_n, min_sector_score, min_vol_ratio)
        for item in entries:
            if len(positions) >= max_positions:
                break
            code = item["code"]
            if code in positions:
                continue
            o = open_.at[date, code]
            prev_close = float(close.at[prev, code])
            limit_up_open = round(prev_close * (1 + board_price_limit(code, is_st=False)), 2)
            if pd.isna(o) or o <= 0 or o >= limit_up_open - 1e-6:
                continue  # 停牌或一字板买不进（文档 §3 规则 4）
            gap = float(o) / prev_close - 1
            if not _mode_gap_ok(item["mode"], gap):
                continue  # §5 各参与方式的竞价缺口过滤（日线用开盘缺口近似）
            fill = costs.slipped_price(float(o), is_buy=True)
            equity_now = _equity_on(date)
            budget = min(cash, equity_now * daily_budget)
            shares = math.floor(budget / fill / 100) * 100
            if shares < 100:
                continue
            amount = fill * shares
            fee = costs.commission(amount)
            if amount + fee > cash:
                shares -= 100
                if shares < 100:
                    continue
                amount = fill * shares
                fee = costs.commission(amount)
            cash -= amount + fee
            entries_per_week[week_key] = entries_per_week.get(week_key, 0) + 1
            positions[code] = {
                "entry_date": date, "entry_price": fill, "shares": shares, "cost": amount + fee,
                "mode": item["mode"], "score": item["score"], "peak": fill, "prev_close": prev_close,
                "kept_limit": False, "half_taken": False,
            }
        equity_curve.append((date.strftime("%Y-%m-%d"), _equity_on(date)))

    # ---- 指标 ----
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
        "min_sector_score": min_sector_score, "min_vol_ratio": min_vol_ratio, "weekly_entry_cap": weekly_entry_cap,
        "sentiment_scope": sentiment_scope, "exit_mode": exit_mode, "partial_take": partial_take,
        "trailing_stop": trailing_stop, "stop_loss": stop_loss,
    }

    if persist:
        # 只落因子实验室的表（signal_backtest_runs/signal_trades）；不写 backtest_runs——
        # 那是模拟交易 Backtrader 回测的展示来源，保持各自页面职责干净。
        run_id = f"hot_{uuid4().hex[:12]}"
        database.execute(
            "INSERT INTO signal_backtest_runs(id,strategy_name,start_date,end_date,metrics_json,created_at) VALUES(?,?,?,?,?,?)",
            (run_id, "短线人气·热度共振", start_date, end_date,
             json.dumps(metrics, ensure_ascii=False), utc_now_text()),
        )
        database.executemany(
            """INSERT INTO signal_trades(run_id,code,name,entry_date,entry_price,shares,exit_date,exit_price,
               pnl,pnl_pct,holding_days,status,entry_reason,exit_reason)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(run_id, t["code"], t["name"], t["entry_date"], t["entry_price"], t["shares"],
              t["exit_date"], t["exit_price"], t["pnl"], t["pnl_pct"], t["holding_days"], "CLOSED",
              f"{t['mode']}/评分{t['score']:.0f}", t["reason"]) for t in trades],
        )
    metrics["trades"] = trades
    metrics["equity"] = equity_curve
    return metrics
