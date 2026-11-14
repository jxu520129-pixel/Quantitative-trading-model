"""涨停回马枪 信号漏斗诊断：统计形态链路每一层的存活数量，解释「信号为什么少」。

复用 limit_pullback_strategy 的检测函数，只统计、不回测。
用法：python scripts/diagnose_pullback.py [--start 2024-01-01] [--end 2026-08-28]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        if k in ("TUSHARE_TOKEN", "TUSHARE_API_URL") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from ashare_quant.database import Database  # noqa: E402
from ashare_quant.limit_pullback_strategy import (  # noqa: E402
    _detect_pullback_events, _scan_entry_signals, _load_panels, _MAIN_BOARD_PREFIXES,
)
from ashare_quant.market_rules import board_price_limit  # noqa: E402

START, END = "2024-01-01", "2026-08-28"


def main() -> None:
    db = Database(ROOT / "data" / "ashare_quant_hist.db")
    industry_map = {r["code"]: r["industry"] for r in db.query_all("SELECT code,industry FROM stock_industry")}
    universe = [
        it for it in db.query_all(
            "SELECT code,name FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        ) if it["code"].startswith(_MAIN_BOARD_PREFIXES)
    ]
    print(f"① 沪深主板股票池（剔 ST/退市）：{len(universe)} 只")

    lookback = (datetime.strptime(START, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    panels = _load_panels(db, universe, lookback, END)
    close = panels["close"]

    # 全市场每日涨停家数（大盘闸门口径）
    sent_u = [
        it for it in db.query_all(
            "SELECT code FROM stock_basic WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0"
        ) if not it["code"].startswith(("43", "83", "87", "88", "920"))
    ]
    sc = _load_panels(db, sent_u, lookback, END).get("close")
    sc_pre = sc.shift(1)
    sc_lp = pd.Series({c: board_price_limit(c, is_st=False) for c in sc.columns})
    limit_up_n = (sc >= (sc_pre * (1 + sc_lp)).round(2) - 1e-6).sum(axis=1)

    # ② T 涨停启动 + H 冲高确认 事件数
    events = _detect_pullback_events(panels, min_surge=1.05)
    print(f"② 涨停启动(T)+次日冲高确认(H) 形态事件：{len(events)} 个（{START}~{END}）")

    # 行业排名（信号扫描需要）
    pct = close / close.shift(1) - 1
    sectors = sorted({v for v in industry_map.values() if v})
    industry_rank = pd.DataFrame(index=close.index)
    if sectors:
        ind_pos = {n: i for i, n in enumerate(sectors)}
        onehot = np.zeros((len(close.columns), len(sectors)), dtype=np.float32)
        for j, code in enumerate(close.columns):
            p = ind_pos.get(industry_map.get(code, ""))
            if p is not None:
                onehot[j, p] = 1.0
        den = pct.notna().astype(np.float32).to_numpy() @ onehot
        num = pct.fillna(0.0).to_numpy(dtype=np.float32) @ onehot
        ind_mean = pd.DataFrame(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), index=close.index, columns=sectors)
        industry_rank = ind_mean.rank(axis=1, ascending=False)

    # ③ 缩量回调 + 双信号止跌 后的最终信号
    signals = _scan_entry_signals(panels, events, industry_map, industry_rank)
    total_sig = sum(len(v) for v in signals.values())
    sig_days = len(signals)
    print(f"③ 通过缩量回调+双信号止跌：{total_sig} 个买入信号，分布在 {sig_days} 个交易日")

    # ④ 大盘闸门（前日涨停≥50家）
    gated = 0
    gated_days = 0
    for d, sigs in signals.items():
        pos = sc.index.get_loc(d) if d in sc.index else None
        prev_up = int(limit_up_n.iloc[pos - 1]) if pos is not None and pos > 0 else 0
        if prev_up >= 50:
            gated += len(sigs)
            gated_days += 1
    print(f"④ 大盘闸门(前日涨停≥50家)后：{gated} 个信号，分布在 {gated_days} 个交易日")

    # 交易日统计
    n_trade_days = len([d for d in close.index if START <= d.strftime("%Y-%m-%d") <= END])
    print(f"\n区间交易日约 {n_trade_days} 天，④最终信号 {gated} 个 ≈ 平均每 {n_trade_days/max(gated,1):.1f} 个交易日出 1 个")


if __name__ == "__main__":
    main()
