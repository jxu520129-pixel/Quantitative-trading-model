"""超跌反弹·量比确认：单参数变体长区间定位（找出过拟合来源）。

坐标下降「最优」(near_low 0.2 + vol_ratio 1.0 + take 20%) 短区间 24.88%/12.69% 很亮，
但长区间 10.61%/27.43% 反而不如基准 14.13%/26.21%。本轮逐个单改参数、放长区间定位，
判断是「保持不动」还是「单独采纳某个改进」。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        if k in ("TUSHARE_TOKEN", "TUSHARE_API_URL") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())

from ashare_quant.services.runtime import build_runtime  # noqa: E402
from ashare_quant.lab import run_signal_backtest, resolve_factor_expressions  # noqa: E402

LONG = ("2020-01-01", "2026-08-28")
SHORT = ("2024-01-01", "2026-08-28")

BASE = dict(near_low=0.15, up_days=0.6, ma20_dev=-0.02, vol_ratio=1.2, stop=0.06, take=0.15)

rt = build_runtime(prefer_hist_db=True)
rt.settings.raw.setdefault("scan", {})["min_average_amount"] = 1e5
exprs = resolve_factor_expressions(rt.database, ["near_low_20", "up_days_5", "ma20_dev", "vol_ratio"])


def run(c: dict, start: str, end: str) -> dict:
    entry = {"combine": "AND", "conditions": [
        {"factor": "near_low_20", "op": "<", "value": c["near_low"]},
        {"factor": "up_days_5", "op": ">", "value": c["up_days"]},
        {"factor": "ma20_dev", "op": ">", "value": c["ma20_dev"]},
        {"factor": "vol_ratio", "op": ">", "value": c["vol_ratio"]},
    ]}
    exit_ = {"combine": "OR", "conditions": [],
             "stop_loss_pct": c["stop"], "take_profit_pct": c["take"],
             "trailing_stop_pct": 0.0, "breakout_exit": False}
    _t, m, _e = run_signal_backtest(rt.data, rt.settings, exprs, entry, exit_, start, end)
    return m


def fmt(m: dict) -> str:
    pf = m["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
    return (f"年化 {m['annual_return']*100:6.2f}% | 回撤 {m['max_drawdown']*100:6.2f}% | "
            f"盈亏比 {pf_txt:>5} | 胜率 {m['win_rate']*100:5.2f}% | 交易 {m['total_trades']:4d}")


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("超跌反弹·量比确认 | 单参数变体长区间定位")
    print("=" * 100)
    combos = [
        ("基准", dict(BASE)),
        ("take→20%", {**BASE, "take": 0.20}),
        ("vol→1.0", {**BASE, "vol_ratio": 1.0}),
        ("near_low→0.2", {**BASE, "near_low": 0.2}),
        ("vol1.0+near0.2", {**BASE, "vol_ratio": 1.0, "near_low": 0.2}),
        ("三者全改(贪心)", {**BASE, "take": 0.20, "vol_ratio": 1.0, "near_low": 0.2}),
    ]
    for name, c in combos:
        m_long = run(c, *LONG)
        m_short = run(c, *SHORT)
        print(f"\n【{name:14s}】take={c['take']:.0%} vol={c['vol_ratio']} near_low={c['near_low']} stop={c['stop']:.0%}")
        print(f"  长区间 2020~2026  {fmt(m_long)}")
        print(f"  短区间 2024~2026  {fmt(m_short)}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
