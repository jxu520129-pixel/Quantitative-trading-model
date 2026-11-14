"""箱体突破·放量确认：候选组合长区间复验（回撤优先裁决）。

第一轮坐标下降暴露贪心「按年化选优」会过拟合（box→20%、trailing→8%，长区间回撤爆到 52.65%）。
本轮改为：固定 take=30%、box=15%（均为已验证甜点），只对比 vol_ratio × trailing_stop
的候选组合，用 2020-01-01 ~ 2026-08-28 长区间 + 2024-01-01 ~ 2026-08-28 短区间双口径裁决，
优先控制回撤（符合项目「收益持平、回撤改善」的一贯偏好）。
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

rt = build_runtime(prefer_hist_db=True)
rt.settings.raw.setdefault("scan", {})["min_average_amount"] = 1e5
exprs = resolve_factor_expressions(rt.database, ["breakout_4", "vol_ratio", "ma20_dev", "box_range_5"])

# 候选组合：vol_ratio × trailing_stop，take=30%、box=15% 固定
CANDIDATES = [
    ("基准",       2.0, 0.12),
    ("历史优化",   1.5, 0.12),
    ("trail15",   2.0, 0.15),
    ("trail18",   2.0, 0.18),
    ("1.5+trail15", 1.5, 0.15),
    ("1.5+trail18", 1.5, 0.18),
]


def run(vol_ratio: float, trailing: float, start: str, end: str) -> dict:
    entry = {"combine": "AND", "conditions": [
        {"factor": "breakout_4", "op": ">", "value": 0.005},
        {"factor": "vol_ratio", "op": ">", "value": vol_ratio},
        {"factor": "ma20_dev", "op": ">", "value": 0.0},
        {"factor": "box_range_5", "op": "<", "value": 0.15},
    ]}
    exit_ = {"combine": "OR", "conditions": [],
             "stop_loss_pct": 0.0, "take_profit_pct": 0.30,
             "trailing_stop_pct": trailing, "breakout_exit": False}
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
    print("箱体突破·放量确认 | 候选组合双区间复验（take=30%、box=15% 固定）")
    print("=" * 100)
    for name, v, tr in CANDIDATES:
        m_long = run(v, tr, *LONG)
        m_short = run(v, tr, *SHORT)
        print(f"\n【{name:12s}】vol_ratio={v} trailing={tr:.0%}")
        print(f"  长区间 2020~2026  {fmt(m_long)}")
        print(f"  短区间 2024~2026  {fmt(m_short)}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
