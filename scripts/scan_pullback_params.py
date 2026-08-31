"""策略2（涨停回马枪）止损/止盈参数扫描。

在「通用 bug 已修复」的基础上，扫描 stop_loss × trailing_stop 组合，
找年化收益最高、回撤可控的参数。use_sig1_stop 已证为负贡献，固定为 False。
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

from ashare_quant.database import Database  # noqa: E402
from ashare_quant.limit_pullback_strategy import run_limit_pullback_backtest  # noqa: E402

db = Database(ROOT / "data" / "ashare_quant_hist.db")

STOP_LOSSES = [0.05, 0.06, 0.08]
TRAILING_STOPS = [0.05, 0.08, 0.12]

print(f"{'stop_loss':>10} {'trailing':>9} | {'年化':>8} {'回撤':>8} {'夏普':>6} {'胜率':>7} {'交易':>5} {'期末净值':>10}")
print("-" * 75)
results = []
for sl in STOP_LOSSES:
    for ts in TRAILING_STOPS:
        m = run_limit_pullback_backtest(
            db, None, start_date="2024-01-01", end_date="2026-08-28",
            threshold=50.0, max_positions=3, initial_cash=1_000_000.0,
            daily_budget=0.4, use_sig1_stop=False, stop_loss=sl, trailing_stop=ts, persist=False,
        )
        results.append((sl, ts, m))
        print(f"{sl*100:9.1f}% {ts*100:8.1f}% | {m['annual_return']*100:7.2f}% {m['max_drawdown']*100:7.2f}% "
              f"{m['sharpe']:6.2f} {m['win_rate']*100:6.2f}% {m['total_trades']:5d} {m['final_equity']:>10,.0f}", flush=True)

best = max(results, key=lambda r: r[2]["annual_return"])
print("-" * 75)
print(f"最优：stop_loss={best[0]*100:.1f}% trailing_stop={best[1]*100:.1f}% "
      f"→ 年化 {best[2]['annual_return']*100:.2f}% 回撤 {best[2]['max_drawdown']*100:.2f}%")
