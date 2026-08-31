"""滚动样本外验证（walk-forward）：检验涨停回马枪参数是否过拟合。

做法：滚动划分「训练段 → 测试段」，训练段内扫描 ``threshold`` 找年化最优，
再用该参数在紧接着的测试段上回测，汇总各测试段（样本外）表现，与
「固定 threshold=70 全期」（样本内）对比。

由于涨停回马枪单次回测峰值内存约 1.4GB，连续多次回测会 OOM，本脚本通过
**子进程隔离**每次回测（每次回测在独立进程跑完即释放内存）。

用法：
    python scripts/walk_forward.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIST_DB = ROOT / "data" / "ashare_quant_hist.db"
PYTHON = sys.executable
THRESHOLD_GRID = [60, 70, 80]

# 固定参数（除 threshold 外均为当前最优，来自 limit_pullback_strategy.py 默认值）
FIXED = dict(
    top_n=3, max_positions=3, initial_cash=1_000_000.0, daily_budget=0.4,
    use_sig1_stop=False, stop_loss=0.06, trailing_stop=0.05,
    sig2_vol_ratio=2.0, sig2_gain_pct=0.02, min_limit_up=50,
    entry_mode="close", limit_break_exit=True, min_surge=1.05, sig1_shrink=0.4,
)

WORKER = r'''
import json, sys, os
sys.path.insert(0, {root!r})
for line in open({env!r}, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        if k in ("TUSHARE_TOKEN", "TUSHARE_API_URL") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())
from ashare_quant.database import Database
from ashare_quant.limit_pullback_strategy import run_limit_pullback_backtest
db = Database({hist_db!r})
m = run_limit_pullback_backtest(db, None, start_date={start!r}, end_date={end!r},
                                threshold={threshold}, persist=False, **{fixed!r})
keys = ["annual_return", "max_drawdown", "sharpe", "win_rate", "total_trades"]
print("RESULT_JSON:" + json.dumps({{k: m.get(k) for k in keys}}))
'''


def _run(start: str, end: str, threshold: float) -> dict:
    """在子进程里跑一次涨停回马枪回测，返回指标字典（内存隔离）。"""
    code = WORKER.format(
        root=str(ROOT), env=str(ROOT / ".env"), hist_db=str(HIST_DB),
        start=start, end=end, threshold=threshold, fixed=FIXED,
    )
    proc = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"回测子进程无结果：\nstdout={proc.stdout}\nstderr={proc.stderr}")


def _fmt(m: dict) -> str:
    return f"年化 {m['annual_return']*100:+.2f}% 回撤 {m['max_drawdown']*100:.2f}% 夏普 {m['sharpe']:.2f} 胜率 {m['win_rate']*100:.1f}%"


def main() -> None:
    # 1) 样本内：固定 threshold=70 全期
    full = _run("2024-01-01", "2026-08-28", 70.0)
    print("=" * 64)
    print("样本内（threshold=70 全期 2024-01~2026-08）：")
    print("  " + _fmt(full))

    # 2) 分年度稳健性（固定 threshold=70）
    print("\n分年度（threshold=70）：")
    for ys, ye in [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-08-28")]:
        print(f"  {ys[:4]} 年：{_fmt(_run(ys, ye, 70.0))}")

    # 3) walk-forward：训练段扫 threshold，测试段用最优 threshold
    print(f"\nwalk-forward（训练段扫 threshold {THRESHOLD_GRID}，测试段用训练最优）：")
    windows = [
        ("2024-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
        ("2024-07-01", "2025-06-30", "2025-07-01", "2025-12-31"),
        ("2025-01-01", "2025-12-31", "2026-01-01", "2026-06-30"),
        ("2025-07-01", "2026-06-30", "2026-07-01", "2026-08-28"),
    ]
    oos_annuals, oos_drawdowns = [], []
    for train_s, train_e, test_s, test_e in windows:
        best_th, best_ret = None, -1e9
        for th in THRESHOLD_GRID:
            m = _run(train_s, train_e, th)
            if m["annual_return"] > best_ret:
                best_ret, best_th = m["annual_return"], th
        t = _run(test_s, test_e, best_th)
        oos_annuals.append(t["annual_return"])
        oos_drawdowns.append(t["max_drawdown"])
        print(f"  训练[{train_s}~{train_e}] 选 threshold={best_th}（训练年化 {best_ret*100:+.1f}%）"
              f" → 测试[{test_s}~{test_e}] {_fmt(t)}")

    print("\n" + "=" * 64)
    print(f"样本外各测试段年化：{[f'{x*100:+.1f}%' for x in oos_annuals]}")
    print(f"样本外平均年化：{sum(oos_annuals)/len(oos_annuals)*100:+.2f}%")
    print(f"样本外平均回撤：{sum(oos_drawdowns)/len(oos_drawdowns)*100:.2f}%")
    diff = sum(oos_annuals) / len(oos_annuals) - full["annual_return"]
    verdict = "基本一致，参数稳健" if abs(diff) < 0.04 else "差异较大，存在过拟合风险"
    print(f"对比样本内年化 {full['annual_return']*100:+.2f}%（差值 {diff*100:+.2f} 个点）：{verdict}")


if __name__ == "__main__":
    main()
