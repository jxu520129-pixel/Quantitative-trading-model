"""两个新策略模板的 5 年回测：布林带收口·放量突破 / 超卖反转·KDJ回升。

策略定义见 ashare_quant/lab.py 的 BUILTIN_STRATEGY_TEMPLATES（2026-11 新增示例模板）。
用法（项目根目录，需已拉取历史库 data/ashare_quant_hist.db）：
    python scripts/backtest_new_templates.py
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

START, END = "2021-09-01", "2026-08-28"  # 约 5 年

rt = build_runtime(prefer_hist_db=True)
# hist 库 amount 单位是「千元」，把 config 里「元」口径的成交额阈值换算成千元口径
rt.settings.raw.setdefault("scan", {})["min_average_amount"] = 1e5


def fmt(m: dict) -> str:
    pf = m["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:5.2f}"
    return (f"年化 {m['annual_return']*100:7.2f}% | 回撤 {m['max_drawdown']*100:7.2f}% | "
            f"盈亏比 {pf_txt:>5} | 胜率 {m['win_rate']*100:6.2f}% | "
            f"交易 {m['total_trades']:4d} 笔 | 累计 {m['total_return']*100:+6.1f}%")


def backtest(name: str, factors: list[str], entry: dict, exit_: dict) -> dict:
    exprs = resolve_factor_expressions(rt.database, factors)
    t0 = time.time()
    _trades, m, _eq = run_signal_backtest(rt.data, rt.settings, exprs, entry, exit_, START, END)
    print(f"\n【{name}】 {START} ~ {END}（约 5 年）")
    print(f"  {fmt(m)}   (耗时 {time.time()-t0:.1f}s)")
    return m


def main() -> None:
    print("=" * 100)
    print(f"两个新策略模板 5 年回测 | {START} ~ {END}")
    print("=" * 100)

    backtest(
        "布林带收口·放量突破",
        ["boll_width", "breakout_4", "vol_ratio", "ma20_dev"],
        {"combine": "AND", "conditions": [
            {"factor": "boll_width", "op": "<", "value": 0.08},
            {"factor": "breakout_4", "op": ">", "value": 0.005},
            {"factor": "vol_ratio", "op": ">", "value": 1.5},
            {"factor": "ma20_dev", "op": ">", "value": 0.0},
        ]},
        {"combine": "OR", "conditions": [],
         "stop_loss_pct": 0.0, "take_profit_pct": 0.20,
         "trailing_stop_pct": 0.10, "breakout_exit": False},
    )

    backtest(
        "超卖反转·KDJ回升",
        ["kdj_j", "updown_ratio", "near_low_20", "bias_5"],
        {"combine": "AND", "conditions": [
            {"factor": "kdj_j", "op": "<", "value": 0.0},
            {"factor": "updown_ratio", "op": ">", "value": 0.5},
            {"factor": "near_low_20", "op": "<", "value": 0.15},
            {"factor": "bias_5", "op": ">", "value": 0.0},
        ]},
        {"combine": "OR", "conditions": [],
         "stop_loss_pct": 0.06, "take_profit_pct": 0.15,
         "trailing_stop_pct": 0.0, "breakout_exit": False},
    )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
