"""箱体突破·放量确认 策略参数优化扫描（坐标下降 + 长区间复验）。

策略定义见 ashare_quant/lab.py 的 BUILTIN_STRATEGY_TEMPLATES["箱体突破·放量确认"]：
  收盘突破前4日高点 + 量比>vol_ratio + 站上20日线 + 5日箱体振幅<box_range_5；
  移动止损 trailing_stop / 止盈 take_profit（无固定止损）。

用法（项目根目录，需已拉取历史库 data/ashare_quant_hist.db）：
    python scripts/optimize_box_breakout.py

- 阶段A：坐标下降，区间 2024-01-01 ~ 2026-08-28（与历史 optimize 口径一致，2.5 年）
- 阶段B：把阶段A的最优组合放到 2020-01-01 ~ 2026-08-28（6.5 年）复验，防单区间过拟合
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

START, END = "2024-01-01", "2026-08-28"
LONG_START, LONG_END = "2020-01-01", "2026-08-28"

# 基准参数（当前内置模板值）
BASE = dict(vol_ratio=2.0, trailing=0.12, take=0.30, box=0.15)

rt = build_runtime(prefer_hist_db=True)
# hist 库 amount 单位是「千元」，把 config 里「元」口径的成交额阈值换算成千元口径
# （否则 1e8 元阈值会把几乎所有股票过滤掉，回测 0 笔）
rt.settings.raw.setdefault("scan", {})["min_average_amount"] = 1e5

FACTORS = ["breakout_4", "vol_ratio", "ma20_dev", "box_range_5"]
exprs = resolve_factor_expressions(rt.database, FACTORS)


def run(vol_ratio: float, trailing: float, take: float, box: float,
        start: str = START, end: str = END) -> dict:
    entry = {"combine": "AND", "conditions": [
        {"factor": "breakout_4", "op": ">", "value": 0.005},
        {"factor": "vol_ratio", "op": ">", "value": vol_ratio},
        {"factor": "ma20_dev", "op": ">", "value": 0.0},
        {"factor": "box_range_5", "op": "<", "value": box},
    ]}
    exit_ = {"combine": "OR", "conditions": [],
             "stop_loss_pct": 0.0, "take_profit_pct": take,
             "trailing_stop_pct": trailing, "breakout_exit": False}
    _trades, m, _eq = run_signal_backtest(rt.data, rt.settings, exprs, entry, exit_, start, end)
    return m


def fmt(m: dict) -> str:
    pf = m["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:5.2f}"
    return (f"年化 {m['annual_return']*100:7.2f}% | 回撤 {m['max_drawdown']*100:7.2f}% | "
            f"盈亏比 {pf_txt:>5} | 胜率 {m['win_rate']*100:6.2f}% | "
            f"交易 {m['total_trades']:4d} 笔 | 累计 {m['total_return']*100:+6.1f}%")


def pick(results: list[tuple[dict, dict]], metric_key: str = "annual_return") -> dict:
    """回撤优先选优：在年化收益不低于最优 85% 的候选里，选回撤最低者。

    纯按年化选优会过拟合（2026-11 踩过：贪心把 box→20%、trailing→8%，
    长区间回撤从 37.6% 爆到 52.65%）。坐标下降只是粗扫定位甜点区间，
    最终参数必须以 `optimize_box_breakout_long.py` 的长区间复验为准。
    """
    best_ret = max(r[1][metric_key] for r in results)
    candidates = [r for r in results if r[1][metric_key] >= best_ret * 0.85]
    return min(candidates, key=lambda r: r[1]["max_drawdown"])[0]


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print(f"箱体突破·放量确认 参数优化 | 主区间 {START} ~ {END}")
    print("=" * 100)

    # 基准
    base_m = run(**BASE)
    print(f"\n【基准】vol_ratio={BASE['vol_ratio']} trailing={BASE['trailing']:.0%} "
          f"take={BASE['take']:.0%} box={BASE['box']:.0%}")
    print(f"  {fmt(base_m)}   (单组耗时 {time.time()-t0:.1f}s)")

    cur = dict(BASE)

    # 阶段A1：放量门槛 vol_ratio
    print("\n" + "-" * 100)
    print("① vol_ratio（放量门槛，其余固定）")
    cand = []
    for v in (1.2, 1.5, 2.0, 2.5, 3.0):
        m = run(vol_ratio=v, trailing=cur["trailing"], take=cur["take"], box=cur["box"])
        cand.append(({"vol_ratio": v}, m))
        print(f"  vol_ratio={v:<4} {fmt(m)}")
    cur["vol_ratio"] = pick(cand)["vol_ratio"]
    print(f"  → 选 vol_ratio={cur['vol_ratio']}")

    # 阶段A2：移动止损 trailing
    print("\n" + "-" * 100)
    print("② trailing_stop（移动止损，其余固定）")
    cand = []
    for t in (0.06, 0.08, 0.10, 0.12, 0.15, 0.18):
        m = run(vol_ratio=cur["vol_ratio"], trailing=t, take=cur["take"], box=cur["box"])
        cand.append(({"trailing": t}, m))
        print(f"  trailing={t:.0%} {fmt(m)}")
    cur["trailing"] = pick(cand)["trailing"]
    print(f"  → 选 trailing={cur['trailing']:.0%}")

    # 阶段A3：止盈 take
    print("\n" + "-" * 100)
    print("③ take_profit（止盈，其余固定）")
    cand = []
    for t in (0.20, 0.25, 0.30, 0.40, 0.50):
        m = run(vol_ratio=cur["vol_ratio"], trailing=cur["trailing"], take=t, box=cur["box"])
        cand.append(({"take": t}, m))
        print(f"  take={t:.0%} {fmt(m)}")
    cur["take"] = pick(cand)["take"]
    print(f"  → 选 take={cur['take']:.0%}")

    # 阶段A4：箱体振幅上限 box
    print("\n" + "-" * 100)
    print("④ box_range_5（箱体振幅上限，其余固定）")
    cand = []
    for b in (0.10, 0.12, 0.15, 0.18, 0.20):
        m = run(vol_ratio=cur["vol_ratio"], trailing=cur["trailing"], take=cur["take"], box=b)
        cand.append(({"box": b}, m))
        print(f"  box={b:<4} {fmt(m)}")
    cur["box"] = pick(cand)["box"]
    print(f"  → 选 box={cur['box']:.0%}")

    # 阶段B：长区间复验
    print("\n" + "=" * 100)
    print(f"阶段B：长区间复验 {LONG_START} ~ {LONG_END}（防单区间过拟合）")
    print("=" * 100)
    base_long = run(**BASE, start=LONG_START, end=LONG_END)
    opt_long = run(**cur, start=LONG_START, end=LONG_END)
    print(f"  基准   {fmt(base_long)}")
    print(f"  优化   {fmt(opt_long)}")

    print("\n" + "=" * 100)
    print(f"结论：基准 vol_ratio={BASE['vol_ratio']} trailing={BASE['trailing']:.0%} "
          f"take={BASE['take']:.0%} box={BASE['box']:.0%}")
    print(f"      优化 vol_ratio={cur['vol_ratio']} trailing={cur['trailing']:.0%} "
          f"take={cur['take']:.0%} box={cur['box']:.0%}")
    print(f"总耗时 {time.time()-t0:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
