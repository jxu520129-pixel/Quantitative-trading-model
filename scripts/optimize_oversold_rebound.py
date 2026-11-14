"""超跌反弹·量比确认 策略参数优化（回撤优先 + 长区间复验）。

策略定义见 ashare_quant/lab.py 的 BUILTIN_STRATEGY_TEMPLATES["超跌反弹·量比确认"]：
  距20日低点<near_low(超跌) + 5日上涨比例>up_days(止跌) + 20日线偏离>ma20_dev + 量比>vol_ratio(放量确认)；
  固定止损 stop_loss / 止盈 take_profit（无移动止损）。

用法（项目根目录，需已拉取历史库 data/ashare_quant_hist.db）：
    python scripts/optimize_oversold_rebound.py

- 阶段A：坐标下降（回撤优先 pick），主区间 2024-01-01 ~ 2026-08-28
- 阶段B：候选组合放长区间 2020-01-01 ~ 2026-08-28 复验，防单区间过拟合
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
BASE = dict(near_low=0.15, up_days=0.6, ma20_dev=-0.02, vol_ratio=1.2, stop=0.06, take=0.15)

rt = build_runtime(prefer_hist_db=True)
rt.settings.raw.setdefault("scan", {})["min_average_amount"] = 1e5
exprs = resolve_factor_expressions(rt.database, ["near_low_20", "up_days_5", "ma20_dev", "vol_ratio"])


def run(near_low: float, up_days: float, ma20_dev: float, vol_ratio: float,
        stop: float, take: float, start: str = START, end: str = END) -> dict:
    entry = {"combine": "AND", "conditions": [
        {"factor": "near_low_20", "op": "<", "value": near_low},
        {"factor": "up_days_5", "op": ">", "value": up_days},
        {"factor": "ma20_dev", "op": ">", "value": ma20_dev},
        {"factor": "vol_ratio", "op": ">", "value": vol_ratio},
    ]}
    exit_ = {"combine": "OR", "conditions": [],
             "stop_loss_pct": stop, "take_profit_pct": take,
             "trailing_stop_pct": 0.0, "breakout_exit": False}
    _t, m, _e = run_signal_backtest(rt.data, rt.settings, exprs, entry, exit_, start, end)
    return m


def fmt(m: dict) -> str:
    pf = m["profit_factor"]
    pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
    return (f"年化 {m['annual_return']*100:6.2f}% | 回撤 {m['max_drawdown']*100:6.2f}% | "
            f"盈亏比 {pf_txt:>5} | 胜率 {m['win_rate']*100:5.2f}% | 交易 {m['total_trades']:4d}")


def pick(results: list[tuple[dict, dict]]) -> dict:
    """回撤优先：年化不低于最优 85% 的候选里选回撤最低者（防贪心过拟合）。"""
    best_ret = max(r[1]["annual_return"] for r in results)
    candidates = [r for r in results if r[1]["annual_return"] >= best_ret * 0.85]
    return min(candidates, key=lambda r: r[1]["max_drawdown"])[0]


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print(f"超跌反弹·量比确认 参数优化 | 主区间 {START} ~ {END}")
    print("=" * 100)

    base_m = run(**BASE)
    print(f"\n【基准】near_low={BASE['near_low']} up_days={BASE['up_days']} "
          f"ma20_dev={BASE['ma20_dev']} vol_ratio={BASE['vol_ratio']} "
          f"stop={BASE['stop']:.0%} take={BASE['take']:.0%}")
    print(f"  {fmt(base_m)}")

    cur = dict(BASE)

    # ① 止损
    print("\n" + "-" * 100 + "\n① stop_loss（固定止损）")
    cand = []
    for s in (0.04, 0.06, 0.08, 0.10, 0.12):
        m = run(near_low=cur["near_low"], up_days=cur["up_days"], ma20_dev=cur["ma20_dev"],
                vol_ratio=cur["vol_ratio"], stop=s, take=cur["take"])
        cand.append(({"stop": s}, m))
        print(f"  stop={s:.0%} {fmt(m)}")
    cur["stop"] = pick(cand)["stop"]
    print(f"  → 选 stop={cur['stop']:.0%}")

    # ② 止盈
    print("\n" + "-" * 100 + "\n② take_profit（止盈）")
    cand = []
    for t in (0.10, 0.15, 0.20, 0.25):
        m = run(near_low=cur["near_low"], up_days=cur["up_days"], ma20_dev=cur["ma20_dev"],
                vol_ratio=cur["vol_ratio"], stop=cur["stop"], take=t)
        cand.append(({"take": t}, m))
        print(f"  take={t:.0%} {fmt(m)}")
    cur["take"] = pick(cand)["take"]
    print(f"  → 选 take={cur['take']:.0%}")

    # ③ 量比（放量确认强度）
    print("\n" + "-" * 100 + "\n③ vol_ratio（放量门槛）")
    cand = []
    for v in (1.0, 1.2, 1.5, 2.0):
        m = run(near_low=cur["near_low"], up_days=cur["up_days"], ma20_dev=cur["ma20_dev"],
                vol_ratio=v, stop=cur["stop"], take=cur["take"])
        cand.append(({"vol_ratio": v}, m))
        print(f"  vol_ratio={v:<4} {fmt(m)}")
    cur["vol_ratio"] = pick(cand)["vol_ratio"]
    print(f"  → 选 vol_ratio={cur['vol_ratio']}")

    # ④ 超跌深度
    print("\n" + "-" * 100 + "\n④ near_low_20（距20日低点上限，越小越超跌）")
    cand = []
    for n in (0.05, 0.10, 0.15, 0.20):
        m = run(near_low=n, up_days=cur["up_days"], ma20_dev=cur["ma20_dev"],
                vol_ratio=cur["vol_ratio"], stop=cur["stop"], take=cur["take"])
        cand.append(({"near_low": n}, m))
        print(f"  near_low={n:<4} {fmt(m)}")
    cur["near_low"] = pick(cand)["near_low"]
    print(f"  → 选 near_low={cur['near_low']}")

    # 阶段B：长区间复验（基准 vs 坐标下降最优 vs 2 个关键候选）
    print("\n" + "=" * 100)
    print(f"阶段B：长区间复验 {LONG_START} ~ {LONG_END}")
    print("=" * 100)
    combos = [
        ("基准", dict(BASE)),
        ("坐标下降最优", dict(cur)),
        ("止损8%变体", {**cur, "stop": 0.08}),
        ("量比1.0变体", {**cur, "vol_ratio": 1.0}),
    ]
    for name, c in combos:
        m_long = run(**c, start=LONG_START, end=LONG_END)
        m_short = run(**c)
        print(f"\n【{name}】{c}")
        print(f"  长区间 2020~2026  {fmt(m_long)}")
        print(f"  短区间 2024~2026  {fmt(m_short)}")

    print("\n" + "=" * 100)
    print(f"结论：基准 stop={BASE['stop']:.0%} take={BASE['take']:.0%} "
          f"vol_ratio={BASE['vol_ratio']} near_low={BASE['near_low']}")
    print(f"      优化 stop={cur['stop']:.0%} take={cur['take']:.0%} "
          f"vol_ratio={cur['vol_ratio']} near_low={cur['near_low']}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
