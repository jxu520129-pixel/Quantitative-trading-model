"""多策略组合回测：横截面因子策略 + 涨停回马枪 的组合对比。

在 data/ashare_quant_hist.db（全市场）上跑多个低相关策略的周度调仓回测，
得到各自日度收益序列，再用等权 / 逆波动率（≈风险平价）合并，
输出「单策略 vs 组合」的年化、最大回撤、夏普对比表。

用法：
    python scripts/portfolio_backtest.py
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

import pandas as pd  # noqa: E402

from ashare_quant.database import Database  # noqa: E402
from ashare_quant.limit_pullback_strategy import run_limit_pullback_backtest  # noqa: E402
from ashare_quant.portfolio import (  # noqa: E402
    apply_market_timing,
    combine_returns,
    cross_sectional_returns,
    load_hist_bars,
    market_benchmark,
    summary,
)

HIST_DB = ROOT / "data" / "ashare_quant_hist.db"
START = "2024-01-01"
END = "2026-08-28"
PREHEAT_START = "2023-07-01"  # 因子预热（动量60日、52周新高等需要前置数据）

# 横截面策略：因子规格（因子名, 权重），权重符号=方向（正越高越好，负越低越好）
CROSS_STRATEGIES = {
    "动量轮动": [("momentum", 1.0)],
    "低波动": [("volatility", -1.0), ("atr", -0.5)],
    "趋势跟踪": [("high_52w", 1.0), ("ma_trend", 0.6), ("macd", 0.4)],
    "短期反转": [("reversal", 1.0)],
    "流动性": [("liquidity", 1.0), ("amihud", -0.5)],
    "增强多因子": [("momentum", 0.5), ("reversal", 0.2), ("volatility", -0.4), ("liquidity", 0.25), ("high_52w", 0.3)],
}

TOP_N = 5
# 近20日平均成交额下限。注意：hist 库（Tushare）的 amount 单位是「千元」，
# 与默认演示库（元）不同，故此处 2000 万元 = 20_000 千元。
MIN_AMOUNT = 20_000


def main() -> None:
    close, amount, high, low = load_hist_bars(HIST_DB, start=PREHEAT_START, end=END)
    print(f"数据：{close.shape[0]} 个交易日 × {close.shape[1]} 只股票（{PREHEAT_START} 起预热）\n")

    returns: dict[str, pd.Series] = {}
    for name, specs in CROSS_STRATEGIES.items():
        r = cross_sectional_returns(
            close, amount, high, low, factor_specs=specs, top_n=TOP_N,
            rebalance="W", min_average_amount=MIN_AMOUNT,
        )
        returns[name] = r[r.index >= START]

    # 涨停回马枪（日线近似，用其资金曲线转日收益）
    db = Database(HIST_DB)
    lp = run_limit_pullback_backtest(db, None, start_date=START, end_date=END, persist=False)
    lp_equity = pd.Series({pd.Timestamp(d): v for d, v in lp["equity"]}, dtype=float).sort_index()
    returns["涨停回马枪"] = lp_equity.pct_change().dropna()

    # 组合
    singles = {k: v for k, v in returns.items() if "组合" not in k}
    returns["组合·全部等权"] = combine_returns(singles, "equal")
    returns["组合·全部逆波动率"] = combine_returns(singles, "inverse_vol")
    # 优选：仅纳入回测期内年化 > 0 的策略，避免负收益的追涨因子拖累组合
    positive = {k: v for k, v in singles.items() if summary(v)["annual_return"] > 0}
    best_port = combine_returns(positive, "inverse_vol")
    returns["组合·优选逆波动率"] = best_port
    returns["组合·优选低回撤加权"] = combine_returns(positive, "min_drawdown")

    # 大盘择时：对优选组合应用全市场等权基准的均线择时（熊市降仓）
    bench = market_benchmark(close)
    for window, bear in [(60, 0.3), (60, 0.0), (200, 0.3)]:
        label = f"组合·优选·择时MA{window}仓位{bear:.0%}"
        returns[label] = apply_market_timing(best_port, bench, ma_window=window, bear_exposure=bear)

    # 输出对比表
    print(f"{'策略':<12} {'年化':>9} {'最大回撤':>9} {'夏普':>7}")
    print("-" * 42)
    for name, r in returns.items():
        s = summary(r)
        print(f"{name:<12} {s['annual_return']*100:>8.2f}% {s['max_drawdown']*100:>8.2f}% {s['sharpe']:>7.2f}")

    # 相关性矩阵（单策略之间）
    corr = pd.DataFrame(singles).corr()
    print("\n单策略日收益相关性：")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
