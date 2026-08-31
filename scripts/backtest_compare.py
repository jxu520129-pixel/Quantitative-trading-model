"""短线策略 优化前 vs 优化后 回测对比（最终版）。

用法（在项目根目录，需已拉取历史库 data/ashare_quant_hist.db）：
    python scripts/backtest_compare.py [--start 2024-01-01] [--end 2026-08-28]

- 策略1（短线人气·热度共振）：exit_mode=legacy（原退出）vs optimized（分批止盈+移动止盈）
- 策略2（涨停回马枪）：当前默认（已含 bug 修复 + 信号②放量 2.0x 最优参数）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key in {"TUSHARE_TOKEN", "TUSHARE_API_URL"} and value and key not in os.environ:
            os.environ[key] = value


def _fmt(m: dict) -> str:
    return (
        f"年化 {m['annual_return']*100:8.2f}% | 回撤 {m['max_drawdown']*100:7.2f}% | "
        f"夏普 {m['sharpe']:6.2f} | 胜率 {m['win_rate']*100:6.2f}% | "
        f"交易 {m['total_trades']:4d} 笔 | 期末净值 {m['final_equity']:,.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-08-28")
    parser.add_argument("--db", default=str(ROOT / "data" / "ashare_quant_hist.db"))
    args = parser.parse_args()

    _load_env()
    from ashare_quant.database import Database
    from ashare_quant.hot_strategy import run_hot_backtest
    from ashare_quant.limit_pullback_strategy import run_limit_pullback_backtest

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"历史库不存在：{db_path}，请先运行 scripts/fetch_hist_data.py", file=sys.stderr)
        sys.exit(1)
    db = Database(db_path)

    print("=" * 90)
    print(f"回测区间：{args.start} ~ {args.end}")
    print("=" * 90)

    print("\n【策略1】短线人气·热度共振（threshold=75, 单票90%仓位, 每周≤3笔）")
    for mode in ("legacy", "optimized"):
        m = run_hot_backtest(
            db, None, start_date=args.start, end_date=args.end,
            threshold=75.0, max_positions=1, initial_cash=100_000.0,
            daily_budget=0.9, sentiment_scope="hs", exit_mode=mode, persist=False,
        )
        print(f"  {mode:>9} : {_fmt(m)}")

    print("\n【策略2】涨停回马枪·冲高回调低吸（threshold=70, 最多3只, 单日≤40%仓位）")
    m = run_limit_pullback_backtest(
        db, None, start_date=args.start, end_date=args.end,
        max_positions=3, initial_cash=1_000_000.0,
        daily_budget=0.4, persist=False,
    )
    print(f"  优化后默认 : {_fmt(m)}")


if __name__ == "__main__":
    main()
