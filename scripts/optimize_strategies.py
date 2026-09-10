"""把「按市场风格」优化后的参数应用到 signal_strategies 表（因子实验室策略）。

优化依据：用全市场真实历史库（data/ashare_quant_hist.db）在 2025-09-01 ~ 2026-08-28
对每个策略做参数扫描，取「收益 / 回撤 / 胜率 / 样本数」综合更优者，而非拍脑袋调参。

用法（默认只预演，不会改库）::

    python scripts/optimize_strategies.py            # 预演：打印将发生的改动
    python scripts/optimize_strategies.py --apply    # 实际写入数据库

云端容器内执行::

    docker exec ashare-quant-scheduler-1 python scripts/optimize_strategies.py --apply

说明：
- 按**策略名包含关键词**匹配（例如「箱体突破」），策略改名后仍能命中；
- 只改 `entry_json` 里的指定因子阈值与 `exit_json` 里的指定键，其余字段原样保留；
- 未匹配到的策略会明确列出，不会静默跳过。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "ashare_quant.db"

# 策略名关键词 -> (entry 因子阈值补丁, exit 键补丁, 优化说明)
OPTIMIZATIONS: dict[str, tuple[dict[str, float], dict[str, float], str]] = {
    "箱体突破": (
        {"vol_ratio": 1.5},          # 放量门槛 2.0 → 1.5
        {},                           # 移动止损保持 12%（改 18% 与 1.5 组合后反而变差）
        "vol_ratio 2.0→1.5。【1 年】年化 +44.5%→+49.9%、回撤 10.2%→8.4%；"
        "【3 年复验】年化持平（+19.6%→+19.8%）但回撤 20.8%→14.5%（-6.2pp）。"
        "分年度互有胜负（弱势的 2026 明显更优：-46.6%→-11.0%；2024 强势市略差），"
        "综合为「收益持平、回撤显著改善」，故保留",
    ),
    "超跌反弹": (
        {},
        {},                           # 扫描确认 stop_loss 6% 即最优，不调整
        "经扫描确认 6% 止损已是甜点（4% 仅 +0.65%、8% +14.3%、10% 胜率更高但收益降至 +21.5%），保持不动。"
        "3 年复验：年化 +13.5%、回撤 15.8%、137 笔",
    ),
    "主升浪": (
        {},
        {},                           # 3 年复验推翻了 1 年「加硬止损」的结论，保持原配置
        "【3 年复验后撤回，保持原配置】1 年样本曾显示加 8% 硬止损可由亏转盈（-18.6%→+12.0%），"
        "但 3 年区间该改动反而使年化 +19.3%→+10.4%、回撤 22.7%→30.4%"
        "（止损被反复触发，交易数 101→201 笔、成本抬升），属 1 年样本过拟合，撤回",
    ),
    "涨停回马枪": (
        {},
        {},                           # 扫描确认 threshold 70 最优
        "打分阈值 70 经扫描确认最优（60/65 年化同为 +24.1% 但回撤 -3.2%、胜率 76.9%），保持不动。"
        "3 年复验：年化 +7.8%、回撤 -5.6%、胜率 62.5%、32 笔"
        "（与项目 README 记录的年化 8.79% / 回撤 5.56% / 胜率 62.5% 一致，交叉验证通过）",
    ),
    "人气": (
        {},
        {},                           # 日线近似回测不可信，不做参数改动
        "该策略是盘中实时型（人气榜无法历史回填），日线近似回测系统性偏追涨、结果不可信，"
        "故不据回测调参；建议以模拟盘信号记录验证。阈值 85 会导致全年 0 信号，维持 75。"
        "3 年复验年化 -21.8%、回撤 -51.1%，进一步印证该口径不可用于调参",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="应用策略优化参数")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 数据库路径")
    parser.add_argument("--apply", action="store_true", help="实际写入（默认仅预演）")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"数据库不存在：{db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id,name,entry_json,exit_json FROM signal_strategies ORDER BY created_at DESC").fetchall()
    if not rows:
        raise SystemExit("signal_strategies 表为空：请先在看板「因子实验室 → 策略定义」创建策略")

    matched: set[str] = set()
    updates: list[tuple[str, str, str]] = []
    print("=" * 100)
    for row in rows:
        name = str(row["name"])
        for keyword, (entry_patch, exit_patch, note) in OPTIMIZATIONS.items():
            if keyword not in name:
                continue
            matched.add(keyword)
            entry = json.loads(row["entry_json"]) if row["entry_json"] else {}
            exit_ = json.loads(row["exit_json"]) if row["exit_json"] else {}
            changed: list[str] = []
            for cond in entry.get("conditions", []):
                factor = str(cond.get("factor", ""))
                if factor in entry_patch and float(cond.get("value", 0)) != entry_patch[factor]:
                    changed.append(f"{factor}: {cond.get('value')} → {entry_patch[factor]}")
                    cond["value"] = entry_patch[factor]
            for key, value in exit_patch.items():
                if exit_.get(key) != value:
                    changed.append(f"{key}: {exit_.get(key)} → {value}")
                    exit_[key] = value
            print(f"【{name}】")
            print(f"  优化说明：{note}")
            if changed:
                for item in changed:
                    print(f"  改动：{item}")
                updates.append((json.dumps(entry, ensure_ascii=False),
                                json.dumps(exit_, ensure_ascii=False), name))
            else:
                print("  改动：无（当前参数已是最优）")
            print()
            break

    unmatched = set(OPTIMIZATIONS) - matched
    if unmatched:
        print(f"未匹配到策略（关键词）：{'、'.join(unmatched)}")
        print("  如你的策略名不含这些关键词，请手动对照上面的说明调整参数。\n")

    if not updates:
        print("没有需要写入的改动。")
        conn.close()
        return

    if not args.apply:
        print(f"（预演）将更新 {len(updates)} 个策略；确认无误后加 --apply 实际写入。")
        conn.close()
        return

    with conn:
        for entry_text, exit_text, name in updates:
            conn.execute("UPDATE signal_strategies SET entry_json=?, exit_json=? WHERE name=?",
                         (entry_text, exit_text, name))
    conn.close()
    print(f"已更新 {len(updates)} 个策略的参数。")


if __name__ == "__main__":
    main()
