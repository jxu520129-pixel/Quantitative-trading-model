"""校验最新交易日日线覆盖率，输出详细报告（供手动排查 / 挂 cron 告警）。

用法（容器内，项目根目录）：
    docker compose exec -T scheduler python scripts/check_daily_coverage.py

退出码：0 = 当日数据完整（或数据不足、无法对比，不误报）；1 = 覆盖率低于阈值。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ashare_quant.config import load_settings  # noqa: E402
from ashare_quant.data.service import DataService  # noqa: E402
from ashare_quant.database import Database  # noqa: E402
from ashare_quant.utils import today_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="校验最新交易日日线覆盖率")
    parser.add_argument("--config", help="可选 YAML 配置文件（覆盖 config/default.yaml）")
    parser.add_argument("--db", help="可选数据库路径（默认读配置 db_path）")
    parser.add_argument("--threshold", type=float, help="覆盖率阈值（默认读配置 min_daily_coverage）")
    args = parser.parse_args()

    settings = load_settings(args.config)
    database = Database(Path(args.db) if args.db else Path(settings.db_path))
    data = DataService(database, settings)
    report = data.daily_coverage_report(args.threshold)

    print("=" * 56)
    print("最新交易日日线覆盖率校验")
    print("=" * 56)
    print(f"最新交易日   : {report['latest_date'] or '（无）'}   （今天 {today_text()}）")
    print(f"当日条数     : {report['latest_count']}")
    print(f"前一交易日   : {report['previous_date'] or '（无）'}   （条数 {report['previous_count']}）")
    if report["coverage"] is None:
        print(f"覆盖率       : 无法计算 —— {report['reason']}")
    else:
        print(f"覆盖率       : {report['coverage']:.2%}   （阈值 {report['threshold']:.0%}）")
    print("-" * 56)
    print(f"结论         : {'完整 ✅' if report['complete'] else '不完整 ❌'}")
    if report["reason"] and not report["complete"]:
        print(f"原因         : {report['reason']}")
    print("=" * 56)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
