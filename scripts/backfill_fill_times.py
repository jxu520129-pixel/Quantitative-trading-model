"""把 ``fills.filled_at`` 历史的 UTC 错位时间回填为 ``YYYY-MM-DD 09:30:SS`` 形式（开盘成交时间）。

背景
----
``ashare_quant/brokers/paper.py::_fill`` 历史上用 ``utc_now_text()`` 写入 ``filled_at`` 字段，
dashboard 直接把它当本地时间展示，叠加 UTC+8 时差会出现 01:35 / 04:36 这种凌晨时间。
模拟撮合按当日开盘价成交，对应真实交易时段是 09:30 连续竞价开始，因此用 ``trade_date 09:30:SS``
更符合直觉。秒数 SS 由 ``order_id`` 的 md5 哈希分散到 0-59，同日多笔成交的秒数互不重复。

用法
----
::

    python scripts/backfill_fill_times.py                  # 回填默认库 data/ashare_quant.db
    python scripts/backfill_fill_times.py --dry-run        # 仅打印将修改的条数与样例
    python scripts/backfill_fill_times.py --db data/xxx.db # 指定其他库

说明
----
- 一次性的数据修复脚本，跑完可保留作历史档案。
- 新成交已由 ``paper.py`` 用 ``market_open_fill_time`` 直接写出正确时间，无需再回填。
- ``orders.trade_date`` 缺失或为空的记录会跳过（保留原值，避免无依据的改写）。
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "ashare_quant.db"


def _fill_time(trade_date: str, order_id: str) -> str:
    base = (trade_date or "")[:10]
    second = hashlib.md5((order_id or "").encode("utf-8")).digest()[0] % 60
    return f"{base} 09:30:{second:02d}"


def backfill(db_path: Path, dry_run: bool = False) -> tuple[int, int]:
    if not db_path.exists():
        raise SystemExit(f"数据库不存在：{db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT f.id, o.trade_date, f.order_id, f.filled_at "
            "FROM fills f LEFT JOIN orders o ON o.id = f.order_id"
        ).fetchall()
        updates: list[tuple[str, int]] = []
        skipped = 0
        for fill_id, trade_date, order_id, old_value in rows:
            if not trade_date or not order_id:
                skipped += 1
                continue
            new_value = _fill_time(trade_date, order_id)
            if old_value != new_value:
                updates.append((new_value, fill_id))
        print(f"数据库：{db_path}")
        print(f"fills 总条数：{len(rows)}；将回填：{len(updates)}；跳过：{skipped}（缺 trade_date/order_id）")
        if updates:
            samples = updates[:5]
            for new_value, fill_id in samples:
                print(f"  样例 id={fill_id} -> {new_value}")
        if dry_run or not updates:
            return len(updates), len(rows)
        with conn:
            conn.executemany("UPDATE fills SET filled_at=? WHERE id=?", updates)
        print(f"已回填 {len(updates)} 条成交时间。")
        return len(updates), len(rows)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 fills.filled_at 为开盘成交时间")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 数据库路径（默认 data/ashare_quant.db）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将修改的条数与样例，不写库")
    args = parser.parse_args()
    backfill(Path(args.db), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
