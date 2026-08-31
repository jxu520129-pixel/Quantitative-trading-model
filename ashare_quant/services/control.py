"""Persistent runtime switches shared by scheduler and dashboard processes."""

from __future__ import annotations

from ..database import Database
from ..models import utc_now_text


class SystemControl:
    """持久化运行开关：调度器与看板进程共享 system_settings 表，跨进程同步策略/成交开关。"""

    def __init__(self, database: Database):
        self.database = database

    def get_bool(self, key: str, default: bool = False) -> bool:
        """读取布尔型开关，缺失时返回默认值。"""
        return self.get(key, str(default).lower()).lower() == "true"

    def get(self, key: str, default: str = "") -> str:
        """读取字符串型设置，缺失时返回默认值。"""
        row = self.database.query_one("SELECT value FROM system_settings WHERE key=?", (key,))
        return default if not row else str(row["value"])

    def set_bool(self, key: str, value: bool) -> None:
        """写入布尔型开关（存储为 ``true``/``false`` 小写字符串）。"""
        self.set(key, str(bool(value)).lower())

    def set(self, key: str, value: str) -> None:
        """写入字符串型设置（存在则覆盖，不存在则插入）。"""
        self.database.execute(
            """INSERT INTO system_settings(key,value,updated_at) VALUES(?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",
            (key, value, utc_now_text()),
        )
