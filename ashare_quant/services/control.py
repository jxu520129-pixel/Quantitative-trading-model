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

    def migrate_dashboard_refresh(self, default_seconds: int = 30) -> None:
        """把看板「浮动盈亏刷新间隔」从**旧默认 10 秒**一次性升级到新的默认值。

        该键（``holdings_refresh_seconds``）在看板每次打开时都会被自动写回，
        所以老库里存的 ``10`` 多半不是用户主动选的、而是旧默认值落库的结果。
        用 ``holdings_refresh_version`` 做版本标记，保证只迁移一次——
        迁移后用户自己选的值不会再被覆盖（包括他主动选回 10 秒）。
        """
        if self.get("holdings_refresh_version", "") == "2":
            return
        if self.get("holdings_refresh_seconds", "") in ("", "10"):
            self.set("holdings_refresh_seconds", str(default_seconds))
        self.set("holdings_refresh_version", "2")
