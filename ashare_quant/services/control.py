"""Persistent runtime switches shared by scheduler and dashboard processes."""

from __future__ import annotations

from ..database import Database
from ..models import utc_now_text


class SystemControl:
    def __init__(self, database: Database):
        self.database = database

    def get_bool(self, key: str, default: bool = False) -> bool:
        return self.get(key, str(default).lower()).lower() == "true"

    def get(self, key: str, default: str = "") -> str:
        row = self.database.query_one("SELECT value FROM system_settings WHERE key=?", (key,))
        return default if not row else str(row["value"])

    def set_bool(self, key: str, value: bool) -> None:
        self.set(key, str(bool(value)).lower())

    def set(self, key: str, value: str) -> None:
        self.database.execute(
            """INSERT INTO system_settings(key,value,updated_at) VALUES(?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",
            (key, value, utc_now_text()),
        )
