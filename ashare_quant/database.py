"""SQLite storage with explicit schema and parameterized statements."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .models import utc_now_text


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS stock_basic (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, security_type TEXT NOT NULL,
    board TEXT NOT NULL DEFAULT 'MAIN', is_st INTEGER NOT NULL DEFAULT 0,
    is_delisted INTEGER NOT NULL DEFAULT 0, is_suspended INTEGER NOT NULL DEFAULT 0,
    list_date TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS universe_members (
    universe_code TEXT NOT NULL, code TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY(universe_code, code)
);
CREATE TABLE IF NOT EXISTS daily_bars (
    code TEXT NOT NULL, trade_date TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL DEFAULT 0,
    amount REAL NOT NULL DEFAULT 0, pre_close REAL, source TEXT NOT NULL,
    PRIMARY KEY (code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(trade_date);
CREATE TABLE IF NOT EXISTS market_quotes (
    code TEXT PRIMARY KEY, quote_time TEXT NOT NULL, price REAL NOT NULL,
    pre_close REAL, volume REAL NOT NULL DEFAULT 0, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, code TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', action TEXT NOT NULL,
    as_of_date TEXT NOT NULL, strategy TEXT NOT NULL, target_weight REAL NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'NEW',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(as_of_date, strategy);
CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_action ON signals(code,action,as_of_date,strategy);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, code TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', side TEXT NOT NULL,
    quantity INTEGER NOT NULL, requested_price REAL, fill_price REAL, filled_quantity INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL, strategy TEXT NOT NULL, trade_date TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, trade_date);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, code TEXT NOT NULL, side TEXT NOT NULL,
    quantity INTEGER NOT NULL, price REAL NOT NULL, gross_amount REAL NOT NULL, commission REAL NOT NULL,
    stamp_duty REAL NOT NULL, filled_at TEXT NOT NULL, FOREIGN KEY(order_id) REFERENCES orders(id)
);
CREATE TABLE IF NOT EXISTS positions (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, quantity INTEGER NOT NULL, sellable_quantity INTEGER NOT NULL,
    avg_cost REAL NOT NULL, latest_price REAL NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_state (
    account_id TEXT PRIMARY KEY, cash REAL NOT NULL, market_value REAL NOT NULL,
    total_equity REAL NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT NOT NULL, snapshot_date TEXT NOT NULL,
    cash REAL NOT NULL, market_value REAL NOT NULL, total_equity REAL NOT NULL,
    daily_pnl REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    UNIQUE(account_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_account_date ON account_snapshots(account_id, snapshot_date);
CREATE TABLE IF NOT EXISTS risk_state (
    account_id TEXT PRIMARY KEY, failure_count INTEGER NOT NULL DEFAULT 0,
    daily_open_blocked INTEGER NOT NULL DEFAULT 0, paused INTEGER NOT NULL DEFAULT 0,
    last_reset_date TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, event_time TEXT NOT NULL, level TEXT NOT NULL,
    category TEXT NOT NULL, message TEXT NOT NULL, code TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id TEXT PRIMARY KEY, strategy TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
    initial_cash REAL NOT NULL, final_equity REAL NOT NULL, annual_return REAL NOT NULL,
    max_drawdown REAL NOT NULL, sharpe REAL, win_rate REAL NOT NULL, total_trades INTEGER NOT NULL,
    parameters_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtest_equity (
    run_id TEXT NOT NULL, trade_date TEXT NOT NULL, equity REAL NOT NULL,
    PRIMARY KEY(run_id, trade_date), FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
);
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_factors (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, expression TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_strategies (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, entry_json TEXT NOT NULL,
    exit_json TEXT NOT NULL, created_at TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS signal_backtest_runs (
    id TEXT PRIMARY KEY, strategy_name TEXT NOT NULL, start_date TEXT NOT NULL,
    end_date TEXT NOT NULL, metrics_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, code TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '', entry_date TEXT NOT NULL, entry_price REAL NOT NULL,
    shares INTEGER NOT NULL, exit_date TEXT, exit_price REAL, pnl REAL, pnl_pct REAL,
    holding_days INTEGER, status TEXT NOT NULL DEFAULT 'OPEN',
    entry_reason TEXT NOT NULL DEFAULT '', exit_reason TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(run_id) REFERENCES signal_backtest_runs(id)
);
CREATE TABLE IF NOT EXISTS stock_industry (
    code TEXT PRIMARY KEY, industry TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
    event_time TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL DEFAULT '其他',
    source TEXT NOT NULL DEFAULT '', source_level TEXT NOT NULL DEFAULT 'B',
    magnitude INTEGER NOT NULL DEFAULT 1, causal_direction TEXT NOT NULL DEFAULT '中性',
    affected_industries TEXT NOT NULL DEFAULT '', persistence TEXT NOT NULL DEFAULT '短期',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, event_time);
CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator TEXT NOT NULL, period TEXT NOT NULL, value REAL NOT NULL,
    updated_at TEXT NOT NULL, PRIMARY KEY(indicator, period)
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trained_at TEXT NOT NULL,
    samples INTEGER NOT NULL, positive_rate REAL NOT NULL,
    validation_auc REAL NOT NULL, validation_accuracy REAL NOT NULL,
    backend TEXT NOT NULL, parameters_json TEXT NOT NULL DEFAULT '{}'
);
"""


class Database:
    """围绕单个 SQLite 数据库文件的线程安全访问层（RLock 串行化写 + WAL）。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        """新建一个启用 WAL、外键与忙等待的连接（每次操作独立连接）。"""
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self, initial_cash: float) -> None:
        """建库建表、执行旧库列迁移并写入模拟账户/风控/开关初始状态（幂等）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)
            # 迁移：为旧库补充 signal_strategies.enabled 列
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(signal_strategies)")}
            if "enabled" not in columns:
                conn.execute("ALTER TABLE signal_strategies ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
            # 迁移：为旧库补充 events 的 causal_direction / affected_industries 列
            event_cols = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
            if "causal_direction" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN causal_direction TEXT NOT NULL DEFAULT '中性'")
            if "affected_industries" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN affected_industries TEXT NOT NULL DEFAULT ''")
            if "persistence" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN persistence TEXT NOT NULL DEFAULT '短期'")
            now = utc_now_text()
            conn.execute(
                "INSERT OR IGNORE INTO account_state(account_id,cash,market_value,total_equity,updated_at) VALUES(?,?,?,?,?)",
                ("paper", initial_cash, 0.0, initial_cash, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO risk_state(account_id,updated_at) VALUES(?,?)", ("paper", now)
            )
            conn.executemany(
                "INSERT OR IGNORE INTO system_settings(key,value,updated_at) VALUES(?,?,?)",
                [("strategy_enabled", "true", now), ("paper_execution_enabled", "true", now)],
            )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """执行单条写语句（自动提交）。"""
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> None:
        """批量执行写语句（自动提交）。"""
        with self._lock, self.connect() as conn:
            conn.executemany(sql, params)

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        """查询单行，返回字典（无结果返回 None）。"""
        with self._lock, self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """查询多行，返回字典列表。"""
        with self._lock, self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Use as a context manager only for multi-step atomic state changes."""
        return _Transaction(self)


class _Transaction:
    """多步原子事务上下文：进入时加锁并 BEGIN IMMEDIATE，退出时提交/回滚。"""

    def __init__(self, database: Database):
        self.database = database
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.database._lock.acquire()
        self.connection = self.database.connect()
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.connection is not None
        try:
            if exc_type:
                self.connection.rollback()
            else:
                self.connection.commit()
        finally:
            self.connection.close()
            self.database._lock.release()
