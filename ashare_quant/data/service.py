"""Local cache, cleaning, provider fallback and universe selection."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable

import pandas as pd

from ..config import Settings
from ..database import Database
from ..market_rules import is_supported_security
from ..utils import normalize_date, today_text
from .providers import AkShareProvider, MarketDataProvider, TushareProvider


LOG = logging.getLogger(__name__)


class DataService:
    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings
        self.primary: MarketDataProvider = AkShareProvider()
        self.fallback: MarketDataProvider | None = (
            TushareProvider(settings.tushare_token, settings.tushare_api_url) if settings.tushare_token else None
        )

    def refresh_universe(self) -> int:
        """Fetch basic securities, retaining V1 supported instruments only."""
        frame = self._with_fallback("list_securities")
        if frame.empty:
            return 0
        rows = []
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        for item in frame.to_dict("records"):
            code = str(item["code"]).zfill(6)
            security_type = str(item.get("security_type", "STOCK")).upper()
            if not is_supported_security(code, security_type, include_etf=bool(self.settings.data["include_etf"])):
                continue
            rows.append((
                code, str(item.get("name", code)), security_type, str(item.get("board", "MAIN")),
                int(bool(item.get("is_st", False))), 0, 0, str(item.get("list_date", "")), now,
            ))
        self.database.executemany(
            """INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,is_suspended,list_date,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(code) DO UPDATE SET
               name=excluded.name, security_type=excluded.security_type, board=excluded.board,
               is_st=excluded.is_st, list_date=excluded.list_date, updated_at=excluded.updated_at""",
            rows,
        )
        try:
            members = self.primary.index_constituents(str(self.settings.data["stock_universe_index"]))  # type: ignore[attr-defined]
            codes = list(dict.fromkeys(str(code).zfill(6) for code in members["code"].tolist()))
            self.database.execute("DELETE FROM universe_members WHERE universe_code=?", (str(self.settings.data["stock_universe_index"]),))
            self.database.executemany(
                "INSERT OR IGNORE INTO universe_members(universe_code,code,updated_at) VALUES(?,?,?)",
                [(str(self.settings.data["stock_universe_index"]), code, now) for code in codes],
            )
        except Exception as error:
            LOG.warning("无法获取指数成分股，保留现有或演示证券池：%s", error)
        return len(rows)

    def update_daily(self, codes: Iterable[str] | None = None, end_date: str | None = None) -> dict[str, int]:
        """增量更新日线：并发拉取（并发数由 data.update_concurrency 控制），失败标的隔离不影响整体。"""
        if codes is None:
            eligible = self.eligible_universe(limit=int(self.settings.data["max_update_symbols"]))
            codes = eligible["code"].tolist()
        normalized = [str(code).zfill(6) for code in codes]
        # 预取证券类型与各标的最后交易日，避免逐标的两次查询。
        security_types = {
            row["code"]: row["security_type"]
            for row in self.database.query_all("SELECT code, security_type FROM stock_basic")
        }
        last_dates = {
            row["code"]: row["last_date"]
            for row in self.database.query_all("SELECT code, MAX(trade_date) AS last_date FROM daily_bars GROUP BY code")
        }

        def _update_one(code: str) -> dict[str, int]:
            security_type = security_types.get(code, "STOCK")
            previous = last_dates.get(code)
            if previous:
                start = (pd.Timestamp(previous) + timedelta(days=1)).date().isoformat()
            else:
                start = str(self.settings.data["history_start"])
            try:
                frame = self._with_fallback("daily_bars", code, start, end_date or today_text(), security_type)
                count = self.store_bars(code, frame)
                return {"updated": 1, "bars": count, "failed": 0}
            except Exception as error:
                LOG.exception("证券 %s 日线更新失败：%s", code, error)
                return {"updated": 0, "bars": 0, "failed": 1}

        concurrency = int(self.settings.data.get("update_concurrency", 4) or 4)
        result = {"updated_symbols": 0, "bars": 0, "failed": 0}
        if concurrency <= 1 or len(normalized) <= 1:
            for code in normalized:
                item = _update_one(code)
                result["updated_symbols"] += item["updated"]
                result["bars"] += item["bars"]
                result["failed"] += item["failed"]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_update_one, code) for code in normalized]
                for future in as_completed(futures):
                    item = future.result()
                    result["updated_symbols"] += item["updated"]
                    result["bars"] += item["bars"]
                    result["failed"] += item["failed"]
        return result

    def store_bars(self, code: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        cleaned = frame.copy()
        cleaned["trade_date"] = cleaned["trade_date"].map(normalize_date)
        cleaned = cleaned.drop_duplicates("trade_date", keep="last")
        rows = [
            (code, item.trade_date, float(item.open), float(item.high), float(item.low), float(item.close),
             float(item.volume or 0), float(item.amount or 0), None if pd.isna(item.pre_close) else float(item.pre_close), item.source)
            for item in cleaned.itertuples(index=False)
        ]
        self.database.executemany(
            """INSERT INTO daily_bars(code,trade_date,open,high,low,close,volume,amount,pre_close,source)
               VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(code,trade_date) DO UPDATE SET
               open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
               volume=excluded.volume, amount=excluded.amount, pre_close=excluded.pre_close, source=excluded.source""",
            rows,
        )
        return len(rows)

    def load_bars(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        clauses = ["code=?"]
        params: list[str] = [code]
        if start_date:
            clauses.append("trade_date>=?")
            params.append(normalize_date(start_date))
        if end_date:
            clauses.append("trade_date<=?")
            params.append(normalize_date(end_date))
        frame = pd.DataFrame(self.database.query_all(
            f"SELECT trade_date,open,high,low,close,volume,amount,pre_close FROM daily_bars WHERE {' AND '.join(clauses)} ORDER BY trade_date",
            params,
        ))
        if frame.empty:
            return frame
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame.set_index("trade_date", drop=False)

    def load_bars_many(self, codes: Iterable[str], start_date: str | None = None, end_date: str | None = None) -> dict[str, pd.DataFrame]:
        """一次性批量加载多只标的日线，返回 ``{code: DataFrame}``。

        形状与 ``load_bars`` 一致（以 trade_date 为索引，并保留 trade_date 列）；
        无数据的标的不会出现在返回字典中。分批构造 IN 子句以规避 SQLite 变量上限。
        """
        codes = list(dict.fromkeys(str(code).zfill(6) for code in codes))
        if not codes:
            return {}
        clauses: list[str] = []
        params: list[str] = []
        if start_date:
            clauses.append("trade_date>=?")
            params.append(normalize_date(start_date))
        if end_date:
            clauses.append("trade_date<=?")
            params.append(normalize_date(end_date))
        columns = ["trade_date", "open", "high", "low", "close", "volume", "amount", "pre_close"]
        frames: dict[str, pd.DataFrame] = {}
        for i in range(0, len(codes), 500):
            chunk = codes[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            sql = f"SELECT code,{','.join(columns)} FROM daily_bars WHERE code IN ({placeholders})"
            if clauses:
                sql += " AND " + " AND ".join(clauses)
            sql += " ORDER BY code, trade_date"
            rows = self.database.query_all(sql, [*chunk, *params])
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            for code, sub in frame.groupby("code", sort=False):
                frames[code] = sub[columns].set_index("trade_date", drop=False)
        return frames

    def latest_bar(self, code: str) -> dict[str, object] | None:
        return self.database.query_one("SELECT * FROM daily_bars WHERE code=? ORDER BY trade_date DESC LIMIT 1", (code,))

    def refresh_quotes(self, codes: list[str]) -> int:
        """Refresh the 09:35 execution snapshot used by the paper broker."""
        frame = self.primary.realtime_quotes(codes)
        if frame.empty:
            return 0
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for item in frame.to_dict("records"):
            price = pd.to_numeric(item.get("price"), errors="coerce")
            if pd.isna(price) or float(price) <= 0:
                continue
            rows.append((
                str(item["code"]).zfill(6), now, float(price),
                None if pd.isna(item.get("pre_close")) else float(item.get("pre_close")),
                float(item.get("volume", 0) or 0), self.primary.name,
            ))
        self.database.executemany(
            """INSERT INTO market_quotes(code,quote_time,price,pre_close,volume,source) VALUES(?,?,?,?,?,?)
               ON CONFLICT(code) DO UPDATE SET quote_time=excluded.quote_time,price=excluded.price,
               pre_close=excluded.pre_close,volume=excluded.volume,source=excluded.source""",
            rows,
        )
        return len(rows)

    def eligible_universe(self, limit: int | None = None) -> pd.DataFrame:
        """候选池 = 指数成分股 + 全部 ETF，剔除 ST/退市/停牌。

        优先按 ``stock_universe_index`` 的成分股过滤；若尚未写入成分数据（如演示行情或
        指数接口失败），则退化为全部 A 股 + ETF，保证系统可用。
        """
        universe_code = str(self.settings.data["stock_universe_index"])
        has_members = bool(self.database.query_one(
            "SELECT 1 FROM universe_members WHERE universe_code=?", (universe_code,)
        ))
        query = """SELECT code,name,security_type,board FROM stock_basic
                   WHERE is_st=0 AND is_delisted=0 AND is_suspended=0
                   AND security_type IN ('STOCK','ETF')"""
        params: list[str] = []
        if has_members:
            query += """ AND (security_type='ETF' OR code IN
                       (SELECT code FROM universe_members WHERE universe_code=?))"""
            params.append(universe_code)
        query += " ORDER BY CASE WHEN security_type='STOCK' THEN 0 ELSE 1 END, code"
        if limit:
            query += f" LIMIT {int(limit)}"
        return pd.DataFrame(self.database.query_all(query, params))

    def has_data(self) -> bool:
        row = self.database.query_one("SELECT COUNT(*) AS count FROM daily_bars")
        return bool(row and row["count"])

    def _with_fallback(self, method: str, *args: object) -> pd.DataFrame:
        try:
            return getattr(self.primary, method)(*args)
        except Exception as primary_error:
            if not self.fallback:
                raise RuntimeError(f"主数据源调用失败：{primary_error}") from primary_error
            LOG.warning("主数据源调用失败，改用 Tushare 备用数据源：%s", primary_error)
            return getattr(self.fallback, method)(*args)
