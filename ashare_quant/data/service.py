"""Local cache, cleaning, provider fallback and universe selection."""

from __future__ import annotations

import logging
import time
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
    """数据服务层：证券池维护、日线增量更新与缓存读写，主源失败时回退备用源。"""

    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings
        primary_name = str(settings.data.get("primary", "akshare")).lower()
        fallback_name = str(settings.data.get("fallback", "tushare")).lower()
        self.primary: MarketDataProvider = self._build_provider(primary_name, settings)
        self.fallback: MarketDataProvider | None = (
            self._build_provider(fallback_name, settings) if fallback_name != primary_name else None
        )

    @staticmethod
    def _build_provider(name: str, settings: Settings) -> MarketDataProvider:
        """按 ``data.primary``/``data.fallback`` 构建数据源；tushare 缺 token 时回退 akshare。"""
        if name == "tushare":
            if settings.tushare_token:
                return TushareProvider(settings.tushare_token, settings.tushare_api_url)
            LOG.warning("配置 data.primary=tushare 但缺少 TUSHARE_TOKEN，回退 akshare")
        return AkShareProvider()

    def refresh_universe(self) -> int:
        """拉取并落库证券列表（仅保留 V1 支持标的），并尽力刷新指数成分股。"""
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
            # 指数成分股仅 akshare 提供；tushare 主源时用 akshare 兜底拉成分股
            provider = self.primary if hasattr(self.primary, "index_constituents") else AkShareProvider()
            members = provider.index_constituents(str(self.settings.data["stock_universe_index"]))
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
        """增量更新日线：并发拉取（并发数由 data.update_concurrency 控制），失败标的隔离不影响整体。

        单轮结束后，若失败标的占比超过 ``data.update_retry_failure_ratio``（默认 30%），
        自动降并发到 1 对失败标的补拉一轮，缓解数据源限流导致的「当日数据大面积缺失」。
        """
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

        def _update_one(code: str) -> dict[str, object]:
            security_type = security_types.get(code, "STOCK")
            previous = last_dates.get(code)
            if previous:
                start = (pd.Timestamp(previous) + timedelta(days=1)).date().isoformat()
            else:
                start = str(self.settings.data["history_start"])
            try:
                frame = self._with_fallback("daily_bars", code, start, end_date or today_text(), security_type)
                count = self.store_bars(code, frame)
                return {"updated": 1, "bars": count, "failed": 0, "code": code}
            except Exception as error:
                LOG.warning("证券 %s 日线更新失败：%s", code, error)
                return {"updated": 0, "bars": 0, "failed": 1, "code": code}

        def _run(codes: list[str], concurrency: int) -> tuple[dict[str, int], list[str]]:
            result = {"updated_symbols": 0, "bars": 0, "failed": 0}
            failed_codes: list[str] = []
            if concurrency <= 1 or len(codes) <= 1:
                for code in codes:
                    item = _update_one(code)
                    result["updated_symbols"] += int(item["updated"])
                    result["bars"] += int(item["bars"])
                    result["failed"] += int(item["failed"])
                    if item["failed"]:
                        failed_codes.append(str(item["code"]))
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(_update_one, code) for code in codes]
                    for future in as_completed(futures):
                        item = future.result()
                        result["updated_symbols"] += int(item["updated"])
                        result["bars"] += int(item["bars"])
                        result["failed"] += int(item["failed"])
                        if item["failed"]:
                            failed_codes.append(str(item["code"]))
            return result, failed_codes

        concurrency = int(self.settings.data.get("update_concurrency", 2) or 2)
        result, failed_codes = _run(normalized, concurrency)

        # 整批重试：失败占比超阈值时降并发补拉，规避限流导致的当日大面积缺失
        retry_ratio = float(self.settings.data.get("update_retry_failure_ratio", 0.30) or 0.30)
        if failed_codes and normalized and (len(failed_codes) / len(normalized)) > retry_ratio:
            LOG.warning("日线更新失败 %s/%s 超阈值，降并发=1 补拉失败标的", len(failed_codes), len(normalized))
            time.sleep(2)  # 稍候让限流窗口冷却
            retry_result, retry_failed = _run(failed_codes, 1)
            result["updated_symbols"] += retry_result["updated_symbols"]
            result["bars"] += retry_result["bars"]
            result["failed"] = retry_result["failed"]
        return result

    def daily_data_complete(self, threshold: float | None = None) -> bool:
        """最新交易日日线覆盖率是否达标（盘后流程闸门，避免用过期收盘价推送）。

        以「最新交易日条数 / 前一交易日条数」近似覆盖率；低于 ``data.min_daily_coverage``
        （默认 0.80）判定为当日数据拉取不完整。
        """
        threshold = threshold if threshold is not None else float(self.settings.data.get("min_daily_coverage", 0.80) or 0.80)
        rows = self.database.query_all(
            "SELECT trade_date, COUNT(*) AS cnt FROM daily_bars GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2"
        )
        if len(rows) < 2:
            return True  # 只有一天数据，无从对比，视作完整
        latest, previous = int(rows[0]["cnt"]), int(rows[1]["cnt"])
        if previous <= 0:
            return True
        return (latest / previous) >= threshold

    def store_bars(self, code: str, frame: pd.DataFrame) -> int:
        """清洗并落库单只标的日线（按交易日去重、UPSERT），返回写入条数。"""
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
        """加载单只标的日线（按日期升序，以 trade_date 为索引）。"""
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
        where_extra = (" AND " + " AND ".join(clauses)) if clauses else ""
        frames: dict[str, pd.DataFrame] = {}
        # 用单个连接循环分批查询：① 避免 12 次「建连接 + PRAGMA」开销，让 50MB 页缓存跨批生效；
        # ② pd.read_sql_query 走 pandas 的 C 层读取，跳过「sqlite3.Row → dict → DataFrame」的 Python 中间层。
        conn = self.database.connect()
        try:
            for i in range(0, len(codes), 500):
                chunk = codes[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                sql = f"SELECT code,{','.join(columns)} FROM daily_bars WHERE code IN ({placeholders}){where_extra}"
                # 不在此处 ORDER BY：主键 (code, trade_date) 覆盖索引已保证结果按 code+日期有序，
                # 显式 ORDER BY 会触发全量排序，767 万条数据下慢约 8 倍。
                frame = pd.read_sql_query(sql, conn, params=[*chunk, *params])
                if frame.empty:
                    continue
                frame["trade_date"] = pd.to_datetime(frame["trade_date"])
                # 索引已保证有序，一次性 set_index 后按 code 分组即可，
                # 不再逐股 sort_values + set_index（全市场 5000+ 股的 Python 循环开销可观）。
                frame = frame.set_index("trade_date", drop=False)
                for code, sub in frame.groupby("code", sort=False):
                    frames[code] = sub[columns]
        finally:
            conn.close()
        return frames

    def latest_bar(self, code: str) -> dict[str, object] | None:
        """返回单只标的最新一根日线（无数据返回 None）。"""
        return self.database.query_one("SELECT * FROM daily_bars WHERE code=? ORDER BY trade_date DESC LIMIT 1", (code,))

    def lookback_start(self, n_bars: int = 260) -> str | None:
        """返回最近 ``n_bars`` 个交易日的最早日期（YYYY-MM-DD）。

        供扫描/信号生成限制日线加载范围：因子最长窗口是 52 周新高（约 250 个交易日），
        加载全历史（2020 至今 1600+ 天）会把全市场扫描拖慢数分钟。
        """
        row = self.database.query_one(
            "SELECT trade_date FROM (SELECT DISTINCT trade_date FROM daily_bars ORDER BY trade_date DESC LIMIT ?)"
            " ORDER BY trade_date ASC LIMIT 1",
            (n_bars,),
        )
        return str(row["trade_date"]) if row else None

    def refresh_quotes(self, codes: list[str]) -> int:
        """拉取给定标的的实时快照并写入 market_quotes 表（供模拟成交与盘中扫描使用）。

        实时行情仅 akshare（新浪）支持，tushare 主源不支持（会抛 NotImplementedError），
        故优先用备用源（akshare）；无备用源时退回主源。
        """
        provider = self.fallback if self.fallback is not None else self.primary
        frame = provider.realtime_quotes(codes)
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
                float(item.get("volume", 0) or 0), provider.name,
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
        """判断日线表是否已有任何数据（用于决定是否先生成演示行情）。"""
        row = self.database.query_one("SELECT COUNT(*) AS count FROM daily_bars")
        return bool(row and row["count"])

    def _with_fallback(self, method: str, *args: object) -> pd.DataFrame:
        """调用主数据源，失败时自动回退到 Tushare 备用源（无备用源则抛错）。"""
        try:
            return getattr(self.primary, method)(*args)
        except Exception as primary_error:
            if not self.fallback:
                raise RuntimeError(f"主数据源调用失败：{primary_error}") from primary_error
            LOG.warning("主数据源调用失败，改用 Tushare 备用数据源：%s", primary_error)
            return getattr(self.fallback, method)(*args)
