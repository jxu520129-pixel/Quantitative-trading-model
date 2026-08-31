"""Resilient AkShare and optional Tushare market-data providers."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from ..utils import as_compact_date, normalize_date


LOG = logging.getLogger(__name__)
T = TypeVar("T")


def with_retry(attempts: int = 3, delay_seconds: float = 1.5) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(function: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return function(*args, **kwargs)
                except Exception as error:  # Provider errors are deliberately isolated.
                    last_error = error
                    if attempt < attempts:
                        LOG.warning("数据接口 %s 第 %s/%s 次调用失败：%s", function.__name__, attempt, attempts, error)
                        time.sleep(delay_seconds * attempt)
            assert last_error is not None
            raise last_error

        return wrapped

    return decorator


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def list_securities(self) -> pd.DataFrame:
        """Return code, name, security_type, board, is_st and list_date columns."""

    @abstractmethod
    def daily_bars(self, code: str, start_date: str, end_date: str, security_type: str = "STOCK") -> pd.DataFrame:
        """Return normalized OHLCV daily bars."""

    @abstractmethod
    def realtime_quotes(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Return code, price, pre_close and volume when the provider supports it."""


def _with_exchange_prefix(code: str) -> str:
    code = str(code).zfill(6)
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def tushare_pro_from_env() -> Any | None:
    """按 .env 构建 Tushare pro API（TUSHARE_API_URL 可指向第三方代理），未配置 token 返回 None。

    供不经过 DataService 的模块（主升浪财务、因子实验室指标）做数据回退使用。
    """
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return None
    return TushareProvider(token, os.getenv("TUSHARE_API_URL", ""))._pro()


def sina_spot_prices(codes: list[str], timeout: float = 5.0) -> dict[str, float]:
    """新浪实时报价（按需轻量拉取，只请求给定代码，适合看板高频轮询展示）。

    返回 {code: 最新价}；价格无效或停牌为 0 的代码不会出现在结果里，
    调用方应回退到数据库 latest_price。失败时直接抛异常由调用方兜底。
    """
    import requests

    symbols = ",".join(_with_exchange_prefix(code) for code in codes)
    response = requests.get(
        f"https://hq.sinajs.cn/list={symbols}",
        headers={"Referer": "https://finance.sina.com.cn"},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = "gbk"
    result: dict[str, float] = {}
    for code, line in zip(codes, response.text.splitlines()):
        parts = line.split('"')
        if len(parts) < 2:
            continue
        fields = parts[1].split(",")
        try:
            price = float(fields[3])
        except (IndexError, ValueError):
            continue
        if price > 0:
            result[str(code).zfill(6)] = price
    return result


class AkShareProvider(MarketDataProvider):
    name = "akshare"

    def __init__(self, attempts: int = 3):
        self.attempts = attempts

    @staticmethod
    def _ak() -> Any:
        try:
            import akshare as ak
        except ImportError as error:
            raise RuntimeError("未安装 AkShare，请执行：pip install -r requirements.txt") from error
        return ak

    @with_retry()
    def list_securities(self) -> pd.DataFrame:
        ak = self._ak()
        stocks = ak.stock_info_a_code_name()
        stocks = stocks.rename(columns={"code": "code", "name": "name", "代码": "code", "名称": "name"})
        stocks = stocks[["code", "name"]].copy()
        stocks["code"] = stocks["code"].astype(str).str.zfill(6)
        stocks["security_type"] = "STOCK"
        stocks["board"] = "MAIN"
        stocks["is_st"] = stocks["name"].astype(str).str.upper().str.contains("ST").astype(int)
        stocks["list_date"] = ""

        try:
            etfs = ak.fund_etf_spot_em().rename(columns={"代码": "code", "名称": "name"})[["code", "name"]].copy()
            etfs["code"] = etfs["code"].astype(str).str.zfill(6)
            etfs["security_type"] = "ETF"
            etfs["board"] = "ETF"
            etfs["is_st"] = 0
            etfs["list_date"] = ""
            return pd.concat([stocks, etfs], ignore_index=True)
        except Exception as error:
            LOG.warning("无法从 AkShare 获取 ETF 证券池：%s", error)
            return stocks

    @with_retry()
    def daily_bars(self, code: str, start_date: str, end_date: str, security_type: str = "STOCK") -> pd.DataFrame:
        ak = self._ak()
        symbol = _with_exchange_prefix(code)
        if security_type.upper() == "ETF":
            raw = ak.fund_etf_hist_sina(symbol=symbol)
        else:
            raw = ak.stock_zh_a_daily(symbol=symbol, start_date=as_compact_date(start_date), end_date=as_compact_date(end_date), adjust="qfq")
        result = normalize_bars(raw, self.name)
        if security_type.upper() == "ETF":
            start = normalize_date(start_date)
            end = normalize_date(end_date)
            result = result[(result["trade_date"] >= start) & (result["trade_date"] <= end)]
        return result

    @with_retry()
    def index_constituents(self, index_code: str = "000300") -> pd.DataFrame:
        raw = self._ak().index_stock_cons(symbol=index_code)
        result = raw.rename(columns={"品种代码": "code", "代码": "code", "品种名称": "name", "名称": "name"})
        if "code" not in result:
            raise ValueError("AkShare 指数成分股数据中缺少证券代码列")
        result["code"] = result["code"].astype(str).str.zfill(6)
        return result[[column for column in ["code", "name"] if column in result.columns]]

    @with_retry()
    def realtime_quotes(self, codes: list[str] | None = None) -> pd.DataFrame:
        raw = self._ak().stock_zh_a_spot()
        result = raw.rename(columns={"代码": "code", "最新价": "price", "昨收": "pre_close", "成交量": "volume"})
        result = result[[column for column in ["code", "price", "pre_close", "volume"] if column in result.columns]].copy()
        result["code"] = result["code"].astype(str).str[-6:].str.zfill(6)
        if codes:
            result = result[result["code"].isin(codes)]
        return result.reset_index(drop=True)


class TushareProvider(MarketDataProvider):
    """Fallback source. It remains inactive unless TUSHARE_TOKEN is configured."""

    name = "tushare"

    def __init__(self, token: str, api_url: str = ""):
        if not token:
            raise ValueError("启用 Tushare 备用数据源需要配置 TUSHARE_TOKEN")
        self.token = token
        self.api_url = api_url

    def _pro(self) -> Any:
        try:
            import tushare as ts
        except ImportError as error:
            raise RuntimeError("未安装 Tushare") from error
        pro = ts.pro_api(self.token)
        if self.api_url:
            # 第三方代理（如付费转发服务）通过覆盖请求地址接入；服务到期会在这里报错，
            # 由上层回退 AkShare 主源兜底，清空 TUSHARE_API_URL 即恢复官方接口。
            pro._DataApi__http_url = self.api_url
        return pro

    @with_retry()
    def list_securities(self) -> pd.DataFrame:
        raw = self._pro().stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,list_date")
        result = raw.rename(columns={"symbol": "code", "name": "name", "list_date": "list_date"})
        result["security_type"] = "STOCK"
        result["board"] = "MAIN"
        result["is_st"] = result["name"].astype(str).str.upper().str.contains("ST").astype(int)
        return result[["code", "name", "security_type", "board", "is_st", "list_date"]]

    @with_retry()
    def daily_bars(self, code: str, start_date: str, end_date: str, security_type: str = "STOCK") -> pd.DataFrame:
        pro = self._pro()
        exchange = ".SH" if code.startswith(("5", "6", "9")) else ".SZ"
        ts_code = f"{code}{exchange}"
        if security_type.upper() == "ETF":
            raw = pro.fund_daily(ts_code=ts_code, start_date=as_compact_date(start_date), end_date=as_compact_date(end_date))
        else:
            raw = pro.daily(ts_code=ts_code, start_date=as_compact_date(start_date), end_date=as_compact_date(end_date))
        raw = raw.rename(
            columns={"trade_date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "vol": "成交量", "amount": "成交额", "pre_close": "昨收"}
        )
        return normalize_bars(raw, self.name)

    def realtime_quotes(self, codes: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError("V1 的 Tushare 备用数据源暂不支持实时行情")


def normalize_bars(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize Chinese/English provider columns to the local storage contract."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume", "amount", "pre_close", "source"])
    aliases = {
        "日期": "trade_date", "trade_date": "trade_date", "date": "trade_date",
        "开盘": "open", "open": "open", "最高": "high", "high": "high",
        "最低": "low", "low": "low", "收盘": "close", "close": "close",
        "成交量": "volume", "vol": "volume", "volume": "volume",
        "成交额": "amount", "amount": "amount", "昨收": "pre_close", "pre_close": "pre_close",
    }
    result = frame.rename(columns=aliases).copy()
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"数据源返回的日线数据缺少必要字段：{missing}")
    for column in ["open", "high", "low", "close", "volume", "amount", "pre_close"]:
        if column not in result:
            result[column] = 0.0 if column not in {"pre_close"} else None
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["trade_date"] = result["trade_date"].map(normalize_date)
    result = result.dropna(subset=["trade_date", "open", "high", "low", "close"])
    result = result[(result["close"] > 0) & (result["high"] >= result["low"])]
    result = result.sort_values("trade_date")
    # 复权一致性：pre_close 统一取同序列上一交易日收盘，与 close 保持同一口径，
    # 避免「前复权 close 与原始昨收」混用导致涨跌停判断失真。
    result["pre_close"] = result["close"].shift(1)
    result["source"] = source
    return result[["trade_date", "open", "high", "low", "close", "volume", "amount", "pre_close", "source"]]
