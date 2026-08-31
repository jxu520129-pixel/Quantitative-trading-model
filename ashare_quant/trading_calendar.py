"""A-share trading-day calendar with a weekday fallback."""

from __future__ import annotations

import logging
import threading

from .utils import is_weekday, normalize_date


LOG = logging.getLogger(__name__)


class TradingCalendar:
    """交易所交易日历（新浪），惰性加载并缓存。

    ``is_trading_day`` 仅在真实交易日返回 True：法定节假日与调休休市日被排除，
    调休补班的周末交易日会被纳入。当日历源不可用时退化为「周一至周五」判断。
    """

    def __init__(self) -> None:
        self._days: set[str] | None = None
        self._lock = threading.Lock()

    def refresh(self) -> set[str]:
        """拉取并缓存完整交易日历（不持锁，调用方负责串行化或接受幂等覆盖）。"""
        import akshare as ak

        frame = ak.tool_trade_date_hist_sina()
        days = {normalize_date(value) for value in frame["trade_date"].tolist()}
        self._days = days
        return days

    def is_trading_day(self, value: str | object) -> bool:
        """判断给定日期是否为交易日（惰性加载并缓存日历，接口失败退回工作日判断）。"""
        if self._days is None:
            with self._lock:
                if self._days is None:
                    try:
                        self.refresh()
                    except Exception as error:
                        LOG.warning("交易日历获取失败，退回工作日判断：%s", error)
                        self._days = set()
        text = normalize_date(value)
        if not self._days:
            return is_weekday(text)
        return text in self._days
