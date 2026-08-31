"""Small utilities with no external market-data dependencies."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


def today_text() -> str:
    """返回本地日期字符串（格式 ``YYYY-MM-DD``），作为默认的交易日/数据日期。"""
    return datetime.now().date().isoformat()


def normalize_date(value: str | date | datetime) -> str:
    """把 ``str / date / datetime`` 统一规范为 ``YYYY-MM-DD`` 字符串。

    兼容紧凑形 ``YYYYMMDD``（8 位数字）与 ``YYYY-MM-DD`` 两种输入，
    其余情况取前 10 个字符兜底。
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def as_compact_date(value: str | date | datetime) -> str:
    """把日期转为紧凑形 ``YYYYMMDD``（供 AkShare/Tushare 接口参数使用）。"""
    return normalize_date(value).replace("-", "")


def is_weekday(value: str | date | datetime) -> bool:
    """判断给定日期是否为周一至周五（不含节假日判断，仅作交易日历兜底）。"""
    return datetime.fromisoformat(normalize_date(value)).weekday() < 5


def project_path(*parts: str) -> Path:
    """返回项目根目录下拼接路径（``parts`` 为相对根目录的分段）。"""
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def safe_float(value: object, default: float = 0.0) -> float:
    """安全地把任意对象转为 float；转换失败返回 ``default``，避免外部脏数据中断流程。"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
