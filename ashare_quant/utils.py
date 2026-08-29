"""Small utilities with no external market-data dependencies."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


def today_text() -> str:
    return datetime.now().date().isoformat()


def normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def as_compact_date(value: str | date | datetime) -> str:
    return normalize_date(value).replace("-", "")


def is_weekday(value: str | date | datetime) -> bool:
    return datetime.fromisoformat(normalize_date(value)).weekday() < 5


def project_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
