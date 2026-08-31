"""Configuration loading. Secrets are deliberately read from environment only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class Settings:
    root: Path
    raw: dict[str, Any]
    db_path: Path
    mode: str
    tushare_token: str
    wecom_webhook: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    email_to: str
    dashboard_password: str
    tushare_api_url: str = ""

    @property
    def trading(self) -> dict[str, Any]:
        return self.raw["trading"]

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw["risk"]

    @property
    def strategies(self) -> dict[str, Any]:
        return self.raw["strategies"]

    @property
    def active_strategy(self) -> str:
        return str(self.raw["strategy"]["active"])

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def paper_initial_cash(self) -> float:
        return float(self.trading["paper_initial_cash"])


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load checked-in defaults, then optional local config and environment secrets."""
    load_dotenv(ROOT / ".env", override=False)
    default_path = ROOT / "config" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    requested = config_path or os.getenv("QUANT_CONFIG")
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        with path.open("r", encoding="utf-8") as handle:
            raw = _deep_merge(raw, yaml.safe_load(handle) or {})

    mode = os.getenv("TRADING_MODE", str(raw["trading"]["mode"])).upper()
    if mode not in {"PAPER", "LIVE"}:
        raise ValueError("TRADING_MODE 只能设置为 PAPER（模拟盘）或 LIVE（实盘）")
    db_path = Path(os.getenv("DATABASE_PATH", str(raw["storage"]["sqlite_path"])))
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    return Settings(
        root=ROOT,
        raw=raw,
        db_path=db_path,
        mode=mode,
        tushare_token=os.getenv("TUSHARE_TOKEN", ""),
        tushare_api_url=os.getenv("TUSHARE_API_URL", ""),
        wecom_webhook=os.getenv("WECOM_WEBHOOK", ""),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        smtp_username=os.getenv("SMTP_USERNAME", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        email_to=os.getenv("EMAIL_TO", ""),
        dashboard_password=os.getenv("DASHBOARD_PASSWORD", ""),
    )
