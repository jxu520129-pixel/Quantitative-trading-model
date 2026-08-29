"""Deterministic synthetic data for offline demos and automated checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..database import Database
from .service import DataService


DEMO_SECURITIES = [
    ("600000", "浦发银行", "STOCK"), ("600036", "招商银行", "STOCK"),
    ("600519", "贵州茅台", "STOCK"), ("600887", "伊利股份", "STOCK"),
    ("601318", "中国平安", "STOCK"), ("601888", "中国中免", "STOCK"),
    ("000001", "平安银行", "STOCK"), ("000333", "美的集团", "STOCK"),
    ("000858", "五粮液", "STOCK"), ("002594", "比亚迪", "STOCK"),
    ("510300", "沪深300ETF", "ETF"), ("510500", "中证500ETF", "ETF"),
    ("512100", "中证1000ETF", "ETF"), ("512880", "证券ETF", "ETF"),
]


def seed_demo_market(database: Database, data_service: DataService, days: int = 520) -> int:
    """Create a realistic enough local market with varied momentum regimes."""
    now = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=now, periods=days)
    database.executemany(
        """INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,is_suspended,list_date,updated_at)
           VALUES(?,?,?,?,0,0,0,?,?) ON CONFLICT(code) DO UPDATE SET name=excluded.name,updated_at=excluded.updated_at""",
        [(code, name, kind, "ETF" if kind == "ETF" else "MAIN", "2010-01-01", now.strftime("%Y-%m-%d %H:%M:%S")) for code, name, kind in DEMO_SECURITIES],
    )
    universe_code = str(data_service.settings.data.get("stock_universe_index", "000300"))
    database.executemany(
        "INSERT OR IGNORE INTO universe_members(universe_code,code,updated_at) VALUES(?,?,?)",
        [(universe_code, code, now.strftime("%Y-%m-%d %H:%M:%S")) for code, _name, kind in DEMO_SECURITIES if kind == "STOCK"],
    )
    inserted = 0
    for ordinal, (code, _name, kind) in enumerate(DEMO_SECURITIES):
        rng = np.random.default_rng(20260815 + ordinal)
        base_price = 8 + ordinal * 4 if kind == "STOCK" else 3 + ordinal * 0.15
        drift = 0.00015 + (ordinal % 5) * 0.00006
        shocks = rng.normal(drift, 0.019 if kind == "STOCK" else 0.012, size=days)
        # Alternating trend windows make momentum rotation observable in the demo.
        shocks += np.where(np.arange(days) % 130 < 65, (ordinal % 3 - 1) * 0.00065, (1 - ordinal % 3) * 0.00065)
        closes = base_price * np.exp(np.cumsum(shocks))
        previous = np.concatenate(([base_price], closes[:-1]))
        opens = previous * (1 + rng.normal(0, 0.004, size=days))
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.018, size=days))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.018, size=days))
        volume = rng.integers(2_000_000, 20_000_000, size=days)
        frame = pd.DataFrame({
            "trade_date": dates.strftime("%Y-%m-%d"), "open": opens.round(2), "high": highs.round(2),
            "low": lows.round(2), "close": closes.round(2), "volume": volume,
            "amount": (volume * closes).round(2), "pre_close": previous.round(2), "source": "demo",
        })
        inserted += data_service.store_bars(code, frame)
    return inserted
