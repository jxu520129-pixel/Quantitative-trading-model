"""Load cached market data, run a strategy, and persist standardized signals."""

from __future__ import annotations

from ..config import Settings
from ..data.service import DataService
from ..database import Database
from ..models import Signal, utc_now_text
from ..notifications import NotificationHub
from ..presentation import label_value
from ..strategies import build_strategy
from ..strategies.base import StrategyContext


class SignalService:
    def __init__(self, database: Database, data: DataService, settings: Settings, notifications: NotificationHub):
        self.database = database
        self.data = data
        self.settings = settings
        self.notifications = notifications

    def generate(self, as_of_date: str | None = None, strategy_name: str | None = None) -> list[Signal]:
        strategy_name = strategy_name or self.settings.active_strategy
        universe = self.data.eligible_universe(limit=int(self.settings.data["strategy_scan_symbols"]))
        if universe.empty:
            raise RuntimeError("可用证券池为空，请先更新数据或生成演示行情")
        bars_by_code = self.data.load_bars_many(universe["code"].tolist(), end_date=as_of_date)
        latest_dates = [frame["trade_date"].iloc[-1].date().isoformat() for frame in bars_by_code.values()]
        if not latest_dates:
            raise RuntimeError("没有可用于生成信号的日线数据")
        effective_date = as_of_date or max(latest_dates)
        # A symbol without the effective session's bar is treated as suspended/stale.
        bars_by_code = {
            code: frame for code, frame in bars_by_code.items()
            if frame["trade_date"].iloc[-1].date().isoformat() == effective_date
        }
        universe = universe[universe["code"].isin(bars_by_code)].reset_index(drop=True)
        if universe.empty:
            raise RuntimeError(f"{effective_date} 没有可用于生成信号的当期日线数据")
        held = {row["code"] for row in self.database.query_all("SELECT code FROM positions WHERE quantity>0")}
        strategy = build_strategy(strategy_name, dict(self.settings.strategies[strategy_name]))
        signals = strategy.generate(StrategyContext(
            as_of_date=effective_date, universe=universe, bars_by_code=bars_by_code,
            held_codes=held, max_positions=int(self.settings.risk["max_positions"]),
        ))
        self.database.executemany(
            """INSERT OR IGNORE INTO signals(id,code,name,action,as_of_date,strategy,target_weight,score,reason,status,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,'NEW',?)""",
            [(item.id, item.code, item.name, item.action.value, item.as_of_date, item.strategy,
              item.target_weight, item.score, item.reason, utc_now_text()) for item in signals],
        )
        if signals:
            summary = "\n".join(
                f"{label_value(item.action.value, 'action')} {item.code} {item.name}，评分={item.score:.4f}"
                for item in signals
            )
            self.notifications.send("策略交易信号", summary)
        return signals
