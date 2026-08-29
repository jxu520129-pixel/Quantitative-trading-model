"""Dependency construction shared by CLI, scheduler, tests, and dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from ..backtest import BacktestEngine
from ..brokers.paper import PaperBroker
from ..config import Settings, load_settings
from ..data.service import DataService
from ..database import Database
from ..notifications import NotificationHub
from ..risk import RiskManager
from ..trading_calendar import TradingCalendar
from .control import SystemControl
from .signals import SignalService
from .trading import TradingService


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    database: Database
    data: DataService
    risk: RiskManager
    broker: PaperBroker
    notifications: NotificationHub
    signals: SignalService
    trading: TradingService
    backtest: BacktestEngine
    control: SystemControl
    calendar: TradingCalendar


def build_runtime(config_path: str | None = None) -> Runtime:
    settings = load_settings(config_path)
    database = Database(settings.db_path)
    database.initialize(settings.paper_initial_cash)
    notifications = NotificationHub(settings)
    data = DataService(database, settings)
    risk = RiskManager(database, settings)
    broker = PaperBroker(database, settings, risk)
    return Runtime(
        settings=settings, database=database, data=data, risk=risk, broker=broker,
        notifications=notifications,
        signals=SignalService(database, data, settings, notifications),
        trading=TradingService(database, data, broker, settings, notifications),
        backtest=BacktestEngine(database, data, settings), control=SystemControl(database),
        calendar=TradingCalendar(),
    )
