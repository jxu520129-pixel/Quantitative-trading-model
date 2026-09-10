"""Dependency construction shared by CLI, scheduler, tests, and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


# 本文件位于 ashare_quant/services/ 下，需上溯两层才到项目根
# （此前误用 parents[1]，只到 ashare_quant 包目录，导致 HIST_DB 指向不存在的路径，
#   prefer_hist_db=True 从未生效、回测一直落在空的演示库上）
ROOT = Path(__file__).resolve().parents[2]
HIST_DB = ROOT / "data" / "ashare_quant_hist.db"


@dataclass(frozen=True)
class Runtime:
    """一次运行所需的全部依赖集合（依赖注入容器），由 ``build_runtime`` 统一装配。"""

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


def build_runtime(config_path: str | None = None, prefer_hist_db: bool = False) -> Runtime:
    """加载配置并装配全部服务依赖，返回可用的 Runtime（CLI/调度器/看板/测试共用）。

    ``prefer_hist_db=True`` 时优先用 ``data/ashare_quant_hist.db``（全市场真实历史库，
    适合回测），仅在该文件存在时生效；否则退回 ``settings.db_path``（默认演示库，
    适合实盘/调度）。hist 库不会被 ``initialize``（它已经有完整数据，且
    initialize 会插入演示初始 cash/strategy_enabled 污染回测）。
    """
    settings = load_settings(config_path)
    use_hist = prefer_hist_db and HIST_DB.exists()
    db_path = HIST_DB if use_hist else settings.db_path
    database = Database(db_path)
    if use_hist:
        # hist 库不做 initialize（避免插入演示初始资金/开关污染回测数据），
        # 但需补齐缺失的表（回测落库、选股池查询等依赖），仅建表、不写数据。
        database.ensure_schema()
    else:
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
