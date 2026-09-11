"""盘中即时交易（模拟盘买点口径）的回归测试。

背景：模拟盘原本是「15:30 生成信号 → 次日 09:26 按开盘价成交」，买卖都要跨夜。
现改为 **在盘中扫描时点（09:37/10:00/10:30/14:30/14:50）以实时价即时成交**，
盘后 15:30 只按当日收盘价做离场、不再排队次日买入。本文件锁定以下行为：

- 买入当天成交、不跨日，且成交时间记录的是真实时刻（不是写死的 09:30）；
- T+1 仍然生效（当日买入不可卖）；
- 止损/离场以实时价即时卖出，不等次日开盘；
- 单日新开名额、同一交易日不重复买同一只；
- 实时行情不可用（含 market_quotes 里的残留旧价）时跳过、不做任何成交；
- 盘中不追涨停。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ashare_quant.brokers.paper import PaperBroker
from ashare_quant.config import load_settings
from ashare_quant.data.seed import seed_demo_market
from ashare_quant.data.service import DataService
from ashare_quant.database import Database
from ashare_quant.lab import fresh_quotes
from ashare_quant.models import Signal, SignalAction
from ashare_quant.notifications import NotificationHub
from ashare_quant.risk import RiskManager
from ashare_quant.services.signals import SignalService
from ashare_quant.services.trading import TradingService, next_weekday
from ashare_quant.utils import today_text


def make_services(tmp_path: Path):
    settings = replace(load_settings(), db_path=tmp_path / "test.db")
    database = Database(settings.db_path)
    database.initialize(settings.paper_initial_cash)
    data = DataService(database, settings)
    seed_demo_market(database, data, days=260)
    notifications = NotificationHub(settings)
    risk = RiskManager(database, settings)
    broker = PaperBroker(database, settings, risk)
    signals = SignalService(database, data, settings, notifications)
    trading = TradingService(database, data, broker, settings, notifications)
    return settings, database, broker, signals, trading


def _latest_close(database: Database, code: str) -> float:
    row = database.query_one(
        "SELECT close FROM daily_bars WHERE code=? ORDER BY trade_date DESC LIMIT 1", (code,)
    )
    return float(row["close"])


def _quote(price: float, day: str, volume: float = 1.0e7) -> dict[str, float]:
    return {"price": price, "volume": volume, "quote_time": f"{day} 10:00:00"}


def _candidates(entries: list[tuple[str, str, float, float]]) -> list[dict]:
    return [{
        "strategy": "盘中对拍策略",
        "matches": [
            {"code": code, "name": name, "price": price, "amount": amount}
            for code, name, price, amount in entries
        ],
    }]


def test_intraday_buy_fills_on_the_spot(tmp_path: Path):
    """买入在盘中时点当天成交：委托不跨日，成交时间记录真实时刻。"""
    _settings, database, broker, _signals, trading = make_services(tmp_path)
    day = today_text()
    price = _latest_close(database, "600000")
    quotes = {"600000": _quote(price, day)}
    candidates = _candidates([("600000", "浦发银行", price, 8.0e8)])

    outcome = trading.execute_intraday(quotes=quotes, candidates=candidates)

    assert outcome["buy_filled"] == 1
    assert outcome["skipped"] == 0
    position = next(item for item in broker.get_positions() if item.code == "600000")
    assert position.quantity > 0
    assert position.sellable_quantity == 0, "T+1：当日买入不可卖出"

    order = database.query_one("SELECT * FROM orders WHERE code='600000' AND side='BUY'")
    assert order["status"] == "FILLED"
    assert order["trade_date"] == day, "委托不得跨日（旧流程是次日开盘成交）"
    fill = database.query_one("SELECT * FROM fills WHERE code='600000'")
    assert fill["filled_at"].startswith(day)
    assert not fill["filled_at"].startswith(f"{day} 09:30"), "应记录真实成交时刻，而非写死的 09:30"
    # 成交价含滑点，应贴近实时价而非某个历史开盘价
    assert abs(float(fill["price"]) - price) / price < 0.01


def test_intraday_exits_with_realtime_price(tmp_path: Path):
    """止损/离场用实时价即时成交，不再等次日开盘。"""
    _settings, database, broker, _signals, trading = make_services(tmp_path)
    day = today_text()
    entry = _latest_close(database, "600000")
    trading.execute_intraday(
        quotes={"600000": _quote(entry, day)},
        candidates=_candidates([("600000", "浦发银行", entry, 8.0e8)]),
    )
    assert any(item.code == "600000" for item in broker.get_positions())

    broker.roll_to_new_day("2099-12-31")  # 模拟次日：T+1 解锁可卖
    bar = database.query_one(
        "SELECT close, pre_close FROM daily_bars WHERE code='600000' ORDER BY trade_date DESC LIMIT 1"
    )
    pre_close = float(bar["pre_close"] or bar["close"])
    floor_price = round(pre_close * 0.90, 2)  # 跌停价：跌破它会被「跌停拦截」拒单
    stop_price = min(round(pre_close * 0.95, 2), round(entry * 0.98, 2))
    assert stop_price > floor_price, "测试前提：止损价须在跌停价之上才可成交"
    exit_signal = Signal(
        code="600000", name="浦发银行", action=SignalAction.SELL, as_of_date=day,
        strategy="盘中对拍策略", reason="止损（-8%）",
    )

    outcome = trading.execute_intraday(
        quotes={"600000": _quote(stop_price, day)}, exit_signals=[exit_signal]
    )

    assert outcome["sell_filled"] == 1
    assert not [item for item in broker.get_positions() if item.code == "600000"]
    fill = database.query_one("SELECT * FROM fills WHERE code='600000' AND side='SELL'")
    assert float(fill["price"]) < entry, "止损应按实时价卖出，而不是等到次日开盘"


def test_intraday_skips_when_quotes_unavailable(tmp_path: Path):
    """实时行情不可用时不做任何成交（宁可空仓也不拿陈旧价成交）。"""
    _settings, database, _broker, _signals, trading = make_services(tmp_path)
    price = _latest_close(database, "600000")

    outcome = trading.execute_intraday(
        quotes={}, candidates=_candidates([("600000", "浦发银行", price, 8.0e8)])
    )

    assert outcome == {"buy_filled": 0, "buy_rejected": 0, "sell_filled": 0,
                       "sell_rejected": 0, "failed": 0, "skipped": 1}
    assert database.query_all("SELECT * FROM orders") == []


def test_intraday_respects_daily_new_position_limit(tmp_path: Path):
    """单日新开名额（risk.max_daily_new_positions）在盘中多次扫描之间共享。"""
    settings, database, _broker, _signals, trading = make_services(tmp_path)
    day = today_text()
    codes = ["600000", "600036", "600519", "600887", "601318"]
    quotes = {code: _quote(_latest_close(database, code), day) for code in codes}
    entries = [(code, code, _latest_close(database, code), 1.0e9 - index * 1.0e7)
               for index, code in enumerate(codes)]

    outcome = trading.execute_intraday(quotes=quotes, candidates=_candidates(entries))

    assert outcome["buy_filled"] == 2, "默认 max_daily_new_positions=2"
    bought = database.query_all("SELECT * FROM orders WHERE side='BUY' AND status='FILLED'")
    assert len(bought) == 2


def test_intraday_second_tick_does_not_rebuy_same_symbol(tmp_path: Path):
    """同一交易日多个盘中时点之间不重复买同一只。"""
    settings, database, _broker, _signals, trading = make_services(tmp_path)
    settings.raw["risk"]["max_daily_new_positions"] = 5  # 放开名额，单独验证去重
    day = today_text()
    codes = ["600000", "600036"]
    quotes = {code: _quote(_latest_close(database, code), day) for code in codes}
    candidates = _candidates([(code, code, _latest_close(database, code), 5.0e8) for code in codes])

    first = trading.execute_intraday(quotes=quotes, candidates=candidates)
    second = trading.execute_intraday(quotes=quotes, candidates=candidates)

    assert first["buy_filled"] == 2
    assert second["buy_filled"] == 0
    for code in codes:
        orders = database.query_all("SELECT * FROM orders WHERE code=? AND side='BUY'", (code,))
        assert len(orders) == 1, f"{code} 当日不应重复下单"


def test_intraday_rejects_limit_up_buy(tmp_path: Path):
    """盘中不追涨停：实时价已达涨停则拒单。"""
    _settings, database, _broker, _signals, trading = make_services(tmp_path)
    day = today_text()
    bar = database.query_one(
        "SELECT close, pre_close FROM daily_bars WHERE code='600000' ORDER BY trade_date DESC LIMIT 1"
    )
    pre_close = float(bar["pre_close"] or bar["close"])
    limit_price = round(pre_close * 1.10, 2)

    outcome = trading.execute_intraday(
        quotes={"600000": _quote(limit_price, day)},
        candidates=_candidates([("600000", "浦发银行", limit_price, 8.0e8)]),
    )

    assert outcome["buy_filled"] == 0
    assert outcome["buy_rejected"] == 1
    order = database.query_one("SELECT * FROM orders WHERE code='600000'")
    assert order["status"] == "REJECTED"
    assert "涨停拦截" in order["error"]


def test_fresh_quotes_drops_stale_and_invalid_rows():
    """market_quotes 不会自动过期，交易前必须按 quote_time 剔除当日未刷新的残留价。"""
    day = "2026-09-11"
    quotes = {
        "600000": {"price": 10.0, "volume": 1.0, "quote_time": f"{day} 10:00:00"},
        "600036": {"price": 20.0, "volume": 1.0, "quote_time": "2026-09-10 15:00:00"},
        "600519": {"price": 0.0, "volume": 1.0, "quote_time": f"{day} 10:00:00"},
        "601318": {"price": 50.0, "volume": 1.0, "quote_time": ""},
    }
    assert set(fresh_quotes(quotes, day)) == {"600000"}
    assert fresh_quotes(None, day) == {}


def test_scheduler_intraday_times_exclude_close_session():
    """买入时点必须是盘中时段：15:30 属盘后固定价格交易时段（只能按收盘价成交），不得作为买点。"""
    times = load_settings().raw["scheduler"]["intraday_scan_times"]
    assert times == ["09:37", "10:00", "10:30", "14:30", "14:50"]
    assert all("09:30" <= moment <= "15:00" for moment in times)
    assert "15:30" not in times


def test_next_weekday_used_by_legacy_queue_path_is_unchanged():
    """旧的手工排队路径保留可用（CLI queue-orders / 看板手动平仓仍走它）。"""
    assert next_weekday("2026-09-11") == "2026-09-14"


def test_intraday_summary_text():
    """推送摘要要如实反映「跳过 = 行情不可用」与逐笔成交结果。"""
    from ashare_quant.scheduler import _intraday_summary

    assert "实时行情不可用" in _intraday_summary(
        {"buy_filled": 0, "buy_rejected": 0, "sell_filled": 0, "sell_rejected": 0,
         "failed": 0, "skipped": 1}
    )
    filled = _intraday_summary(
        {"buy_filled": 2, "buy_rejected": 1, "sell_filled": 1, "sell_rejected": 0,
         "failed": 0, "skipped": 0}
    )
    assert "买入成交 2 笔" in filled and "卖出成交 1 笔" in filled
    assert "买入被拒 1" in filled
    empty = _intraday_summary(
        {"buy_filled": 0, "buy_rejected": 0, "sell_filled": 0, "sell_rejected": 0,
         "failed": 0, "skipped": 0}
    )
    assert "本次无成交" in empty


def test_close_prices_reads_today_close(tmp_path: Path):
    """盘后离场取的是**当日**收盘价，缺失的标的不得混入。"""
    from types import SimpleNamespace

    from ashare_quant.scheduler import _close_prices

    _settings, database, _broker, _signals, _trading = make_services(tmp_path)
    day = today_text()
    prices = _close_prices(SimpleNamespace(database=database), ["600000", "999999"])
    assert "999999" not in prices
    bar = database.query_one(
        "SELECT close FROM daily_bars WHERE code='600000' AND trade_date=?", (day,)
    )
    assert prices["600000"] == float(bar["close"])


def test_run_scheduler_registers_intraday_trade_jobs(monkeypatch):
    """调度器必须注册盘中即时交易任务，且不得把 15:30（盘后固定价格时段）登记为买点。"""
    from types import SimpleNamespace

    import ashare_quant.scheduler as scheduler_module

    registered: list[tuple[str, str]] = []

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def add_job(self, func, trigger, args=None, id=None):
            registered.append((str(id), func.__name__))

        def start(self):
            registered.append(("__started__", ""))

    settings = load_settings()
    monkeypatch.setattr(scheduler_module, "BlockingScheduler", FakeScheduler)
    monkeypatch.setattr(
        scheduler_module, "build_runtime",
        lambda config_path=None: SimpleNamespace(settings=settings),
    )

    scheduler_module.run_scheduler()

    job_ids = dict(registered)
    assert "__started__" in job_ids
    for moment in settings.raw["scheduler"]["intraday_scan_times"]:
        hour, minute = moment.split(":")
        assert job_ids[f"intraday_trade_{int(hour):02d}{int(minute):02d}"] == "intraday_trade"
    assert "intraday_trade_1530" not in job_ids, "15:30 是盘后固定价格时段，不得作为买入时点"
    # 盘中扫描的旧任务名不应再出现（已由 intraday_trade 取代）
    assert not [job for job in job_ids if job.startswith("buy_scan_")]
