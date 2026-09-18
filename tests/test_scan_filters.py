"""盘中买点扫描的「板块 / 价格」过滤条件（看板可配置）的回归测试。

背景：过滤条件原本只写在 config/default.yaml，改一次要动代码。现暴露到看板
「因子实验室 → 盘中买点扫描」，写入 system_settings，与调度器进程共享，
保证「页面改多少、盘中交易就按多少执行」。本文件锁定：
- board_of_code 的代码前缀 → 板块映射（board 字段恒为 MAIN，不可靠）；
- effective_scan_filters 的 system_settings 覆盖 + config 兜底；
- 板块过滤确实作用到候选扫描（允许名单外的一只都不出）。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ashare_quant.brokers.paper import PaperBroker
from ashare_quant.config import load_settings
from ashare_quant.data.seed import seed_demo_market
from ashare_quant.data.service import DataService
from ashare_quant.database import Database
from ashare_quant.lab import effective_scan_filters, find_buy_candidates
from ashare_quant.market_rules import board_of_code
from ashare_quant.models import utc_now_text
from ashare_quant.notifications import NotificationHub
from ashare_quant.risk import RiskManager
from ashare_quant.services.control import SystemControl
from ashare_quant.services.signals import SignalService
from ashare_quant.services.trading import TradingService


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
    return settings, database, data


@pytest.mark.parametrize(
    "code, expected",
    [
        ("600000", "主板"), ("000001", "主板"), ("002594", "主板"), ("003816", "主板"),
        ("688981", "科创板"), ("300750", "创业板"), ("301060", "创业板"),
        ("830799", "北交所"), ("430047", "北交所"), ("920002", "北交所"),
        ("510300", "基金"), ("159915", "基金"),
    ],
)
def test_board_of_code(code: str, expected: str) -> None:
    assert board_of_code(code) == expected


def test_effective_scan_filters_falls_back_to_config(tmp_path: Path):
    """未在页面改过时，读 config/default.yaml 的 scan 默认值。"""
    _settings, database, _data = make_services(tmp_path)
    filters = effective_scan_filters(database, _settings)
    assert filters["max_price"] == 50.0
    assert filters["min_price"] == 5.0
    assert filters["allowed_boards"] is None, "未配置板块名单时不过滤板块"


def test_effective_scan_filters_override_and_persist(tmp_path: Path):
    """页面改过的价格区间与板块名单覆盖 config，并跨「进程」沿用。"""
    settings, database, _data = make_services(tmp_path)
    control = SystemControl(database)
    control.set("scan_max_price", "30")
    control.set("scan_min_price", "8")
    control.set("scan_boards", "主板,创业板")

    filters = effective_scan_filters(database, settings)  # 模拟调度器进程另起读取
    assert filters["max_price"] == 30.0
    assert filters["min_price"] == 8.0
    assert filters["allowed_boards"] == {"主板", "创业板"}

    # 只改价格、不碰板块 → 板块名单保持上次设置
    control.set("scan_max_price", "20")
    assert effective_scan_filters(database, settings)["max_price"] == 20.0
    assert effective_scan_filters(database, settings)["allowed_boards"] == {"主板", "创业板"}


def test_scan_board_filter_excludes_disallowed_board(tmp_path: Path):
    """板块过滤要真正作用到候选扫描：允许名单外的一只都不出。"""
    settings, database, data = make_services(tmp_path)
    database.execute(
        "INSERT INTO signal_strategies(id,name,entry_json,exit_json,created_at,enabled) "
        "VALUES(?,?,?,?,?,?)",
        ("s", "全市场", json.dumps({"conditions": [{"factor": "rally_5", "op": ">", "value": -1}], "combine": "AND"}),
         json.dumps({"conditions": [], "combine": "OR"}), utc_now_text(), 1),
    )
    # 演示股全是 600xxx（主板）→ 只允许科创板时，一个候选都不该有
    star_only = find_buy_candidates(data, database, allowed_boards={"科创板"})
    assert all(not item["matches"] for item in star_only), "科创板名单外的主板股不应出现"
    # 只允许主板 → 有候选
    main_only = find_buy_candidates(data, database, allowed_boards={"主板"})
    assert any(item["matches"] for item in main_only)
