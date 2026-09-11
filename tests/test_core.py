from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.backtest import BacktestEngine
from ashare_quant.brokers.paper import PaperBroker
from ashare_quant.config import load_settings
from ashare_quant.data.seed import seed_demo_market
from ashare_quant.data.service import DataService
from ashare_quant.database import Database
from ashare_quant.market_rules import TradingCosts, at_limit, round_to_lot
from ashare_quant.notifications import NotificationHub
from ashare_quant.cli import build_parser
from ashare_quant.presentation import localize_dataframe, localize_payload
from ashare_quant.risk import RiskManager
from ashare_quant.services.signals import SignalService
from ashare_quant.services.trading import TradingService, next_weekday
from ashare_quant.strategies.factors import compute_factor
from ashare_quant.strategies.factory import STRATEGIES
from ashare_quant.lab import BUILTIN_FACTOR_FORMULAS, evaluate_conditions, evaluate_factor, run_signal_backtest


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
    return settings, database, data, broker, signals, trading


def test_market_rules():
    costs = TradingCosts()
    assert round_to_lot(1234) == 1200
    assert costs.commission(1000) == 5
    assert costs.stamp_duty(10000, True) == 10
    assert at_limit(11, 10, "600000", "BUY")


def test_board_price_limit_2026():
    from ashare_quant.market_rules import board_price_limit

    # 2026-07-06 起：沪深主板 ST/*ST 涨跌幅上调至 10%，与普通股一致
    assert board_price_limit("600000", is_st=True) == 0.10
    assert board_price_limit("600000", is_st=False) == 0.10
    # 科创板/创业板/北交所维持各自板块涨跌幅，ST 不降档
    assert board_price_limit("300001", is_st=True) == 0.20
    assert board_price_limit("688001", is_st=True) == 0.20
    assert board_price_limit("830001", is_st=True) == 0.30


def test_chinese_display_text():
    payload = localize_payload({
        "mode": "PAPER",
        "strategy": "momentum_rotation",
        "status": "FILLED",
        "reason": "No longer in selected portfolio",
    })
    assert payload == {
        "运行模式": "模拟盘",
        "策略": "周度动量轮动",
        "状态": "已成交",
        "信号依据": "不再属于目标组合",
    }
    table = localize_dataframe(pd.DataFrame([{
        "code": "600000", "side": "BUY", "status": "FILLED", "strategy": "momentum_rotation",
    }]))
    assert table.to_dict("records") == [{
        "证券代码": "600000", "方向": "买入", "状态": "已成交", "策略": "周度动量轮动",
    }]
    help_text = build_parser().format_help()
    assert "模拟盘优先的 A 股量化交易系统" in help_text
    assert "显示此帮助信息并退出" in help_text
    assert "show this help message" not in help_text


def test_paper_signal_to_fill(tmp_path: Path):
    _settings, database, _data, broker, signals, trading = make_services(tmp_path)
    generated = signals.generate(strategy_name="momentum_rotation")
    assert generated
    queued = trading.queue_new_signals(generated[0].as_of_date)
    assert 0 < queued["queued"] <= 2
    result = trading.execute_morning(next_weekday(generated[0].as_of_date), refresh_quotes=False)
    assert result["filled"] > 0
    assert broker.get_account().total_equity > 0
    positions = broker.get_positions()
    assert positions and all(position.sellable_quantity == 0 for position in positions)
    trading.execute_morning(next_weekday(generated[0].as_of_date), refresh_quotes=False)
    assert all(position.sellable_quantity == 0 for position in broker.get_positions())
    assert database.query_one("SELECT COUNT(*) AS count FROM fills")["count"] == result["filled"]


def test_backtrader_run(tmp_path: Path):
    settings, database, data, _broker, _signals, _trading = make_services(tmp_path)
    dates = database.query_one("SELECT MIN(trade_date) AS start,MAX(trade_date) AS end FROM daily_bars")
    result = BacktestEngine(database, data, settings).run("momentum_rotation", dates["start"], dates["end"])
    assert result["final_equity"] > 0
    assert result["equity_curve"]
    assert database.query_one("SELECT COUNT(*) AS count FROM backtest_runs")["count"] == 1


def test_factor_library():
    dates = pd.bdate_range("2024-01-01", periods=160)
    closes = 10 * np.exp(np.linspace(0.0, 0.5, len(dates)))
    frame = pd.DataFrame({
        "open": closes * 1.001, "high": closes * 1.02, "low": closes * 0.98,
        "close": closes, "volume": 1_000_000.0, "amount": 20_000_000.0, "pre_close": np.nan,
    })
    assert compute_factor("momentum", frame) > 0
    assert compute_factor("reversal", frame) == -compute_factor("momentum", frame, {"momentum_window": 5})
    rsi = compute_factor("rsi", frame)
    assert 0 <= rsi <= 100
    # 单调上行序列的均线多头排列度应为 1
    assert compute_factor("ma_trend", frame) == 1.0


NEW_STRATEGIES = ["reversal", "low_volatility", "trend", "liquidity", "rsi_mean_reversion", "enhanced_multifactor"]


def test_new_strategies_generate_signals(tmp_path: Path):
    _settings, _database, _data, _broker, signals, _trading = make_services(tmp_path)
    for name in NEW_STRATEGIES:
        assert signals.generate(strategy_name=name), f"{name} 未生成信号"


def test_backtest_new_strategies(tmp_path: Path):
    settings, database, data, _broker, _signals, _trading = make_services(tmp_path)
    dates = database.query_one("SELECT MIN(trade_date) AS start,MAX(trade_date) AS end FROM daily_bars")
    for name in ("reversal", "enhanced_multifactor"):
        result = BacktestEngine(database, data, settings).run(name, dates["start"], dates["end"])
        assert result["final_equity"] > 0
    assert set(NEW_STRATEGIES) <= set(STRATEGIES)


def test_lab_factor_eval():
    frame = pd.DataFrame({
        "open": np.arange(1.0, 11.0), "high": np.arange(1.0, 11.0) + 0.1,
        "low": np.arange(1.0, 11.0) - 0.1, "close": np.arange(1.0, 11.0),
        "volume": np.full(10, 1000.0), "amount": np.full(10, 2000.0), "pre_close": np.arange(1.0, 11.0),
    })
    series = evaluate_factor("close/close.shift(5)-1", frame)
    assert series.iloc[5] == pytest.approx(6.0 / 1.0 - 1.0)
    assert series.iloc[6] == pytest.approx(7.0 / 2.0 - 1.0)
    with pytest.raises(ValueError):
        evaluate_factor("__import__('os')", frame)


def test_lab_conditions():
    values = {
        "a": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": pd.Series([5.0, 4.0, 3.0, 2.0, 1.0]),
    }
    and_mask = evaluate_conditions(values, [
        {"factor": "a", "op": ">", "value": 2.0},
        {"factor": "b", "op": "<", "value": 4.0},
    ], "AND")
    assert and_mask.tolist() == [False, False, True, True, True]
    or_mask = evaluate_conditions(values, [
        {"factor": "a", "op": "<", "value": 2.0},
        {"factor": "b", "op": "<", "value": 2.0},
    ], "OR")
    assert or_mask.tolist() == [True, False, False, False, True]


def test_lab_signal_backtest(tmp_path: Path):
    settings, database, data, _broker, _signals, _trading = make_services(tmp_path)
    dates = database.query_one("SELECT MIN(trade_date) AS a, MAX(trade_date) AS b FROM daily_bars")
    exprs = {"mom": "close/close.shift(5)-1"}
    entry = {"combine": "AND", "conditions": [{"factor": "mom", "op": ">", "value": 0.0}]}
    exit_ = {"combine": "OR", "conditions": [{"factor": "mom", "op": "<", "value": 0.0}]}
    trades, metrics, equity = run_signal_backtest(data, settings, exprs, entry, exit_, dates["a"], dates["b"])
    assert metrics["total_trades"] > 0
    assert len(trades) == metrics["total_trades"]
    assert len(equity) > 0
    assert BUILTIN_FACTOR_FORMULAS


def test_lab_stop_loss(tmp_path: Path):
    settings, database, data, _broker, _signals, _trading = make_services(tmp_path)
    dates = database.query_one("SELECT MIN(trade_date) AS a, MAX(trade_date) AS b FROM daily_bars")
    exprs = {"mom": "close/close.shift(5)-1"}
    # 因子离场条件永不为真，唯一离场方式是 5% 止损
    entry = {"combine": "AND", "conditions": [{"factor": "mom", "op": ">", "value": -999.0}]}
    exit_ = {"combine": "OR", "conditions": [{"factor": "mom", "op": "<", "value": -999.0}], "stop_loss_pct": 0.05}
    trades, metrics, _equity = run_signal_backtest(data, settings, exprs, entry, exit_, dates["a"], dates["b"])
    assert metrics["total_trades"] > 0
    assert any(t["exit_reason"].startswith("止损") for t in trades)


def test_trading_calendar():
    from ashare_quant.trading_calendar import TradingCalendar

    cal = TradingCalendar()
    cal._days = set()  # 模拟日历获取失败，退回工作日判断
    assert cal.is_trading_day("2024-01-01") is True   # 周一
    assert cal.is_trading_day("2024-01-06") is False  # 周六
    cal._days = {"2024-01-02", "2024-01-03"}
    assert cal.is_trading_day("2024-01-02") is True
    assert cal.is_trading_day("2024-01-04") is False


def test_normalize_bars_derives_pre_close():
    from ashare_quant.data.providers import normalize_bars

    frame = pd.DataFrame({
        "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "open": [10.0, 11.0, 12.0], "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0], "close": [10.5, 11.5, 12.5],
        "volume": [100.0, 100.0, 100.0], "amount": [1000.0, 1000.0, 1000.0],
        "pre_close": [9.9, 9.9, 9.9],  # 故意给错，应被 close.shift(1) 覆盖
    })
    out = normalize_bars(frame, "test")
    assert pd.isna(out["pre_close"].iloc[0])
    assert out["pre_close"].iloc[1] == pytest.approx(10.5)
    assert out["pre_close"].iloc[2] == pytest.approx(11.5)


def test_builtin_factors_all_evaluate():
    from ashare_quant.lab import BUILTIN_FACTOR_FORMULAS, evaluate_factor

    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    closes = 10 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(dates))))
    frame = pd.DataFrame({
        "open": closes * (1 + rng.normal(0, 0.003, len(dates))),
        "high": closes * 1.01, "low": closes * 0.99, "close": closes,
        "volume": rng.integers(1e6, 2e7, len(dates)).astype(float),
        "amount": rng.integers(2e7, 5e8, len(dates)).astype(float),
        "pre_close": np.concatenate(([10.0], closes[:-1])),
    })
    assert len(BUILTIN_FACTOR_FORMULAS) >= 24
    for name, (_desc, expr) in BUILTIN_FACTOR_FORMULAS.items():
        series = evaluate_factor(expr, frame)
        assert isinstance(series, pd.Series) and len(series) == len(frame), f"{name} 求值失败"


def test_filter_candidates_by_quality():
    from ashare_quant.lab import filter_candidates_by_quality

    class _Settings:
        raw = {"scan": {"min_sector_change": 0.0, "require_hot_stock": False,
                        "hot_rank_limit": 200, "exclude_negative_notice": True}}

    candidates = [{"strategy": "t", "matches": [
        {"code": "600000", "price": 10.0},  # 货币金融→银行 +1.5%，强势 → 保留
        {"code": "600036", "price": 11.0},  # 煤炭→煤炭 -0.5%，弱势 → 剔除
        {"code": "600519", "price": 12.0},  # 饮料→白酒，且有减持公告 → 剔除
    ]}]
    industries = {"600000": "货币金融", "600036": "煤炭", "600519": "饮料"}
    sector_changes = {"银行": 1.5, "煤炭": -0.5, "白酒": -0.5}
    notices = {"600519": ["股东减持计划公告"]}

    out = filter_candidates_by_quality(candidates, sector_changes, industries, notices, _Settings())
    kept = [m["code"] for item in out for m in item["matches"]]
    assert kept == ["600000"]


def test_builtin_strategy_templates():
    from ashare_quant.lab import BUILTIN_FACTOR_FORMULAS, BUILTIN_STRATEGY_TEMPLATES

    assert BUILTIN_STRATEGY_TEMPLATES
    for name, (_desc, entry, exit_) in BUILTIN_STRATEGY_TEMPLATES.items():
        factors = [c["factor"] for c in entry["conditions"]] + [c["factor"] for c in exit_.get("conditions", [])]
        for factor in factors:
            assert factor in BUILTIN_FACTOR_FORMULAS, f"{name} 引用了未定义因子 {factor}"


def test_dashboard_refresh_interval_persists(tmp_path: Path):
    """看板刷新间隔存 system_settings，跨「重新登录」（新进程 = 新 SystemControl）仍沿用。"""
    from ashare_quant.services.control import SystemControl

    _settings, database, _data, _broker, _signals, _trading = make_services(tmp_path)
    assert SystemControl(database).get("holdings_refresh_seconds", "") == ""

    SystemControl(database).set("holdings_refresh_seconds", "30")
    # 新实例模拟重新打开看板/重启进程：值仍在
    assert SystemControl(database).get("holdings_refresh_seconds") == "30"

    SystemControl(database).set("holdings_refresh_seconds", "60")
    assert SystemControl(database).get("holdings_refresh_seconds", "30") == "60"


def test_dashboard_refresh_migration_upgrades_old_default(tmp_path: Path):
    """老库里的旧默认 10 秒应被一次性升级为 30，且之后不再覆盖用户的选择。"""
    from ashare_quant.services.control import SystemControl

    _settings, database, _data, _broker, _signals, _trading = make_services(tmp_path)

    # 场景一：老库存着旧默认值 10 → 迁移后变 30
    SystemControl(database).set("holdings_refresh_seconds", "10")
    SystemControl(database).migrate_dashboard_refresh(30)
    assert SystemControl(database).get("holdings_refresh_seconds") == "30"

    # 场景二：迁移后再手动选 10 秒（用户主动选择）→ 重复迁移不得改回去
    SystemControl(database).set("holdings_refresh_seconds", "10")
    SystemControl(database).migrate_dashboard_refresh(30)
    assert SystemControl(database).get("holdings_refresh_seconds") == "10"

    # 场景三：全新库（键不存在）→ 落到新默认值
    fresh = tmp_path / "fresh.db"
    fresh_db = Database(fresh)
    fresh_db.initialize(1_000_000)
    SystemControl(fresh_db).migrate_dashboard_refresh(30)
    assert SystemControl(fresh_db).get("holdings_refresh_seconds") == "30"


def test_dashboard_refresh_default_is_30_seconds():
    """护栏：浮动盈亏刷新默认值必须是 30 秒，且 30 出现在下拉可选项里（否则 index 会取值报错）。

    dashboard.py 依赖 streamlit，无法直接 import，故按源码做契约校验——防止默认值被改回 10 秒。
    """
    import re

    source = (Path(__file__).resolve().parents[1] / "dashboard.py").read_text(encoding="utf-8")
    default = re.search(r"^\s*DEFAULT_REFRESH_SECONDS\s*=\s*(\d+)", source, re.M)
    assert default, "dashboard.py 应定义 DEFAULT_REFRESH_SECONDS"
    assert int(default.group(1)) == 30

    options = re.search(r"^\s*refresh_options\s*=\s*\[(.*?)\]", source, re.M)
    assert options, "dashboard.py 应定义 refresh_options"
    raw = options.group(1)
    assert "DEFAULT_REFRESH_SECONDS" in raw or 30 in [int(v) for v in re.findall(r"\d+", raw)]
    assert 0 in [int(v) for v in re.findall(r"\d+", raw)], "0（暂停刷新）必须保留在可选项里"
