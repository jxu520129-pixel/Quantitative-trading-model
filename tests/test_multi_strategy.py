"""多策略并行、独立止损止盈、持仓策略归属的测试。

覆盖：
- 离场判断（止损/止盈/移动止损/跌破箱体/离场条件）与回测共用同一口径
- 单个自定义策略只管理**自己买入**的持仓，不越界平掉别的策略的仓
- 同一标的只买一次；买入信号跨策略竞争剩余名额（多策略共享总名额）
- 建仓时写入策略归属 / 移动止损峰值 / 箱体基准；旧库迁移不丢数据
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from ashare_quant.brokers.paper import PaperBroker
from ashare_quant.database import Database
from ashare_quant.lab import evaluate_exit
from ashare_quant.models import OrderRequest, OrderSide, Signal, SignalAction
from ashare_quant.risk import RiskManager
from ashare_quant.services.signals import SignalService
from ashare_quant.strategies.base import StrategyContext
from ashare_quant.strategies.factory import build_strategy, is_lab_strategy, lab_strategy_key

NOW = "2026-09-10 09:25:00"


class FakeSettings:
    """仅提供策略/风控所需字段的轻量配置替身。"""

    def __init__(self, max_positions: int = 3) -> None:
        self.trading = {"commission_rate": 0.00025, "min_commission": 5.0,
                        "stamp_duty_rate": 0.001, "slippage_rate": 0.0005}
        self.risk = {
            "max_positions": max_positions,
            "max_single_position_weight": 0.20,
            "daily_loss_stop": 0.03,
            "max_daily_new_positions": 2,
            "max_consecutive_failures": 3,
        }
        self.data = {"strategy_scan_symbols": 100}
        self.strategies: dict = {}
        self.active_strategy = ""


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "t.db")
    db.initialize(100_000)
    return db


def _register_factor(db: Database, name: str, expression: str) -> None:
    db.execute(
        "INSERT INTO custom_factors(id,name,expression,created_at) VALUES(?,?,?,?)",
        (name, name, expression, NOW),
    )


def _register_strategy(db: Database, name: str, entry: dict, exit_: dict, enabled: int = 1) -> None:
    db.execute(
        "INSERT INTO signal_strategies(id,name,entry_json,exit_json,created_at,enabled) VALUES(?,?,?,?,?,?)",
        (name, name, json.dumps(entry), json.dumps(exit_), NOW, enabled),
    )


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2026-08-03", periods=len(closes))
    return pd.DataFrame({
        "trade_date": index,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * len(closes),
        "amount": [10_000_000.0] * len(closes),
        "pre_close": [closes[0]] + closes[:-1],
    }, index=index)


def _context(
    frame_map: dict[str, pd.DataFrame],
    held: list[str],
    positions: dict[str, dict],
    max_positions: int = 3,
) -> StrategyContext:
    codes = list(frame_map)
    return StrategyContext(
        as_of_date="2026-09-10",
        universe=pd.DataFrame({"code": codes, "name": [f"股{code}" for code in codes]}),
        bars_by_code=frame_map,
        held_codes=set(held),
        max_positions=max_positions,
        positions=positions,
    )


def _signal(code: str, action: SignalAction, score: float, strategy: str = "策略A") -> Signal:
    return Signal(code=code, action=action, as_of_date="2026-09-10", strategy=strategy,
                  score=score, name=code)


# ----------------------------------------------------------------- 离场口径
def test_evaluate_exit_covers_all_rules() -> None:
    assert "止损" in evaluate_exit({"stop_loss_pct": 0.05}, 10.0, 9.4, 10.0)
    assert "止盈" in evaluate_exit({"take_profit_pct": 0.10}, 10.0, 11.2, 10.0)
    # 峰值 12 回撤 10% → 10.8 触发移动止损
    assert "移动止损" in evaluate_exit({"trailing_stop_pct": 0.10}, 10.0, 10.7, 12.0)
    assert "箱体" in evaluate_exit({"breakout_exit": True}, 10.0, 8.0, 10.0, entry_breakout=9.0)
    assert "离场条件" in evaluate_exit({}, 10.0, 10.0, 10.0, factor_exit=True)
    assert evaluate_exit({"stop_loss_pct": 0.05}, 10.0, 10.2, 10.2) is None
    # 优先级：止损先于止盈
    assert "止损" in evaluate_exit({"stop_loss_pct": 0.05, "take_profit_pct": 0.10}, 10.0, 9.0, 10.0)


# ------------------------------------------------------- 策略只卖自己的持仓
def test_custom_strategy_only_exits_own_positions(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _register_factor(db, "f_close", "close")
    _register_strategy(db, "策略A",
                       {"conditions": [{"factor": "f_close", "op": ">", "value": 0.0}]},
                       {"stop_loss_pct": 0.05})
    strategy = build_strategy(lab_strategy_key("策略A"), {"_database": db})
    frame = _frame([10.0] * 25 + [9.3])  # 最新收盘 9.3，相对成本 10 跌 7% → 触发 5% 止损

    own = {"600000": {"code": "600000", "name": "股600000", "quantity": 1000, "avg_cost": 10.0,
                      "trail_peak": 10.0, "entry_breakout": 10.0, "strategy": "策略A"}}
    sells = [s for s in strategy.generate(_context({"600000": frame}, ["600000"], own))
             if s.action == SignalAction.SELL]
    assert len(sells) == 1
    assert "止损" in sells[0].reason
    assert sells[0].strategy == "策略A"

    # 持仓归属「策略B」→ 策略A 不得越界平仓
    other = {"600000": {**own["600000"], "strategy": "策略B"}}
    assert [s for s in strategy.generate(_context({"600000": frame}, ["600000"], other))
            if s.action == SignalAction.SELL] == []


def test_custom_strategy_take_profit_and_trailing(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _register_factor(db, "f_close", "close")
    _register_strategy(db, "策略T",
                       {"conditions": [{"factor": "f_close", "op": ">", "value": 0.0}]},
                       {"take_profit_pct": 0.10, "trailing_stop_pct": 0.08})
    strategy = build_strategy(lab_strategy_key("策略T"), {"_database": db})

    hold = {"600000": {"code": "600000", "name": "股600000", "quantity": 1000, "avg_cost": 10.0,
                       "trail_peak": 10.0, "entry_breakout": 10.0, "strategy": "策略T"}}
    # 收盘 11.5 ≥ 成本×1.10 → 止盈
    taken = [s for s in strategy.generate(_context({"600000": _frame([10.0] * 25 + [11.5])},
                                                   ["600000"], hold))
             if s.action == SignalAction.SELL]
    assert taken and "止盈" in taken[0].reason

    # 峰值 13、现价 10.9：未达 +10% 止盈线（11.0），但已跌破 峰值×(1−8%)=11.96 → 移动止损
    peak_hold = {"600000": {**hold["600000"], "trail_peak": 13.0}}
    trailed = [s for s in strategy.generate(_context({"600000": _frame([10.0] * 25 + [10.9])},
                                                    ["600000"], peak_hold))
               if s.action == SignalAction.SELL]
    assert trailed and "移动止损" in trailed[0].reason


# --------------------------------------------------------- 买入与同股去重
def test_custom_strategy_buys_only_unheld(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _register_factor(db, "f_close", "close")
    _register_strategy(db, "策略A",
                       {"conditions": [{"factor": "f_close", "op": ">", "value": 5.0}]}, {})
    strategy = build_strategy(lab_strategy_key("策略A"), {"_database": db})
    frames = {"600000": _frame([10.0] * 20), "600001": _frame([20.0] * 20)}

    buys = [s for s in strategy.generate(_context(frames, [], {})) if s.action == SignalAction.BUY]
    assert {s.code for s in buys} == {"600000", "600001"}
    assert all(s.strategy == "策略A" for s in buys)

    # 已被持有（无论归属哪个策略）→ 不再重复买入
    buys2 = [s for s in strategy.generate(_context(frames, ["600000"], {}))
             if s.action == SignalAction.BUY]
    assert {s.code for s in buys2} == {"600001"}


def test_disabled_strategy_produces_nothing(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _register_factor(db, "f_close", "close")
    _register_strategy(db, "策略A",
                       {"conditions": [{"factor": "f_close", "op": ">", "value": 5.0}]}, {}, enabled=0)
    strategy = build_strategy(lab_strategy_key("策略A"), {"_database": db})
    assert strategy.generate(_context({"600000": _frame([10.0] * 20)}, [], {})) == []


# --------------------------------------------------- 跨策略合并与名额分配
def test_merge_signals_keeps_all_sells_and_caps_buys() -> None:
    positions = {"600000": {}, "600001": {}, "600002": {}}
    merged = SignalService._merge_signals(
        [
            _signal("600000", SignalAction.SELL, 0, "策略A"),
            _signal("600001", SignalAction.SELL, 0, "策略B"),
            _signal("900001", SignalAction.BUY, 1.0, "策略A"),
            _signal("900002", SignalAction.BUY, 3.0, "策略B"),
            _signal("900003", SignalAction.BUY, 2.0, "策略A"),
        ],
        positions,
        max_positions=3,
    )
    sells = {s.code for s in merged if s.action == SignalAction.SELL}
    buys = [s for s in merged if s.action == SignalAction.BUY]
    assert sells == {"600000", "600001"}          # 各策略的卖出信号全部保留
    # 持仓 3 只、待卖出 2 只 → 卖出后仅剩 1 只，释放 2 个名额，按评分取 900002、900003
    assert [s.code for s in buys] == ["900002", "900003"]
    assert buys[0].strategy == "策略B"


def test_merge_signals_respects_position_cap() -> None:
    """持仓已满且无卖出信号时，不产生任何买入（多策略共享总名额）。"""
    merged = SignalService._merge_signals(
        [_signal("900001", SignalAction.BUY, 1.0, "策略A")],
        {"600000": {}, "600001": {}, "600002": {}},
        max_positions=3,
    )
    assert merged == []


def test_merge_signals_caps_buys_across_strategies() -> None:
    """两个策略各出 2 个买入候选、空仓 2 个名额 → 只保留评分最高的 2 个（跨策略竞争）。"""
    merged = SignalService._merge_signals(
        [
            _signal("900001", SignalAction.BUY, 1.5, "策略A"),
            _signal("900002", SignalAction.BUY, 2.5, "策略A"),
            _signal("900003", SignalAction.BUY, 3.5, "策略B"),
            _signal("900004", SignalAction.BUY, 0.5, "策略B"),
        ],
        {},
        max_positions=2,
    )
    assert [s.code for s in merged] == ["900003", "900002"]
    assert {s.strategy for s in merged} == {"策略A", "策略B"}


def test_merge_signals_dedupes_same_code() -> None:
    merged = SignalService._merge_signals(
        [
            _signal("900001", SignalAction.BUY, 1.0, "策略A"),
            _signal("900001", SignalAction.BUY, 5.0, "策略B"),
        ],
        {},
        max_positions=3,
    )
    assert len(merged) == 1
    assert merged[0].strategy == "策略B" and merged[0].score == 5.0


def test_drop_limit_up_only_filters_buys(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = SignalService(db, None, FakeSettings(), None)  # type: ignore[arg-type]
    # 600000 昨收 10，收盘 11 = 主板涨停
    db.execute(
        "INSERT INTO daily_bars(code,trade_date,open,high,low,close,volume,amount,pre_close,source) "
        "VALUES('600000','2026-09-10',11,11,11,11,1000,10000,10,'tushare')"
    )
    kept = service._drop_limit_up(
        [_signal("600000", SignalAction.BUY, 1.0), _signal("600000", SignalAction.SELL, 0.0)],
        "2026-09-10",
    )
    actions = {s.action for s in kept}
    assert SignalAction.BUY not in actions   # 涨停买入被剔除
    assert SignalAction.SELL in actions      # 卖出必须放行，否则漏掉止盈止损


# ----------------------------------------------------- 建仓记账与旧库迁移
def test_paper_fill_records_strategy_peak_and_breakout(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.execute(
        "INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,is_suspended,list_date,updated_at) "
        "VALUES('600000','浦发银行','STOCK','MAIN',0,0,0,'2000-01-01',?)", (NOW,)
    )
    for day, close in [("2026-09-08", 10.0), ("2026-09-09", 10.5), ("2026-09-10", 11.0)]:
        db.execute(
            "INSERT INTO daily_bars(code,trade_date,open,high,low,close,volume,amount,pre_close,source) "
            "VALUES('600000',?,?,?,?,?,1000000,10000000,?,'tushare')",
            (day, close, close, close, close, close),
        )
    broker = PaperBroker(db, FakeSettings(), RiskManager(db, FakeSettings()))  # type: ignore[arg-type]
    broker.submit_order(OrderRequest(code="600000", side=OrderSide.BUY, quantity=1000,
                                     strategy="策略A", trade_date="2026-09-10",
                                     requested_price=11.0, name="浦发银行", id="o1"))
    assert broker.execute_pending_orders("2026-09-10")["filled"] == 1

    row = db.query_one("SELECT * FROM positions WHERE code='600000'")
    assert row["strategy"] == "策略A"
    assert float(row["trail_peak"]) > 0
    # 箱体基准 = 建仓日前 4 个交易日收盘最高（此处 09-08=10.0、09-09=10.5 → 10.5）
    assert float(row["entry_breakout"]) == 10.5


def test_positions_migration_preserves_rows(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE positions (
            code TEXT PRIMARY KEY, name TEXT NOT NULL, quantity INTEGER NOT NULL,
            sellable_quantity INTEGER NOT NULL, avg_cost REAL NOT NULL,
            latest_price REAL NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO positions VALUES('600000','浦发银行',1000,1000,10.0,11.0,'2026-09-01 00:00:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize(100_000)
    row = db.query_one("SELECT * FROM positions WHERE code='600000'")
    assert int(row["quantity"]) == 1000                 # 业务数据未丢失
    assert row["strategy"] == ""
    assert float(row["trail_peak"]) == 10.0             # 迁移用成本价兜底
    assert float(row["entry_breakout"]) == 10.0


# ------------------------------------------------------------- 策略标识解析
def test_lab_strategy_key_and_detection() -> None:
    assert lab_strategy_key("箱体突破") == "lab:箱体突破"
    assert is_lab_strategy("lab:箱体突破")
    assert not is_lab_strategy("lab_signal")
    assert not is_lab_strategy("lab:")
    assert not is_lab_strategy("momentum_rotation")


def test_resolve_strategies_falls_back_to_all_custom(tmp_path: Path) -> None:
    """未配置策略时兜底运行全部「已启用」的自定义策略，而非退回合并模式。"""
    db = _db(tmp_path)
    _register_factor(db, "f_close", "close")
    entry = {"conditions": [{"factor": "f_close", "op": ">", "value": 0.0}]}
    _register_strategy(db, "策略A", entry, {})
    _register_strategy(db, "策略B", entry, {}, enabled=0)   # 停用
    service = SignalService(db, None, FakeSettings(), None)  # type: ignore[arg-type]

    assert service.resolve_strategy_names(None) == ["lab:策略A"]
    assert service.resolve_strategy_names("lab:策略A,lab:策略B") == ["lab:策略A", "lab:策略B"]
    assert service.resolve_strategy_names("lab:策略A,lab:策略A") == ["lab:策略A"]  # 去重保序
