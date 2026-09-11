"""Load cached market data, run a strategy, and persist standardized signals."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..data.service import DataService
from ..database import Database
from ..market_rules import board_price_limit
from ..models import Signal, SignalAction, utc_now_text
from ..notifications import NotificationHub
from ..presentation import label_value
from ..strategies import build_strategy
from ..strategies.base import StrategyContext


def apply_realtime_quotes(
    bars_by_code: dict[str, Any], current_quotes: dict[str, Any] | None
) -> dict[str, Any]:
    """用实时价覆盖每只标的最新一根 bar 的 ``close``（及当日累计成交量）。

    与 ``lab.find_buy_candidates`` 完全相同的口径，保证盘中「离场判断」与「买点扫描」
    看到的是同一个价格。改动仅作用于最后一根 bar，历史序列不动。
    """
    if not current_quotes:
        return bars_by_code
    updated: dict[str, Any] = {}
    for code, frame in bars_by_code.items():
        quote = current_quotes.get(code)
        price, volume = 0.0, 0.0
        if isinstance(quote, dict):
            price, volume = float(quote.get("price") or 0), float(quote.get("volume") or 0)
        elif quote is not None:
            price = float(quote or 0)
        if price > 0 and frame is not None and not frame.empty:
            frame = frame.copy()
            frame.iloc[-1, frame.columns.get_loc("close")] = price
            if volume > 0:
                frame.iloc[-1, frame.columns.get_loc("volume")] = volume
        updated[code] = frame
    return updated


class SignalService:
    """加载缓存行情、运行策略、持久化标准化信号，并在有新信号时推送通知。"""

    def __init__(self, database: Database, data: DataService, settings: Settings, notifications: NotificationHub):
        self.database = database
        self.data = data
        self.settings = settings
        self.notifications = notifications

    # ------------------------------------------------------------------ 策略解析
    def resolve_strategy_names(self, strategy_name: str | None = None) -> list[str]:
        """解析本次要运行的策略列表。

        优先级：显式入参 > 看板持久化的 ``active_strategy`` > 配置默认值；都为空的兜底
        则是「因子实验室里全部已启用的自定义策略」。值为逗号分隔时按多策略处理，
        并去重保序。
        """
        raw = strategy_name
        if raw is None:
            row = self.database.query_one("SELECT value FROM system_settings WHERE key='active_strategy'")
            raw = str(row["value"]) if row and row["value"] else str(self.settings.active_strategy)
        names = list(dict.fromkeys(part.strip() for part in str(raw).split(",") if part.strip()))
        if names:
            return names
        # 兜底：未配置任何策略时，运行因子实验室里全部已启用的自定义策略，
        # 避免调度器在「看板尚未选择策略」时回退到无买卖点的合并模式。
        return [
            f"lab:{row['name']}"
            for row in self.database.query_all(
                "SELECT name FROM signal_strategies WHERE enabled=1 ORDER BY created_at DESC"
            )
        ]

    # ------------------------------------------------------------------ 主流程
    def generate(
        self,
        as_of_date: str | None = None,
        strategy_name: str | None = None,
        current_quotes: dict[str, Any] | None = None,
        codes: list[str] | None = None,
    ) -> list[Signal]:
        """对当前候选池运行选中的一个或多个策略，生成并落库调仓信号。

        多策略并行时的合并规则：**卖出信号全部保留**（每个策略只管自己买入的持仓，
        各自独立止盈止损），**买入信号跨策略统一排序**后按剩余持仓名额截取，
        保证总持仓不超过 ``max_positions``（多策略共享总名额）。

        ``current_quotes`` 为 ``{code: 实时价}`` 或 ``{code: {"price","volume"}}``：传入时
        用实时价覆盖最新一根 bar 的收盘价，使**买卖判断都在实时价上做**（盘中即时交易用）。
        ``codes`` 限定只跑这些标的，用于「只评估持仓离场」——此时不会产出买入信号
        （持仓已在 ``held_codes`` 里被排除），开销也从全市场降到几只有仓股。
        """
        strategy_names = self.resolve_strategy_names(strategy_name)
        if not strategy_names:
            raise RuntimeError("未选择任何策略，请先在看板「策略」中至少选择一个")
        if codes is not None and not codes:
            return []  # 无持仓时「只评估离场」应为空，避免误报「证券池为空」

        universe = self.data.eligible_universe(limit=int(self.settings.data["strategy_scan_symbols"]))
        if codes is not None:
            wanted = {str(code) for code in codes}
            universe = universe[universe["code"].isin(wanted)].reset_index(drop=True)
        if universe.empty:
            raise RuntimeError("可用证券池为空，请先更新数据或生成演示行情")
        bars_by_code = apply_realtime_quotes(
            self.data.load_bars_many(
                universe["code"].tolist(), start_date=self.data.lookback_start(260), end_date=as_of_date
            ),
            current_quotes,
        )
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

        positions = {
            row["code"]: dict(row)
            for row in self.database.query_all("SELECT * FROM positions WHERE quantity>0")
        }
        max_positions = int(self.settings.risk["max_positions"])

        collected: list[Signal] = []
        for name in strategy_names:
            params = dict(self.settings.strategies.get(name, {}))
            # 因子实验室适配器（合并模式 lab_signal / 单策略 lab:xxx）需要读 signal_strategies 表
            if name == "lab_signal" or name.startswith("lab:"):
                params["_database"] = self.database
            strategy = build_strategy(name, params)
            collected.extend(strategy.generate(StrategyContext(
                as_of_date=effective_date, universe=universe, bars_by_code=bars_by_code,
                held_codes=set(positions), max_positions=max_positions, positions=positions,
            )))

        signals = self._merge_signals(collected, positions, max_positions)
        # 涨停不追：只剔除「信号生成日已涨停」的**买入**信号（卖出必须放行，否则漏掉止盈止损）
        signals = self._drop_limit_up(signals, effective_date)
        self.database.executemany(
            """INSERT OR IGNORE INTO signals(id,code,name,action,as_of_date,strategy,target_weight,score,reason,status,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,'NEW',?)""",
            [(item.id, item.code, item.name, item.action.value, item.as_of_date, item.strategy,
              item.target_weight, item.score, item.reason, utc_now_text()) for item in signals],
        )
        if signals:
            summary = "\n".join(
                f"{label_value(item.action.value, 'action')} {item.code} {item.name}，策略={item.strategy}，评分={item.score:.4f}"
                for item in signals
            )
            self.notifications.send("策略交易信号", summary)
        return signals

    # ------------------------------------------------------------------ 合并/过滤
    @staticmethod
    def _merge_signals(
        signals: list[Signal], positions: dict[str, dict], max_positions: int
    ) -> list[Signal]:
        """跨策略合并信号：卖出全留，买入按评分竞争剩余名额，同一标的只保留一条。"""
        sells: dict[str, Signal] = {}
        buys: dict[str, Signal] = {}
        for item in signals:
            bucket = sells if item.action == SignalAction.SELL else buys
            current = bucket.get(item.code)
            if current is None or float(item.score or 0) > float(current.score or 0):
                bucket[item.code] = item
        # 卖出成交后会释放名额，故买入可用名额按「持仓数 − 待卖出数」计算
        remaining = max(0, max_positions - (len(positions) - len(sells)))
        ranked = sorted(buys.values(), key=lambda sig: float(sig.score or 0), reverse=True)
        return list(sells.values()) + ranked[:remaining]

    def _drop_limit_up(self, signals: list[Signal], effective_date: str) -> list[Signal]:
        """剔除买入信号中信号生成日当天已涨停的标的（追连板股次日涨停买不进，属无效信号）。"""
        kept: list[Signal] = []
        for item in signals:
            if item.action != SignalAction.BUY:
                kept.append(item)  # 卖出信号必须保留，否则止盈止损会被吞掉
                continue
            bar = self.database.query_one(
                "SELECT close, pre_close FROM daily_bars WHERE code=? AND trade_date=?",
                (item.code, effective_date),
            )
            if bar and bar["pre_close"]:
                limit = board_price_limit(item.code, is_st=False)
                limit_price = round(float(bar["pre_close"]) * (1 + limit), 2)
                if float(bar["close"]) >= limit_price - 0.005:
                    continue
            kept.append(item)
        return kept
