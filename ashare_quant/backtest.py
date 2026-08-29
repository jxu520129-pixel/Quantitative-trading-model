"""Backtrader engine with A-share costs, T+1, lot sizing and price-limit checks."""

from __future__ import annotations

import json
import math
from datetime import datetime
from uuid import uuid4

import backtrader as bt
import numpy as np
import pandas as pd

from .config import Settings
from .data.service import DataService
from .database import Database
from .market_rules import at_limit
from .models import utc_now_text
from .strategies.factor_strategy import FACTOR_STRATEGY_SPECS
from .strategies.factors import cross_sectional_rank_score


class ASharePandasData(bt.feeds.PandasData):
    lines = ("amount", "pre_close")
    params = (("amount", "amount"), ("pre_close", "pre_close"))


class AShareCommissionInfo(bt.CommInfoBase):
    params = (
        ("commission_rate", 0.00025), ("min_commission", 5.0),
        ("stamp_duty_rate", 0.001), ("stocklike", True), ("commtype", bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        amount = abs(size * price)
        commission = max(amount * self.p.commission_rate, self.p.min_commission) if size else 0.0
        return commission + (amount * self.p.stamp_duty_rate if size < 0 else 0.0)


class EquityCurveAnalyzer(bt.Analyzer):
    def start(self):
        self.values = []

    def next(self):
        self.values.append((self.strategy.datetime.date(0).isoformat(), float(self.strategy.broker.getvalue())))

    def get_analysis(self):
        return self.values


class RankingBacktraderStrategy(bt.Strategy):
    """Run the selected cross-sectional strategy every Monday at that session's open."""

    params = (
        ("strategy_name", "momentum_rotation"), ("parameters", None),
        ("max_positions", 5), ("max_daily_buys", 2),
    )

    def __init__(self):
        self.parameters = self.p.parameters or {}
        self.last_rebalance_week = None
        self.last_buy_date: dict[str, object] = {}

    def next_open(self):
        current_date = self.datetime.date(0)
        week = current_date.isocalendar()[:2]
        if week == self.last_rebalance_week:
            return
        self.last_rebalance_week = week
        scored = self._score_universe()
        data_by_name = {data._name: data for data in self.datas}
        ranked = [(data_by_name[code], score) for code, score in scored if code in data_by_name]
        ranked.sort(key=lambda item: item[1], reverse=True)
        selected = [data for data, _score in ranked[: self.p.max_positions]]
        selected_names = {data._name for data in selected}

        for data in self.datas:
            position = self.getposition(data)
            if position.size <= 0 or data._name in selected_names:
                continue
            if self.last_buy_date.get(data._name) == current_date:
                continue
            previous_close = float(data.close[-1]) if len(data) > 1 else None
            if not at_limit(float(data.open[0]), previous_close, data._name, "SELL"):
                self.sell(data=data, size=position.size)

        buys = 0
        target_weight = 1.0 / self.p.max_positions
        for data in selected:
            position = self.getposition(data)
            if position.size <= 0:
                if buys >= self.p.max_daily_buys:
                    continue
                previous_close = float(data.close[-1]) if len(data) > 1 else None
                if at_limit(float(data.open[0]), previous_close, data._name, "BUY"):
                    continue
                buys += 1
            self._order_target_lot(data, target_weight)

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self.last_buy_date[order.data._name] = self.datetime.date(0)

    def _order_target_lot(self, data, target_weight: float) -> None:
        """Backtrader's target-percent helper does not enforce the A-share 100-share lot."""
        price = float(data.open[0])
        if price <= 0:
            return
        position = self.getposition(data)
        target_value = self.broker.getvalue() * target_weight
        difference = target_value - position.size * price
        shares = int(abs(difference) // (price * 100)) * 100
        if shares <= 0:
            return
        if difference > 0:
            self.buy(data=data, size=shares)
        else:
            self.sell(data=data, size=min(position.size, shares))

    def _score(self, data) -> float | None:
        name = self.p.strategy_name
        if name == "momentum_rotation":
            lookback = int(self.parameters.get("lookback_days", 60))
            if len(data) <= lookback + 1:
                return None
            amounts = [float(data.amount[-index]) for index in range(1, min(21, len(data)))]
            if amounts and np.mean(amounts) < float(self.parameters.get("min_average_amount", 0)):
                return None
            return float(data.close[-1] / data.close[-lookback - 1] - 1)
        if name == "dual_moving_average":
            fast = int(self.parameters.get("fast_window", 10))
            slow = int(self.parameters.get("slow_window", 30))
            if len(data) <= slow:
                return None
            fast_ma = np.mean([float(data.close[-index]) for index in range(1, fast + 1)])
            slow_ma = np.mean([float(data.close[-index]) for index in range(1, slow + 1)])
            return float(fast_ma / slow_ma - 1) if fast_ma > slow_ma else None
        raise ValueError(f"不支持的回测策略：{name}")

    def _score_universe(self) -> list[tuple[str, float]]:
        specs = FACTOR_STRATEGY_SPECS.get(self.p.strategy_name)
        if specs is not None:
            frames = {data._name: self._frame(data) for data in self.datas}
            scored = cross_sectional_rank_score(frames, specs, self.parameters)
            return [(code, score) for code, score, _reason in scored]
        ranked: list[tuple[str, float]] = []
        for data in self.datas:
            score = self._score(data)
            if score is not None and math.isfinite(score):
                ranked.append((data._name, score))
        return ranked

    def _frame(self, data, lookback: int = 400) -> pd.DataFrame:
        # 只取已收盘的 bar（ago=1 起），与实盘信号生成使用完整交易日口径一致，避免把当日
        # cheat_on_open 尚未定型的 close 当作已收盘价。
        n = min(len(data) - 1, lookback)
        if n <= 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount", "pre_close"])
        return pd.DataFrame({
            "open": np.asarray(data.open.get(size=n, ago=1), dtype=float),
            "high": np.asarray(data.high.get(size=n, ago=1), dtype=float),
            "low": np.asarray(data.low.get(size=n, ago=1), dtype=float),
            "close": np.asarray(data.close.get(size=n, ago=1), dtype=float),
            "volume": np.asarray(data.volume.get(size=n, ago=1), dtype=float),
            "amount": np.asarray(data.amount.get(size=n, ago=1), dtype=float),
            "pre_close": np.asarray(data.pre_close.get(size=n, ago=1), dtype=float),
        })


class BacktestEngine:
    def __init__(self, database: Database, data_service: DataService, settings: Settings):
        self.database = database
        self.data_service = data_service
        self.settings = settings

    def run(self, strategy_name: str, start_date: str, end_date: str, initial_cash: float | None = None) -> dict[str, object]:
        universe = self.data_service.eligible_universe(limit=int(self.settings.data["backtest_max_symbols"]))
        bars = self.data_service.load_bars_many(universe["code"].tolist(), start_date, end_date)
        feeds: list[tuple[str, pd.DataFrame]] = [
            (code, frame) for code, frame in bars.items() if len(frame) >= 80
        ]
        if not feeds:
            raise RuntimeError("没有可用于回测的日线数据，请先执行 seed-demo 或 update-data")

        cash = float(initial_cash or self.settings.raw["backtest"]["initial_cash"])
        parameters = dict(self.settings.strategies[strategy_name])
        cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
        cerebro.broker.setcash(cash)
        cerebro.broker.addcommissioninfo(AShareCommissionInfo(
            commission_rate=float(self.settings.trading["commission_rate"]),
            min_commission=float(self.settings.trading["min_commission"]),
            stamp_duty_rate=float(self.settings.trading["stamp_duty_rate"]),
        ))
        cerebro.broker.set_slippage_perc(float(self.settings.trading["slippage_rate"]), slip_open=True, slip_match=True)
        for code, frame in feeds:
            data = frame.copy()
            data.index = pd.DatetimeIndex(data["trade_date"])
            cerebro.adddata(ASharePandasData(dataname=data), name=code)
        cerebro.addstrategy(
            RankingBacktraderStrategy, strategy_name=strategy_name, parameters=parameters,
            max_positions=int(self.settings.risk["max_positions"]),
            max_daily_buys=int(self.settings.risk["max_daily_new_positions"]),
        )
        cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        strategy = cerebro.run(runonce=False)[0]

        equity = list(strategy.analyzers.equity.get_analysis())
        final_equity = float(cerebro.broker.getvalue())
        if len(equity) >= 2:
            days = max(1, (pd.Timestamp(equity[-1][0]) - pd.Timestamp(equity[0][0])).days)
        else:
            days = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
        total_return = final_equity / cash - 1
        annual_return = (final_equity / cash) ** (365.25 / days) - 1
        benchmark_code = str(self.settings.raw["backtest"]["benchmark"])
        benchmark = self.data_service.load_bars(benchmark_code, start_date, end_date)
        benchmark_return = 0.0
        benchmark_annual_return = 0.0
        if len(benchmark) >= 2:
            benchmark_return = float(benchmark["close"].iloc[-1] / benchmark["close"].iloc[0] - 1)
            benchmark_annual_return = (1 + benchmark_return) ** (365.25 / days) - 1 if benchmark_return > -1 else -1.0
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        total_trades = int(trades.get("total", {}).get("closed", 0))
        won = int(trades.get("won", {}).get("total", 0))
        sharpe = strategy.analyzers.sharpe.get_analysis().get("sharperatio")
        result: dict[str, object] = {
            "run_id": uuid4().hex, "strategy": strategy_name, "start_date": start_date, "end_date": end_date,
            "initial_cash": cash, "final_equity": final_equity, "total_return": total_return,
            "annual_return": annual_return, "benchmark": benchmark_code,
            "benchmark_return": benchmark_return, "benchmark_annual_return": benchmark_annual_return,
            "excess_annual_return": annual_return - benchmark_annual_return,
            "max_drawdown": float(drawdown.get("max", {}).get("drawdown", 0)) / 100,
            "sharpe": None if sharpe is None else float(sharpe),
            "win_rate": won / total_trades if total_trades else 0.0, "total_trades": total_trades,
            "equity_curve": equity,
        }
        self._store(result, parameters)
        return result

    def _store(self, result: dict[str, object], parameters: dict[str, object]) -> None:
        self.database.execute(
            """INSERT INTO backtest_runs(id,strategy,start_date,end_date,initial_cash,final_equity,annual_return,
               max_drawdown,sharpe,win_rate,total_trades,parameters_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["run_id"], result["strategy"], result["start_date"], result["end_date"], result["initial_cash"],
             result["final_equity"], result["annual_return"], result["max_drawdown"], result["sharpe"],
             result["win_rate"], result["total_trades"], json.dumps({"strategy": parameters, "benchmark": {
                 "code": result["benchmark"], "return": result["benchmark_return"],
                 "annual_return": result["benchmark_annual_return"], "excess_annual_return": result["excess_annual_return"],
             }}, ensure_ascii=False), utc_now_text()),
        )
        self.database.executemany(
            "INSERT INTO backtest_equity(run_id,trade_date,equity) VALUES(?,?,?)",
            [(result["run_id"], date, value) for date, value in result["equity_curve"]],
        )
