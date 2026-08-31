"""Command-line entry point for initialization, data, backtest and paper trading."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from typing import Any

from .data.seed import seed_demo_market
from .fundamental import update_fundamentals
from .hot_strategy import run_hot_backtest
from .lab import build_buy_report, fetch_scan_quotes
from .limit_pullback_strategy import run_limit_pullback_backtest
from .main_rally import generate_main_rally_report, score_main_rally_candidates
from .scheduler import after_close, end_of_day, run_scheduler
from .presentation import localize_payload
from .services.runtime import build_runtime
from .services.trading import next_weekday
from .strategies.factory import STRATEGIES
from .utils import today_text


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自带的英文帮助标题转换为中文。"""

    def __init__(self, *args: Any, **kwargs: Any):
        include_help = kwargs.pop("add_help", True)
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"
        if include_help:
            self.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")

    def format_help(self) -> str:
        return super().format_help().replace("usage: ", "用法：", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "用法：", 1)

    def error(self, message: str) -> None:
        message = message.replace("the following arguments are required:", "缺少必需参数：")
        message = message.replace("unrecognized arguments:", "无法识别的参数：")
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}：错误：{message}\n")


def configure_console_encoding() -> None:
    """确保 Windows 中文控制台与重定向日志均使用 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def print_json(value: Any) -> None:
    """把结果序列化为中文 JSON 打印到控制台（dataclass 自动转 dict）。"""

    def encode(item: Any) -> Any:
        return dataclasses.asdict(item) if dataclasses.is_dataclass(item) else str(item)

    print(json.dumps(localize_payload(value), ensure_ascii=False, indent=2, default=encode))


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器，注册 init-db/seed-demo/backtest 等全部子命令。"""
    parser = ChineseArgumentParser(add_help=False, description="模拟盘优先的 A 股量化交易系统")
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示此帮助信息并退出")
    parser.add_argument("--config", help="可选 YAML 配置文件，将覆盖 config/default.yaml 中的同名配置")
    sub = parser.add_subparsers(
        dest="command", required=True, title="可用命令", metavar="命令", parser_class=ChineseArgumentParser,
    )
    sub.add_parser("init-db", help="初始化 SQLite 数据库与模拟账户")
    seed = sub.add_parser("seed-demo", help="生成可重复的离线演示日线数据")
    seed.add_argument("--days", type=int, default=520, help="每个标的生成的交易日数量")
    update = sub.add_parser("update-data", help="更新证券池与日线数据")
    update.add_argument("--codes", help="以逗号分隔的证券代码；默认使用配置的证券池")
    update.add_argument("--end", default=today_text())
    backtest = sub.add_parser("backtest", help="运行 Backtrader 回测并保存指标与资金曲线")
    backtest.add_argument("--strategy", choices=list(STRATEGIES))
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    signals = sub.add_parser("generate-signals", help="生成并保存下一次调仓信号")
    signals.add_argument("--strategy", choices=list(STRATEGIES))
    signals.add_argument("--date")
    queue = sub.add_parser("queue-orders", help="将新信号转换为模拟委托")
    queue.add_argument("--date")
    execute = sub.add_parser("execute-orders", help="执行模拟盘晨间成交流程")
    execute.add_argument("--date", default=today_text())
    execute.add_argument("--no-quotes", action="store_true", help="不刷新实时行情，使用缓存或演示日线数据")
    sub.add_parser("paper-demo", help="按需生成数据、信号、委托并完成一次离线模拟成交")
    sub.add_parser("status", help="输出当前模拟账户、持仓与风控状态")
    sub.add_parser("scheduler", help="启动长期运行的定时调度器（盘前 09:25/晨间 09:35/盘后 15:30/收盘 15:50/重训 16:10）")
    sub.add_parser("after-close", help="执行一次 15:30 行情更新、信号和委托排队流程")
    sub.add_parser("end-of-day", help="执行一次 15:50 收盘估值与风控流程")
    sub.add_parser("update-fundamentals", help="采集事件库与宏观数据（主升浪策略基础层）")
    sub.add_parser("pre-market", help="盘前流程：抓事件+宏观+候选池+富文本报告")
    sub.add_parser("score-main-rally", help="主升浪评分：事件+宏观+产业链+盈利+技术加权")
    sub.add_parser("find-buys", help="扫描策略买入候选并通过邮件/企业微信发送")
    sub.add_parser("retrain-model", help="重训主升概率模型（收盘后管线，可手动触发）")
    hot = sub.add_parser("hot-backtest", help="短线人气·热度共振策略日线近似回测（落库供看板展示）")
    hot.add_argument("--start", default="2025-01-01", help="回测起始日")
    hot.add_argument("--end", default=None, help="回测结束日，默认今天")
    hot.add_argument("--threshold", type=float, default=75.0, help="强候选评分阈值（小资金档 75）")
    hot.add_argument("--top-n", type=int, default=10, help="每日清单保留只数")
    hot.add_argument("--max-positions", type=int, default=1, help="同时持仓上限（小资金档 1）")
    hot.add_argument("--initial-cash", type=float, default=10000.0, help="初始资金（小资金档默认 1 万）")
    hot.add_argument("--sentiment-scope", choices=["hs", "main"], default="hs", help="情绪统计口径：hs=沪深两市（文档口径），main=仅主板（更严）")
    pullback = sub.add_parser("pullback-backtest", help="涨停回马枪·冲高回调低吸策略日线近似回测（落库供看板展示）")
    pullback.add_argument("--start", default="2020-01-01", help="回测起始日")
    pullback.add_argument("--end", default=None, help="回测结束日，默认今天")
    pullback.add_argument("--threshold", type=float, default=50.0, help="同日多信号打分下限（低于不买）")
    pullback.add_argument("--top-n", type=int, default=3, help="同日最多买入只数")
    pullback.add_argument("--max-positions", type=int, default=3, help="同时持仓上限（文档 §9：3 只）")
    pullback.add_argument("--initial-cash", type=float, default=1_000_000.0, help="初始资金")
    return parser


def main() -> None:
    """命令行入口：解析子命令并分发到对应处理逻辑（初始化/数据/回测/信号/调度等）。"""
    configure_console_encoding()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = build_parser().parse_args()
    if args.command == "scheduler":
        run_scheduler(args.config)
        return
    runtime = build_runtime(args.config)
    if args.command == "init-db":
        print_json({"database": str(runtime.settings.db_path), "mode": runtime.settings.mode, "status": "initialized"})
    elif args.command == "seed-demo":
        print_json({"bars": seed_demo_market(runtime.database, runtime.data, args.days), "source": "synthetic-demo"})
    elif args.command == "update-data":
        universe_count = 0
        if not runtime.database.query_one("SELECT 1 FROM stock_basic LIMIT 1"):
            universe_count = runtime.data.refresh_universe()
        codes = [item.strip().zfill(6) for item in args.codes.split(",")] if args.codes else None
        print_json({"universe": universe_count, **runtime.data.update_daily(codes, args.end)})
    elif args.command == "backtest":
        configured = runtime.settings.raw["backtest"]
        result = runtime.backtest.run(
            args.strategy or runtime.settings.active_strategy,
            args.start or str(configured["start_date"]),
            args.end or str(configured.get("end_date") or today_text()),
        )
        result.pop("equity_curve", None)
        print_json(result)
    elif args.command == "generate-signals":
        generated = runtime.signals.generate(args.date, args.strategy)
        print_json([item.__dict__ for item in generated])
    elif args.command == "queue-orders":
        print_json(runtime.trading.queue_new_signals(args.date))
    elif args.command == "execute-orders":
        print_json(runtime.trading.execute_morning(args.date, not args.no_quotes))
    elif args.command == "paper-demo":
        if not runtime.data.has_data():
            seed_demo_market(runtime.database, runtime.data)
        generated = runtime.signals.generate(strategy_name=runtime.settings.active_strategy)
        as_of = generated[0].as_of_date if generated else today_text()
        queued = runtime.trading.queue_new_signals(as_of)
        executed = runtime.trading.execute_morning(next_weekday(as_of), refresh_quotes=False)
        print_json({
            "signals": len(generated),
            "queued": queued["queued"],
            "skipped": queued["skipped"],
            "executed": executed,
            "account": runtime.broker.get_account(),
        })
    elif args.command == "status":
        print_json({
            "mode": runtime.settings.mode, "account": runtime.broker.get_account(),
            "positions": runtime.broker.get_positions(),
            "risk": runtime.database.query_one("SELECT * FROM risk_state WHERE account_id='paper'"),
        })
    elif args.command == "after-close":
        after_close(runtime)
        print_json({"status": "盘后流程已完成"})
    elif args.command == "end-of-day":
        end_of_day(runtime)
        print_json({"status": "收盘流程已完成"})
    elif args.command == "pre-market":
        update_fundamentals(runtime.database)
        _has, report, _html = generate_main_rally_report(runtime)
        print(report)
    elif args.command == "update-fundamentals":
        print_json(update_fundamentals(runtime.database))
    elif args.command == "score-main-rally":
        print_json(score_main_rally_candidates(runtime))
    elif args.command == "find-buys":
        quotes = fetch_scan_quotes(runtime.data, runtime.database)
        has_candidates, report, html = build_buy_report(runtime.data, runtime.database, current_quotes=quotes)
        if has_candidates:
            runtime.notifications.send("买点扫描", report, "INFO", html=html)
        print(report)
    elif args.command == "retrain-model":
        from . import ml_model
        print_json(ml_model.retrain(runtime.database, runtime.data, runtime.settings))
    elif args.command == "hot-backtest":
        metrics = run_hot_backtest(
            runtime.database, runtime.data, start_date=args.start, end_date=args.end,
            threshold=args.threshold, top_n=args.top_n, max_positions=args.max_positions,
            initial_cash=args.initial_cash, sentiment_scope=args.sentiment_scope,
        )
        trades = metrics.pop("trades")
        equity = metrics.pop("equity")
        print_json(metrics)
        if trades:
            print("\n最近 10 笔交易：")
            for t in trades[-10:]:
                print(
                    f"{t['entry_date']} {t['code']} {t['name']} [{t['mode']}] 评分{t['score']:.0f}"
                    f"  {t['entry_price']} → {t['exit_date']} {t['exit_price']}  {t['pnl_pct']:+.2%}（{t['reason']}）"
                )
    elif args.command == "pullback-backtest":
        metrics = run_limit_pullback_backtest(
            runtime.database, runtime.data, start_date=args.start, end_date=args.end,
            threshold=args.threshold, top_n=args.top_n, max_positions=args.max_positions,
            initial_cash=args.initial_cash,
        )
        trades = metrics.pop("trades")
        metrics.pop("equity", None)
        print_json(metrics)
        if trades:
            print("\n最近 10 笔交易：")
            for t in trades[-10:]:
                print(
                    f"{t['entry_date']} {t['code']} {t['name']} 评分{t['score']:.0f}"
                    f"  {t['entry_price']} → {t['exit_date']} {t['exit_price']}  {t['pnl_pct']:+.2%}（{t['reason']}）"
                )


if __name__ == "__main__":
    main()
