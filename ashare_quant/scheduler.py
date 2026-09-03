"""Long-running APScheduler process for Windows/Linux/server operation."""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from . import ml_model
from .fundamental import update_fundamentals
from .lab import build_buy_report, fetch_scan_quotes
from .main_rally import generate_main_rally_report
from .services.runtime import Runtime, build_runtime
from .utils import today_text


LOG = logging.getLogger(__name__)


def _skip_if_holiday(runtime: Runtime, label: str) -> bool:
    """返回 True 表示今日休市，应跳过该任务（法定节假日与调休休市）。"""
    if runtime.calendar.is_trading_day(today_text()):
        return False
    LOG.info("今日休市（非交易日），跳过%s", label)
    return True


def after_close(runtime: Runtime) -> None:
    """盘后流程（signal_generation 时点）：更新日线 → 买点扫描 → 生成并排队次日信号。"""
    if _skip_if_holiday(runtime, "盘后流程"):
        return
    if not runtime.control.get_bool("strategy_enabled", True):
        LOG.info("看板已关闭策略运行")
        return
    if not runtime.database.query_one("SELECT 1 FROM stock_basic LIMIT 1"):
        runtime.data.refresh_universe()
    update = runtime.data.update_daily()
    # 数据完整性闸门：最新交易日覆盖率不足则推迟买点扫描与信号生成，避免推送过期收盘价
    if not runtime.data.daily_data_complete():
        runtime.notifications.send(
            "盘后流程推迟",
            f"今日日线数据拉取不完整（更新结果={update}），买点扫描与信号生成已推迟，避免用过期价格推送。",
            "WARNING",
        )
        LOG.warning("今日日线数据不完整（%s），推迟盘后买点扫描与信号生成", update)
        return
    has_candidates, buy_report, buy_html = build_buy_report(runtime.data, runtime.database, exclude_prefix="主升浪")
    if has_candidates:
        runtime.notifications.send("买点扫描", buy_report, "INFO", html=buy_html)
        LOG.info("买点扫描：%s", buy_report.replace("\n", " / "))
    strategy_name = runtime.control.get("active_strategy", runtime.settings.active_strategy)
    if strategy_name == "momentum_rotation" and datetime.now().weekday() != int(runtime.settings.raw["scheduler"]["momentum_rebalance_weekday"]):
        LOG.info("日线数据已更新，今日不是周度动量轮动调仓日")
        return
    signals = runtime.signals.generate(strategy_name=strategy_name)
    queued = runtime.trading.queue_new_signals(signals[0].as_of_date if signals else None)
    LOG.info("盘后流程完成：更新结果=%s，信号数量=%s，委托结果=%s", update, len(signals), queued)


def morning_execution(runtime: Runtime) -> None:
    """晨间成交（morning_execution 时点）：执行模拟盘当日待成交委托。"""
    if _skip_if_holiday(runtime, "晨间成交"):
        return
    if not runtime.control.get_bool("paper_execution_enabled", True):
        LOG.info("看板已关闭模拟成交")
        return
    LOG.info("晨间模拟成交结果：%s", runtime.trading.execute_morning(today_text()))


def end_of_day(runtime: Runtime) -> None:
    """收盘估值（end_of_day 时点）：按收盘价更新账户净值并检查日亏损限制。"""
    if _skip_if_holiday(runtime, "收盘估值"):
        return
    account = runtime.broker.mark_to_market(today_text())
    runtime.risk.evaluate_daily_loss(today_text())
    LOG.info("收盘总资产：%.2f", account.total_equity)


def retrain_model(runtime: Runtime) -> None:
    """收盘后重训主升概率模型（技术方案 §17.2「收盘后：重训/更新」）。"""
    if _skip_if_holiday(runtime, "模型重训"):
        return
    try:
        metrics = ml_model.retrain(runtime.database, runtime.data, runtime.settings)
        LOG.info(
            "主升概率模型重训完成：backend=%s 样本=%s 验证AUC=%s",
            metrics.get("backend"), metrics.get("samples"), metrics.get("validation_auc"),
        )
    except Exception as error:
        LOG.warning("主升概率模型重训失败（不影响交易流程）：%s", error)


def intraday_buy_scan(runtime: Runtime) -> None:
    """盘中买点扫描：用实时价与当日成交量覆盖最新 bar，只在有候选时发送邮件。"""
    if _skip_if_holiday(runtime, "盘中买点扫描"):
        return
    quotes = fetch_scan_quotes(runtime.data, runtime.database)
    has_candidates, report, html = build_buy_report(runtime.data, runtime.database, current_quotes=quotes, exclude_prefix="主升浪")
    if not has_candidates:
        return
    runtime.notifications.send("盘中买点扫描", report, "INFO", html=html)
    LOG.info("盘中买点扫描发现候选：%s", report.replace("\n", " / "))


def pre_market_scan(runtime: Runtime) -> None:
    """盘前流程（09:25）：抓事件→更新宏观→拉集合竞价行情→形成候选池→推送富文本报告。"""
    if _skip_if_holiday(runtime, "盘前主升浪扫描"):
        return
    # 抓取全球事件（LLM 抽取）+ 更新宏观状态
    update_fundamentals(runtime.database)
    # 拉取集合竞价实时行情（保证报告里的价格是当日而非过时的历史 bar）
    try:
        quotes = fetch_scan_quotes(runtime.data, runtime.database)
    except Exception:
        LOG.exception("盘前集合竞价行情拉取失败，回退到日线收盘价")
        quotes = {}
    # 形成盘前候选池 + 生成富文本报告（纯文本 + HTML）
    has_candidates, report, html = generate_main_rally_report(runtime, current_quotes=quotes)
    if not has_candidates:
        return
    runtime.notifications.send("盘前主升浪扫描", report, "INFO", html=html)
    LOG.info("盘前主升浪扫描：%s", report.replace("\n", " / "))


def _parse_hhmm(value: str) -> tuple[int, int]:
    """把 ``HH:MM`` 字符串解析为 ``(时, 分)`` 整数元组。"""
    hour_text, minute_text = value.split(":", 1)
    return int(hour_text), int(minute_text)


def run_scheduler(config_path: str | None = None) -> None:
    """启动长期运行的阻塞式调度器：按配置注册盘前/盘中/盘后/重训等定时任务。"""
    runtime = build_runtime(config_path)
    sched_cfg = runtime.settings.raw["scheduler"]
    timezone = str(sched_cfg["timezone"])
    morning_h, morning_m = _parse_hhmm(str(sched_cfg["morning_execution"]))
    signal_h, signal_m = _parse_hhmm(str(sched_cfg["signal_generation"]))
    eod_h, eod_m = _parse_hhmm(str(sched_cfg["end_of_day"]))
    retrain_h, retrain_m = _parse_hhmm(str(sched_cfg.get("retrain", "16:10")))
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(pre_market_scan, CronTrigger(day_of_week="mon-sun", hour=9, minute=25, timezone=timezone), args=[runtime], id="pre_market")
    scheduler.add_job(morning_execution, CronTrigger(day_of_week="mon-sun", hour=morning_h, minute=morning_m, timezone=timezone), args=[runtime], id="morning")
    scheduler.add_job(after_close, CronTrigger(day_of_week="mon-sun", hour=signal_h, minute=signal_m, timezone=timezone), args=[runtime], id="after_close")
    scheduler.add_job(end_of_day, CronTrigger(day_of_week="mon-sun", hour=eod_h, minute=eod_m, timezone=timezone), args=[runtime], id="end_of_day")
    scheduler.add_job(retrain_model, CronTrigger(day_of_week="mon-sun", hour=retrain_h, minute=retrain_m, timezone=timezone), args=[runtime], id="retrain_model")
    for hour, minute in ((9, 37), (10, 00), (10, 30), (14, 30), (14, 50), (15, 30)):
        scheduler.add_job(
            intraday_buy_scan,
            CronTrigger(day_of_week="mon-sun", hour=hour, minute=minute, timezone=timezone),
            args=[runtime], id=f"buy_scan_{hour}_{minute}",
        )
    LOG.info(
        "定时调度器已启动，时区=%s；晨间成交=%02d:%02d，盘后信号=%02d:%02d，收盘估值=%02d:%02d，模型重训=%02d:%02d",
        timezone, morning_h, morning_m, signal_h, signal_m, eod_h, eod_m, retrain_h, retrain_m,
    )
    scheduler.start()
