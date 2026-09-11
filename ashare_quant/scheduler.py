"""Long-running APScheduler process for Windows/Linux/server operation."""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from . import ml_model
from .fundamental import update_fundamentals
from .lab import build_buy_report, fetch_scan_quotes, fresh_quotes, scan_buy_candidates
from .main_rally import generate_daily_events_report, generate_main_rally_report
from .services.runtime import Runtime, build_runtime
from .utils import normalize_date, today_text


LOG = logging.getLogger(__name__)

# 盘中即时交易时点（09:25 集合竞价结束后开始，15:00 收盘前结束）。
# 15:05–15:30 是盘后固定价格交易时段，只能按当日收盘价成交、拿不到盘中实时价，
# 因此不作为买入时点（见 ``after_close`` 的收盘价离场）。
DEFAULT_INTRADAY_TIMES = ("09:37", "10:00", "10:30", "14:30", "14:50")


def _skip_if_holiday(runtime: Runtime, label: str) -> bool:
    """返回 True 表示今日休市，应跳过该任务（法定节假日与调休休市）。"""
    if runtime.calendar.is_trading_day(today_text()):
        return False
    LOG.info("今日休市（非交易日），跳过%s", label)
    return True


def _live_quotes(runtime: Runtime) -> dict:
    """取当日刷新成功的实时行情（剔除 market_quotes 里的残留旧价）。"""
    return fresh_quotes(fetch_scan_quotes(runtime.data, runtime.database), today_text())


def _intraday_summary(outcome: dict) -> str:
    """把盘中撮合结果格式化成一行中文摘要。"""
    if outcome.get("skipped"):
        return "本次未成交：实时行情不可用。"
    buys = f"买入成交 {outcome['buy_filled']} 笔"
    sells = f"卖出成交 {outcome['sell_filled']} 笔"
    if outcome["buy_rejected"] or outcome["sell_rejected"]:
        detail = f"（买入被拒 {outcome['buy_rejected']}、卖出被拒 {outcome['sell_rejected']}）"
    else:
        detail = ""
    if outcome["failed"]:
        detail += f"（执行失败 {outcome['failed']}）"
    tail = "。" if (outcome["buy_filled"] or outcome["sell_filled"]) else "，本次无成交。"
    return f"本次盘中即时撮合：{buys}、{sells}{detail}{tail}"


def after_close(runtime: Runtime) -> None:
    """盘后流程（signal_generation 时点）：更新日线 → 买点扫描 → 按收盘价即时离场。

    买入已改为**盘中实时价即时成交**（见 ``intraday_trade``），因此这里**不再排队次日开盘买入**。
    保留的是**收盘价离场**：15:05–15:30 为盘后固定价格交易时段，按当日收盘价成交，所以那些
    不适合盘中实时判断的离场（如因子「离场条件」，需要用当日完整日线）在这里以收盘价执行。
    """
    if _skip_if_holiday(runtime, "盘后流程"):
        return
    if not runtime.control.get_bool("strategy_enabled", True):
        LOG.info("看板已关闭策略运行")
        return
    if not runtime.database.query_one("SELECT 1 FROM stock_basic LIMIT 1"):
        runtime.data.refresh_universe()
    update = runtime.data.update_daily()
    # 数据完整性闸门：最新交易日覆盖率不足则推迟买点扫描与离场，避免用过期收盘价误判
    if not runtime.data.daily_data_complete():
        runtime.notifications.send(
            "盘后流程推迟",
            f"今日日线数据拉取不完整（更新结果={update}），买点扫描与收盘价离场已推迟，避免用过期价格推送。",
            "WARNING",
        )
        LOG.warning("今日日线数据不完整（%s），推迟盘后买点扫描与离场", update)
        return
    has_candidates, buy_report, buy_html = build_buy_report(runtime.data, runtime.database, exclude_prefix="主升浪")
    if has_candidates:
        note = "\n（提示：买入已在盘中按实时价即时成交，本条仅为收盘后候选回顾，不再触发次日开盘买入。）"
        runtime.notifications.send("买点扫描（收盘后回顾）", buy_report + note, "INFO", html=buy_html)
        LOG.info("买点扫描：%s", buy_report.replace("\n", " / "))
    # 收盘价离场：不传 current_quotes → 用当日完整日线（即收盘价）评估
    held = [position.code for position in runtime.broker.get_positions()]
    if not held:
        LOG.info("盘后流程完成：更新结果=%s，当前无持仓，无需离场", update)
        return
    exits = runtime.signals.generate(codes=held)
    outcome = runtime.trading.execute_intraday(
        quotes=_close_prices(runtime, held), candidates=[], exit_signals=exits,
    )
    LOG.info("盘后流程完成：更新结果=%s，收盘价离场=%s", update, outcome)


def _close_prices(runtime: Runtime, codes: list[str], trade_date: str | None = None) -> dict[str, float]:
    """取指定标的的**当日**收盘价（盘后离场按收盘价成交）。

    ``trade_date`` 默认取今天；抽成参数是为了可测（演示数据的最后一条是「最近一个交易日」，
    节假日/周末调用时与自然日不一致，硬编码今天会让测试依赖运行日期）。
    """
    if not codes:
        return {}
    placeholders = ",".join("?" * len(codes))
    rows = runtime.database.query_all(
        f"SELECT code, close FROM daily_bars WHERE trade_date=? AND code IN ({placeholders})",
        [normalize_date(trade_date or today_text()), *codes],
    )
    return {row["code"]: float(row["close"]) for row in rows if float(row["close"] or 0) > 0}


def morning_execution(runtime: Runtime) -> None:
    """开盘前结算（morning_execution 时点）：T+1 可卖数量重置 + 撮合遗留委托。

    买入已改为盘中即时成交，这里**不再有新委托可执行**；保留该任务是因为 T+1 的
    「昨日买入今日可卖」与风控「日内停止状态」重置都发生在 ``roll_to_new_day``，
    必须在当日第一次盘中交易（09:37）之前完成。
    """
    if _skip_if_holiday(runtime, "开盘前结算"):
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


def intraday_trade(runtime: Runtime) -> None:
    """盘中即时交易（每个 intraday_scan_times 时点）：实时价扫描 → 先卖后买即时成交 → 推送报告。

    这是模拟盘**买点的唯一入口**：不再「15:30 生成信号 → 次日 09:26 按开盘价买入」，
    发现候选的当下就以实时价成交。卖出同样即时（止损/止盈/移动止损/跌破箱体/离场条件），
    不再等到次日开盘，避免止损滞后一天。

    实时行情不可用（接口限流/网络异常/当日未刷新成功）时**跳过本次、不做任何成交**：
    宁可当天不出手，也不拿陈旧价格成交。
    """
    if _skip_if_holiday(runtime, "盘中即时交易"):
        return
    if not runtime.control.get_bool("strategy_enabled", True):
        LOG.info("看板已关闭策略运行")
        return
    if not runtime.control.get_bool("paper_execution_enabled", True):
        LOG.info("看板已关闭模拟成交")
        return
    quotes = _live_quotes(runtime)
    if not quotes:
        runtime.notifications.send(
            "盘中交易跳过",
            "实时行情不可用（接口限流 / 网络异常 / 当日未刷新成功），本次不做任何成交，等待下一个盘中时点重试。",
            "WARNING",
        )
        LOG.warning("实时行情不可用，跳过本次盘中即时交易")
        return
    scan = scan_buy_candidates(
        runtime.data, runtime.database, current_quotes=quotes, exclude_prefix="主升浪"
    )
    held = [position.code for position in runtime.broker.get_positions()]
    exits = runtime.signals.generate(current_quotes=quotes, codes=held) if held else []
    outcome = runtime.trading.execute_intraday(
        quotes=quotes, candidates=scan["candidates"], exit_signals=exits,
    )
    summary = _intraday_summary(outcome)
    LOG.info("盘中即时交易：%s；%s", outcome, summary)
    has_candidates, report, html = build_buy_report(runtime.data, runtime.database, scan=scan)
    if has_candidates:
        runtime.notifications.send(
            "盘中买点扫描", f"{report}\n\n{summary}", "INFO",
            html=f'{html}<p style="margin:10px 0 0;font-size:12.5px;color:#2c3e50;">{summary}</p>',
        )
    elif outcome["buy_filled"] or outcome["sell_filled"]:
        runtime.notifications.send("盘中即时成交", summary, "INFO")


def pre_market_scan(runtime: Runtime) -> None:
    """盘前流程（09:25）：抓事件→更新宏观→拉集合竞价行情→形成候选池→推送富文本报告。"""
    if _skip_if_holiday(runtime, "盘前主升浪扫描"):
        return
    # 抓取全球事件（LLM 抽取）+ 更新宏观状态
    update_fundamentals(runtime.database)
    # 拉取集合竞价实时行情（保证报告里的价格是当日而非过时的历史 bar）；
    # 只保留当日刷新成功的行情，拉取失败时回退为日线收盘价并在报告里标注数据日期。
    try:
        quotes = _live_quotes(runtime)
    except Exception:
        LOG.exception("盘前集合竞价行情拉取失败，回退到日线收盘价")
        quotes = {}
    # 形成盘前候选池 + 生成富文本报告（纯文本 + HTML）
    has_candidates, report, html = generate_main_rally_report(runtime, current_quotes=quotes)
    if has_candidates:
        runtime.notifications.send("盘前主升浪扫描", report, "INFO", html=html)
        LOG.info("盘前主升浪扫描：%s", report.replace("\n", " / "))
        return
    # 无主升浪候选时，退而发「今日重要事件」，保证每天盘前都有当日要闻可看
    has_events, ev_text, ev_html = generate_daily_events_report(runtime)
    if has_events:
        runtime.notifications.send("今日重要事件", ev_text, "INFO", html=ev_html)
        LOG.info("盘前今日重要事件：%s", ev_text.replace("\n", " / "))


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
    # 盘中即时交易：模拟盘买点的唯一入口（实时价成交，不再次日开盘买入）
    intraday_times = tuple(str(t) for t in (sched_cfg.get("intraday_scan_times") or DEFAULT_INTRADAY_TIMES))
    for moment in intraday_times:
        hour, minute = _parse_hhmm(moment)
        scheduler.add_job(
            intraday_trade,
            CronTrigger(day_of_week="mon-sun", hour=hour, minute=minute, timezone=timezone),
            args=[runtime], id=f"intraday_trade_{hour:02d}{minute:02d}",
        )
    LOG.info(
        "定时调度器已启动，时区=%s；盘中即时交易=%s，晨间结算=%02d:%02d，盘后离场=%02d:%02d，收盘估值=%02d:%02d，模型重训=%02d:%02d",
        timezone, ",".join(intraday_times), morning_h, morning_m, signal_h, signal_m,
        eod_h, eod_m, retrain_h, retrain_m,
    )
    scheduler.start()
