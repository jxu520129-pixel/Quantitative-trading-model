"""Streamlit operations dashboard for the paper-trading runtime."""

from __future__ import annotations

import hmac
import json
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from ashare_quant.hot_strategy import run_hot_backtest
from ashare_quant.limit_pullback_strategy import run_limit_pullback_backtest
from ashare_quant.lab import BUILTIN_FACTOR_FORMULAS, BUILTIN_STRATEGY_TEMPLATES, build_buy_report, evaluate_factor, fetch_scan_quotes, resolve_factor_expressions, run_signal_backtest
from ashare_quant.data.providers import sina_spot_prices
from ashare_quant.models import utc_now_text
from ashare_quant.services.runtime import build_runtime
from ashare_quant.services.trading import next_weekday
from ashare_quant.presentation import STRATEGY_LABELS, label_strategy, label_value, localize_dataframe
from ashare_quant.strategies.factory import STRATEGIES
from ashare_quant.utils import today_text


st.set_page_config(page_title="A 股量化交易控制台", page_icon="Q", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background: #090b0d; }
      [data-testid="stSidebar"] { background: #0e1114; border-right: 1px solid #23292e; }
      [data-testid="stMetric"] { background: #12161a; border: 1px solid #252c31; padding: 14px; border-radius: 6px; }
      [data-testid="stMetricValue"] { font-size: 1.45rem; line-height: 1.25; }
      .block-container { padding-top: 1.4rem; max-width: 1500px; }
      h1 { font-size: 1.85rem !important; letter-spacing: 0 !important; }
      h2, h3 { letter-spacing: 0 !important; }
      div.stButton > button { border-radius: 5px; border-color: #30383e; min-height: 40px; }
      [data-testid="stDataFrame"] { border: 1px solid #252c31; }
      [data-testid="stToolbar"], [data-testid="stElementToolbar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def runtime():
    return build_runtime()


app = runtime()


def frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.DataFrame(app.database.query_all(sql, params))


def execution_summary(outcome: dict[str, int]) -> str:
    """将内部执行统计转换为看板上的中文提示。"""
    return "，".join([
        f"成交 {outcome.get('filled', 0)} 笔",
        f"拒绝 {outcome.get('rejected', 0)} 笔",
        f"失败 {outcome.get('failed', 0)} 笔",
    ])


def _lab_sample_frame(periods: int = 220) -> pd.DataFrame:
    """用于校验公式语法的最小合成行情。"""
    return pd.DataFrame({
        "open": np.linspace(10, 22, periods), "high": np.linspace(10.4, 22.4, periods),
        "low": np.linspace(9.6, 21.6, periods), "close": np.linspace(10, 22, periods),
        "volume": np.full(periods, 2_000_000.0), "amount": np.linspace(2e7, 4e7, periods),
        "pre_close": np.linspace(9.9, 21.9, periods),
    })


def _conditions_from_df(edited: pd.DataFrame) -> list[dict]:
    conds = []
    for item in edited.itertuples(index=False):
        if item.factor and pd.notna(item.value):
            conds.append({"factor": str(item.factor), "op": str(item.op), "value": float(item.value)})
    return conds


def authorized() -> bool:
    """会话内记住管理员登录状态，避免每次 rerun 都重新输入密码。"""
    expected = app.settings.dashboard_password
    if not expected:
        return False
    if st.session_state.get("_admin_ok", False):
        return True
    provided = st.text_input("管理员密码", type="password", placeholder="请输入密码后回车", key="admin_password")
    if provided:
        if hmac.compare_digest(provided, expected):
            st.session_state["_admin_ok"] = True
            st.rerun()
        else:
            st.error("密码错误，请重试")
    return False


st.title("A 股量化交易控制台")
st.caption(f"模拟盘 · 东方财富模拟账户 · 数据库：{app.settings.db_path.name}")

if not app.settings.dashboard_password:
    is_admin = False
    st.warning("未配置管理员密码：所有操作按钮已禁用。请在项目 `.env` 中设置 `DASHBOARD_PASSWORD` 后重启看板。")
else:
    st.markdown("**🔒 管理员验证 —— 在下方输入密码并回车解锁：**")
    is_admin = authorized()
    if not is_admin:
        st.info("🔒 账户、持仓、委托与回测数据已锁定，输入正确密码后自动解锁。")
        st.stop()

st.subheader("运行控制 · 主流量化交易流程")
strategy_enabled = app.control.get_bool("strategy_enabled", True)
paper_enabled = app.control.get_bool("paper_execution_enabled", True)

ctrl_toggle_1, ctrl_toggle_2, ctrl_strategy = st.columns(3)
new_strategy_enabled = ctrl_toggle_1.toggle("策略运行", value=strategy_enabled, disabled=not is_admin)
new_paper_enabled = ctrl_toggle_2.toggle("模拟成交", value=paper_enabled, disabled=not is_admin)
if is_admin and new_strategy_enabled != strategy_enabled:
    app.control.set_bool("strategy_enabled", new_strategy_enabled)
if is_admin and new_paper_enabled != paper_enabled:
    app.control.set_bool("paper_execution_enabled", new_paper_enabled)

strategy_options = list(STRATEGIES)
active_strategy = app.control.get("active_strategy", app.settings.active_strategy)
selected_strategy = ctrl_strategy.selectbox(
    "策略", strategy_options, index=strategy_options.index(active_strategy) if active_strategy in strategy_options else 0,
    disabled=not is_admin, format_func=label_strategy,
)
if is_admin and selected_strategy != active_strategy:
    app.control.set("active_strategy", selected_strategy)

st.markdown("**交易步骤**（选股信号 → 风控排队 → 执行成交，覆盖买入/卖出）")
step_1, step_2, step_3 = st.columns(3)
if step_1.button("① 选股信号", disabled=not is_admin, width="stretch"):
    try:
        generated = app.signals.generate(strategy_name=selected_strategy)
        buys = sum(1 for s in generated if s.action.value == "BUY")
        sells = sum(1 for s in generated if s.action.value == "SELL")
        st.success(f"已生成 {len(generated)} 个信号（买入 {buys} / 卖出 {sells}）")
        st.cache_data.clear()
    except Exception as error:
        st.error(str(error))
if step_2.button("② 风控排队", disabled=not is_admin, width="stretch"):
    try:
        queued = app.trading.queue_new_signals()
        st.success(f"已排队 {queued['queued']} 笔委托（跳过 {queued['skipped']} 笔）")
        st.cache_data.clear()
    except Exception as error:
        st.error(str(error))
if step_3.button("③ 执行成交", disabled=not is_admin or not paper_enabled, width="stretch"):
    try:
        st.success(execution_summary(app.trading.execute_morning()))
        st.cache_data.clear()
    except Exception as error:
        st.error(str(error))

if st.button("一键全流程（选股→排队→成交）", disabled=not is_admin or not paper_enabled, width="stretch"):
    try:
        generated = app.signals.generate(strategy_name=selected_strategy)
        as_of = generated[0].as_of_date if generated else today_text()
        queued = app.trading.queue_new_signals(as_of)
        executed = app.trading.execute_morning(next_weekday(as_of), refresh_quotes=False)
        st.success(f"信号 {len(generated)} 个 → 排队 {queued['queued']} 笔 → {execution_summary(executed)}")
        st.cache_data.clear()
    except Exception as error:
        st.error(str(error))

account = app.broker.get_account()
positions = app.broker.get_positions()
metrics = st.columns(5)
metrics[0].metric("总资产", f"¥{account.total_equity:,.2f}")
metrics[1].metric("可用现金", f"¥{account.cash:,.2f}")
metrics[2].metric("持仓市值", f"¥{account.market_value:,.2f}")
metrics[3].metric("持仓数量", f"{len(positions)} / {app.settings.risk['max_positions']}")
risk_state = app.database.query_one("SELECT * FROM risk_state WHERE account_id='paper'") or {}
metrics[4].metric("风控状态", "暂停" if risk_state.get("paused") or risk_state.get("daily_open_blocked") else "正常")

lab_tab, overview, holdings, orders, signals_tab, risk_tab, backtest_tab, settings_tab = st.tabs(
    ["因子实验室", "账户", "持仓", "委托成交", "交易信号", "风险监控", "回测", "配置"]
)
with lab_tab:
    lab_bt, lab_scan, lab_factor, lab_strategy = st.tabs(
        ["回测与记录", "盘中买点扫描", "因子管理", "策略定义"]
    )

    # ── 回测与记录（最常用，默认展开）──
    with lab_bt:
        saved_strategies = app.database.query_all("SELECT * FROM signal_strategies ORDER BY created_at DESC")
        if not saved_strategies:
            st.info("暂无已保存的策略，请先到「策略定义」标签页导入模板或自建策略")
        else:
            selected_strategy_name = st.selectbox("选择策略", [s["name"] for s in saved_strategies], key="lab_bt_strategy")
            strategy_row = next(s for s in saved_strategies if s["name"] == selected_strategy_name)
            d1, d2 = st.columns(2)
            lab_start = d1.date_input("开始日期", value=pd.Timestamp("2024-01-01"), key="lab_bt_start")
            lab_end = d2.date_input("结束日期", value=pd.Timestamp.today(), key="lab_bt_end")
            n1, n2, n3 = st.columns(3)
            lab_cash = n1.number_input("初始资金", value=1_000_000.0, step=100_000.0, min_value=100_000.0, key="lab_bt_cash")
            lab_maxpos = n2.number_input("最大持仓数", value=5, min_value=1, max_value=20, step=1, key="lab_bt_maxpos")
            lab_exposure = n3.number_input("总仓位比例（%，留现金降回撤）", min_value=10, max_value=100, value=100, step=5, key="lab_bt_exposure") / 100.0
            if st.button("运行回测", disabled=not is_admin):
                try:
                    entry_cfg = json.loads(strategy_row["entry_json"])
                    engine = entry_cfg.get("engine")
                    if engine == "hot_score":
                        hot_metrics = run_hot_backtest(
                            app.database, app.data, start_date=lab_start.strftime("%Y-%m-%d"),
                            end_date=lab_end.strftime("%Y-%m-%d"), threshold=float(entry_cfg.get("threshold", 75)),
                            max_positions=int(lab_maxpos), initial_cash=float(lab_cash),
                            daily_budget=float(lab_exposure), sentiment_scope=str(entry_cfg.get("sentiment_scope", "hs")),
                        )
                    elif engine == "limit_pullback_score":
                        hot_metrics = run_limit_pullback_backtest(
                            app.database, app.data, start_date=lab_start.strftime("%Y-%m-%d"),
                            end_date=lab_end.strftime("%Y-%m-%d"), threshold=float(entry_cfg.get("threshold", 50)),
                            max_positions=int(lab_maxpos), initial_cash=float(lab_cash),
                            daily_budget=float(lab_exposure),
                        )
                    else:
                        hot_metrics = None
                    if hot_metrics is not None:
                        equity = hot_metrics.pop("equity", [])
                        trades = hot_metrics.pop("trades", [])
                        metrics = {
                            "total_trades": hot_metrics["total_trades"], "win_rate": hot_metrics["win_rate"],
                            "total_pnl": hot_metrics["final_equity"] - hot_metrics["initial_cash"],
                            "total_return": hot_metrics["final_equity"] / hot_metrics["initial_cash"] - 1,
                            "max_drawdown": hot_metrics["max_drawdown"],
                        }
                        st.session_state["lab_trades"] = trades
                        st.session_state["lab_metrics"] = metrics
                        st.session_state["lab_equity"] = equity
                        st.success(f"回测完成，共 {metrics['total_trades']} 笔交易")
                    else:
                        exit_cfg = json.loads(strategy_row["exit_json"])
                        factor_names = [c["factor"] for c in entry_cfg["conditions"]] + [c["factor"] for c in exit_cfg["conditions"]]
                        exprs = resolve_factor_expressions(app.database, factor_names)
                        trades, metrics, equity = run_signal_backtest(
                            app.data, app.settings, exprs, entry_cfg, exit_cfg,
                            lab_start.strftime("%Y-%m-%d"), lab_end.strftime("%Y-%m-%d"),
                            initial_cash=float(lab_cash), max_positions=int(lab_maxpos), exposure=float(lab_exposure),
                        )
                        run_id = uuid4().hex
                        app.database.execute(
                            """INSERT INTO signal_backtest_runs(id,strategy_name,start_date,end_date,metrics_json,created_at)
                               VALUES(?,?,?,?,?,?)""",
                            (run_id, selected_strategy_name, lab_start.strftime("%Y-%m-%d"), lab_end.strftime("%Y-%m-%d"),
                             json.dumps(metrics, ensure_ascii=False), utc_now_text()),
                        )
                        app.database.executemany(
                            """INSERT INTO signal_trades(run_id,code,name,entry_date,entry_price,shares,exit_date,exit_price,pnl,pnl_pct,holding_days,status,entry_reason,exit_reason)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            [(run_id, t["code"], t["name"], t["entry_date"], t["entry_price"], t["shares"], t["exit_date"],
                              t["exit_price"], t["pnl"], t["pnl_pct"], t["holding_days"], t["status"], t["entry_reason"], t["exit_reason"])
                             for t in trades],
                        )
                        st.session_state["lab_trades"] = trades
                        st.session_state["lab_metrics"] = metrics
                        st.session_state["lab_equity"] = equity
                        st.success(f"回测完成，共 {metrics['total_trades']} 笔交易")
                except Exception as error:
                    st.error(str(error))

            if "lab_metrics" in st.session_state:
                m = st.session_state["lab_metrics"]
                cols = st.columns(5)
                cols[0].metric("总交易笔数", m["total_trades"])
                cols[1].metric("胜率", f"{m['win_rate']:.2%}")
                cols[2].metric("总盈亏", f"¥{m['total_pnl']:,.0f}")
                cols[3].metric("累计收益", f"{m['total_return']:.2%}")
                cols[4].metric("最大回撤", f"{m['max_drawdown']:.2%}")
                if st.session_state["lab_equity"]:
                    eq_frame = pd.DataFrame(st.session_state["lab_equity"], columns=["日期", "资产净值"])
                    st.line_chart(eq_frame, x="日期", y="资产净值", color="#36C98F")
                if st.session_state["lab_trades"]:
                    st.markdown("**本次回测逐笔交易**")
                    st.dataframe(localize_dataframe(pd.DataFrame(st.session_state["lab_trades"])), width="stretch", hide_index=True)

            st.markdown("**历史回测记录**")
            runs = app.database.query_all("SELECT * FROM signal_backtest_runs ORDER BY created_at DESC LIMIT 20")
            if runs:
                run_labels = {r["id"]: f"{r['strategy_name']}（{r['start_date']}~{r['end_date']}）" for r in runs}
                selected_run_id = st.selectbox("选择一次回测查看交易明细", list(run_labels), format_func=lambda rid: run_labels[rid], key="lab_run_sel")
                run_trades = app.database.query_all("SELECT * FROM signal_trades WHERE run_id=? ORDER BY id", (selected_run_id,))
                if run_trades:
                    st.dataframe(localize_dataframe(pd.DataFrame(run_trades)), width="stretch", hide_index=True)
                else:
                    st.caption("该次回测没有产生交易")

    # ── 盘中买点扫描 ──
    with lab_scan:
        st.caption("扫描「策略定义」中已勾选启用的策略，命中即推送邮件/企业微信。")
        if st.button("立即扫描买点", disabled=not is_admin):
            with st.spinner("正在拉取实时行情、公告与板块数据..."):
                try:
                    quotes = fetch_scan_quotes(app.data, app.database)
                    has_candidates, report, html = build_buy_report(app.data, app.database, current_quotes=quotes)
                    if has_candidates:
                        app.notifications.send("盘中买点扫描", report, "INFO", html=html)
                        st.success("发现买入候选，已发送邮件：")
                        st.text(report)
                    else:
                        st.info("当前无符合条件的买入候选（未发送邮件）")
                except Exception as error:
                    st.error(str(error))

    # ── 因子管理 ──
    with lab_factor:
        st.caption("因子公式基于日线 OHLCV 列 `open/high/low/close/volume/amount/pre_close`，可用 `shift/rolling/pct_change/ewm/diff/clip` 等 pandas 方法。")
        with st.form("create_factor", clear_on_submit=True):
            c1, c2 = st.columns([1, 3])
            new_factor_name = c1.text_input("因子名（英文/数字/下划线）")
            new_factor_expr = c2.text_input("公式表达式", placeholder="如 close/close.shift(5)-1")
            create_factor_clicked = st.form_submit_button("创建因子", disabled=not is_admin)
        if create_factor_clicked:
            if not new_factor_name or not new_factor_expr:
                st.error("请填写因子名与公式表达式")
            else:
                try:
                    evaluate_factor(new_factor_expr, _lab_sample_frame())
                    app.database.execute(
                        "INSERT INTO custom_factors(id,name,expression,created_at) VALUES(?,?,?,?)",
                        (uuid4().hex, new_factor_name, new_factor_expr, utc_now_text()),
                    )
                    st.success(f"因子 `{new_factor_name}` 已创建")
                except ValueError as error:
                    st.error(str(error))

        custom_factors = app.database.query_all("SELECT * FROM custom_factors ORDER BY created_at DESC")
        if custom_factors:
            st.markdown("**自定义因子**")
            st.dataframe(pd.DataFrame([{"因子名": f["name"], "公式": f["expression"]} for f in custom_factors]), width="stretch", hide_index=True)
            del_col, del_btn = st.columns([2, 1])
            del_name = del_col.selectbox("删除因子", [f["name"] for f in custom_factors], key="lab_del_factor")
            if del_btn.button("删除该因子", disabled=not is_admin):
                app.database.execute("DELETE FROM custom_factors WHERE name=?", (del_name,))
                st.rerun()

        with st.expander("内置因子库（可一键导入，共 24 个）"):
            st.dataframe(pd.DataFrame([{"因子名": k, "说明": v[0], "公式": v[1]} for k, v in BUILTIN_FACTOR_FORMULAS.items()]), width="stretch", hide_index=True)
            imp_col, imp_btn = st.columns([2, 1])
            import_name = imp_col.selectbox("选择内置因子", list(BUILTIN_FACTOR_FORMULAS), key="lab_import_factor")
            if imp_btn.button("导入为自定义因子", disabled=not is_admin):
                expr = BUILTIN_FACTOR_FORMULAS[import_name][1]
                app.database.execute(
                    "INSERT OR IGNORE INTO custom_factors(id,name,expression,created_at) VALUES(?,?,?,?)",
                    (uuid4().hex, import_name, expr, utc_now_text()),
                )
                st.success(f"已导入因子 `{import_name}`")

    # ── 策略定义 ──
    with lab_strategy:
        with st.expander("内置策略模板（可一键导入）"):
            st.dataframe(
                pd.DataFrame([{"策略": name, "说明": desc} for name, (desc, _e, _x) in BUILTIN_STRATEGY_TEMPLATES.items()]),
                width="stretch", hide_index=True,
            )
            tmpl_col, tmpl_btn = st.columns([2, 1])
            tmpl_name = tmpl_col.selectbox("选择策略模板", list(BUILTIN_STRATEGY_TEMPLATES), key="lab_import_strategy")
            if tmpl_btn.button("导入该策略", disabled=not is_admin):
                _desc, entry_cfg, exit_cfg = BUILTIN_STRATEGY_TEMPLATES[tmpl_name]
                app.database.execute(
                    """INSERT INTO signal_strategies(id,name,entry_json,exit_json,created_at) VALUES(?,?,?,?,?)
                       ON CONFLICT(name) DO UPDATE SET entry_json=excluded.entry_json, exit_json=excluded.exit_json, created_at=excluded.created_at""",
                    (uuid4().hex, tmpl_name, json.dumps(entry_cfg, ensure_ascii=False), json.dumps(exit_cfg, ensure_ascii=False), utc_now_text()),
                )
                st.success(f"策略 `{tmpl_name}` 已导入，可在「回测与记录」或「盘中买点扫描」使用")
        all_factor_names = [f["name"] for f in app.database.query_all("SELECT name FROM custom_factors")] + list(BUILTIN_FACTOR_FORMULAS)
        all_factor_names = list(dict.fromkeys(all_factor_names))
        cond_column_config = {
            "factor": st.column_config.SelectboxColumn("因子", options=all_factor_names, required=True),
            "op": st.column_config.SelectboxColumn("运算符", options=[">", "<", ">=", "<=", "==", "!="], required=True),
            "value": st.column_config.NumberColumn("阈值", required=True),
        }
        empty_conds = pd.DataFrame(columns=["factor", "op", "value"])
        strategy_name = st.text_input("策略名称", key="lab_strategy_name")
        st.markdown("**买入条件**（满足方式）")
        entry_combine = st.radio("买入满足方式", ["AND", "OR"], horizontal=True, key="lab_entry_combine")
        entry_df = st.data_editor(empty_conds, num_rows="dynamic", key="lab_entry_editor", column_config=cond_column_config, width="stretch")
        st.markdown("**离场条件**（满足方式）")
        exit_combine = st.radio("离场满足方式", ["AND", "OR"], horizontal=True, key="lab_exit_combine")
        exit_df = st.data_editor(empty_conds, num_rows="dynamic", key="lab_exit_editor", column_config=cond_column_config, width="stretch")
        sl_col, tp_col = st.columns(2)
        stop_loss_pct = sl_col.number_input("止损比例（%，0=不启用）", min_value=0.0, max_value=50.0, value=0.0, step=1.0, key="lab_stop_loss") / 100.0
        take_profit_pct = tp_col.number_input("止盈比例（%，0=不启用）", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="lab_take_profit") / 100.0
        tr_col, br_col = st.columns(2)
        trailing_stop_pct = tr_col.number_input("移动止损比例（%，0=不启用）", min_value=0.0, max_value=50.0, value=0.0, step=1.0, key="lab_trailing_stop") / 100.0
        breakout_exit = br_col.checkbox("跌破箱体高点即离场", value=False, key="lab_breakout_exit")

        if st.button("保存策略", disabled=not is_admin):
            if not strategy_name:
                st.error("请填写策略名称")
            else:
                entry_conds = _conditions_from_df(entry_df)
                exit_conds = _conditions_from_df(exit_df)
                if not entry_conds or not exit_conds:
                    st.error("买入和离场条件都至少需要一条")
                else:
                    entry_json = json.dumps({"combine": entry_combine, "conditions": entry_conds}, ensure_ascii=False)
                    exit_json = json.dumps({
                        "combine": exit_combine, "conditions": exit_conds,
                        "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
                        "trailing_stop_pct": trailing_stop_pct, "breakout_exit": breakout_exit,
                    }, ensure_ascii=False)
                    app.database.execute(
                        """INSERT INTO signal_strategies(id,name,entry_json,exit_json,created_at) VALUES(?,?,?,?,?)
                           ON CONFLICT(name) DO UPDATE SET entry_json=excluded.entry_json, exit_json=excluded.exit_json, created_at=excluded.created_at""",
                        (uuid4().hex, strategy_name, entry_json, exit_json, utc_now_text()),
                    )
                    st.success(f"策略 `{strategy_name}` 已保存")

        saved_strategies = app.database.query_all("SELECT name, enabled FROM signal_strategies ORDER BY created_at DESC")
        if saved_strategies:
            st.markdown("**已保存的策略（勾选 = 盘中扫描/推送邮箱，可删除）**")
            for strat in saved_strategies:
                c1, c2 = st.columns([4, 1])
                new_enabled = c1.checkbox(strat["name"], value=bool(strat["enabled"]), key=f"en_{strat['name']}", disabled=not is_admin)
                if is_admin and new_enabled != bool(strat["enabled"]):
                    app.database.execute("UPDATE signal_strategies SET enabled=? WHERE name=?", (int(new_enabled), strat["name"]))
                    st.rerun()
                if c2.button("删除", key=f"delbtn_{strat['name']}", disabled=not is_admin):
                    app.database.execute("DELETE FROM signal_strategies WHERE name=?", (strat["name"],))
                    st.rerun()

with overview:
    equity = frame("SELECT snapshot_date,total_equity,cash,market_value FROM account_snapshots WHERE account_id='paper' ORDER BY id")
    if equity.empty:
        st.info("暂无账户净值记录")
    else:
        for column in ["total_equity", "cash", "market_value"]:
            equity[column] = pd.to_numeric(equity[column], errors="coerce")
        latest = equity.iloc[-1]
        prev = equity.iloc[-2] if len(equity) > 1 else latest
        initial_cash = float(app.settings.trading.get("paper_initial_cash") or 0) or float(equity["total_equity"].iloc[0] or 1)
        prev_equity = float(prev["total_equity"] or 0)
        day_change = float(latest["total_equity"] or 0) - prev_equity
        day_change_pct = day_change / prev_equity if prev_equity > 0 else 0.0
        cumulative_return = float(latest["total_equity"] or 0) / initial_cash - 1
        equity_peak = equity["total_equity"].cummax()
        max_drawdown = float(((equity["total_equity"] - equity_peak) / equity_peak).min())
        position_ratio = float(latest["market_value"] or 0) / float(latest["total_equity"]) if latest["total_equity"] else 0.0
        cards = st.columns(5)
        cards[0].metric("最新总资产", f"¥{latest['total_equity']:,.2f}")
        cards[1].metric("较前一交易日", f"¥{day_change:+,.2f}", f"{day_change_pct:+.2%}", delta_color="inverse")
        cards[2].metric("累计收益率", f"{cumulative_return:+.2%}", f"¥{float(latest['total_equity']) - initial_cash:+,.2f}", delta_color="inverse")
        cards[3].metric("最大回撤", f"{max_drawdown:.2%}", help="统计区间内净值相对历史高点的最大跌幅")
        cards[4].metric("仓位占比", f"{position_ratio:.1%}", help="持仓市值 / 总资产")
        display_equity = localize_dataframe(equity)
        st.caption(
            f"统计区间 {display_equity['净值日期'].iloc[0]} 至 {display_equity['净值日期'].iloc[-1]}"
            f" · 共 {len(equity)} 个交易日 · 初始资金 ¥{initial_cash:,.0f}"
        )
        st.subheader("累计收益率（%）")
        returns_frame = pd.DataFrame({
            "净值日期": display_equity["净值日期"],
            "累计收益率": (equity["total_equity"] / initial_cash - 1) * 100,
        })
        st.line_chart(returns_frame, x="净值日期", y="累计收益率", color="#36C98F")
        st.subheader("资产构成")
        st.line_chart(display_equity, x="净值日期", y=["总资产", "可用现金", "持仓市值"], color=["#36C98F", "#8A99A6", "#E7B85C"])
with holdings:
    refresh_seconds = st.selectbox(
        "浮动盈亏实时刷新间隔（秒，0=暂停刷新）",
        [10, 5, 15, 30, 60, 0],
        key="holdings_refresh_seconds",
    )

    def _pnl_color(value: float) -> str:
        if value > 0:
            return "color: #ff6b6b"
        if value < 0:
            return "color: #36C98F"
        return ""

    def _pct_text(value) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{value * 100:+.2f}%"

    @st.fragment(run_every=None if refresh_seconds == 0 else int(refresh_seconds))
    def _live_holdings():
        rows = app.database.query_all(
            "SELECT code,name,quantity,sellable_quantity,avg_cost,latest_price "
            "FROM positions WHERE quantity>0 ORDER BY latest_price*quantity DESC"
        )
        quotes: dict[str, float] = {}
        live_ok = False
        if rows:
            try:
                quotes = sina_spot_prices([row["code"] for row in rows])
                live_ok = bool(quotes)
            except Exception:
                quotes = {}
        records = []
        for row in rows:
            price = float(quotes.get(row["code"]) or row["latest_price"] or 0)
            cost = float(row["avg_cost"] or 0)
            quantity = int(row["quantity"])
            records.append({
                "证券代码": row["code"],
                "证券名称": row["name"],
                "数量": quantity,
                "可卖数量": int(row["sellable_quantity"]),
                "持仓成本": cost,
                "最新价格": price,
                "持仓市值": quantity * price,
                "浮动盈亏": (price - cost) * quantity,
                "盈亏比例": price / cost - 1 if cost > 0 else None,
            })
        if not records:
            st.info("暂无持仓")
            return
        view = pd.DataFrame(records)
        total_pnl = float(view["浮动盈亏"].sum())
        total_color = "#ff6b6b" if total_pnl > 0 else "#36C98F" if total_pnl < 0 else "#e6edf3"
        source = "新浪实时行情" if live_ok else "数据库收盘价（实时行情暂不可用）"
        updated = pd.Timestamp.now().strftime("%H:%M:%S")
        st.markdown(
            f"<span style='font-size:0.92rem'>{source} · 上次更新 {updated} · "
            f"合计浮动盈亏 <b style='color:{total_color}'>¥{total_pnl:+,.2f}</b></span>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            view.style.format({
                "持仓成本": "{:.4f}",
                "最新价格": "{:.2f}",
                "持仓市值": "{:,.2f}",
                "浮动盈亏": "{:+,.2f}",
                "盈亏比例": _pct_text,
            }).map(_pnl_color, subset=["浮动盈亏", "盈亏比例"]),
            width="stretch",
            hide_index=True,
        )

    _live_holdings()
    position_codes = [
        row["code"] for row in app.database.query_all(
            "SELECT code FROM positions WHERE quantity>0 ORDER BY latest_price*quantity DESC"
        )
    ]
    if position_codes:
        close_code = st.selectbox("手动平仓标的", position_codes, disabled=not is_admin)
        if st.button("提交全量平仓委托", disabled=not is_admin):
            try:
                order_id = app.trading.queue_manual_close(close_code)
                st.success(f"全量平仓委托已提交，委托编号：{order_id}")
            except Exception as error:
                st.error(str(error))
with orders:
    st.subheader("委托记录")
    st.dataframe(localize_dataframe(frame("SELECT trade_date,code,name,side,quantity,fill_price,status,strategy,error,created_at FROM orders ORDER BY created_at DESC LIMIT 300")), width="stretch", hide_index=True)
    st.subheader("成交记录")
    st.dataframe(localize_dataframe(frame("SELECT filled_at,code,side,quantity,price,commission,stamp_duty FROM fills ORDER BY id DESC LIMIT 300")), width="stretch", hide_index=True)
with signals_tab:
    st.dataframe(localize_dataframe(frame("SELECT as_of_date,code,name,action,target_weight,score,strategy,status,reason FROM signals ORDER BY created_at DESC LIMIT 300")), width="stretch", hide_index=True)
with risk_tab:
    st.subheader("当前风控状态")
    st.dataframe(localize_dataframe(pd.DataFrame([risk_state])), width="stretch", hide_index=True)
    st.subheader("风险事件")
    st.dataframe(localize_dataframe(frame("SELECT event_time,level,category,code,message FROM risk_events ORDER BY id DESC LIMIT 200")), width="stretch", hide_index=True)
with backtest_tab:
    run = app.database.query_one("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1")
    if not run:
        st.info("暂无回测记录")
    else:
        cols = st.columns(5)
        cols[0].metric("期末权益", f"¥{run['final_equity']:,.2f}")
        cols[1].metric("年化收益", f"{run['annual_return']:.2%}")
        cols[2].metric("最大回撤", f"{run['max_drawdown']:.2%}")
        cols[3].metric("夏普", "-" if run["sharpe"] is None else f"{run['sharpe']:.2f}")
        cols[4].metric("胜率", f"{run['win_rate']:.2%}")
        curve = localize_dataframe(frame("SELECT trade_date,equity FROM backtest_equity WHERE run_id=? ORDER BY trade_date", (run["id"],)))
        st.line_chart(curve, x="交易日期", y="资产净值", color="#36C98F")
with settings_tab:
    if is_admin:
        r1, r2 = st.columns(2)
        if r1.button("重载配置", key="reload_config", help="重新读取 config/default.yaml 与 .env，无需重启看板"):
            st.cache_resource.clear()
            st.rerun()
        if r2.button("退出登录", key="logout"):
            st.session_state.pop("_admin_ok", None)
            st.rerun()
    st.dataframe(pd.DataFrame([
        {"项目": "运行模式", "值": label_value(app.settings.mode, "mode")},
        {"项目": "默认策略", "值": STRATEGY_LABELS.get(selected_strategy, selected_strategy)},
        {"项目": "数据主源", "值": app.settings.data["primary"]},
        {"项目": "Tushare", "值": "已配置" if app.settings.tushare_token else "未配置"},
        {"项目": "企业微信", "值": "已配置" if app.settings.wecom_webhook else "未配置"},
        {"项目": "邮件", "值": "已配置" if app.settings.smtp_host else "未配置"},
    ]), width="stretch", hide_index=True)
