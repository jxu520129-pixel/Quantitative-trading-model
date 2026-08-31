"""主升浪评分：事件分 + 宏观分 + 产业链分 + 盈利分 + 技术分 加权合成。

MVP 版：事件/宏观/产业链用本地数据库 + 关键词匹配，盈利用 AkShare 财务摘要，
技术分由调用方从现有 OHLCV 因子传入。
"""

from __future__ import annotations

import html as _html
import logging
from typing import Any

import pandas as pd

from .database import Database
from .event_similarity import format_similarity_section_html, similar_event_review
from .industry_chain import INDUSTRY_CHAIN
from .ml_model import model_info, predict_main_rally_probability


LOG = logging.getLogger(__name__)

# 评分权重（可按样本外回测调整）
WEIGHTS: dict[str, float] = {
    "event": 0.20, "macro": 0.15, "industry": 0.20, "earnings": 0.20, "technical": 0.25,
}


def _num(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def latest_macro(database: Database, indicator: str) -> float | None:
    row = database.query_one(
        "SELECT value FROM macro_indicators WHERE indicator=? ORDER BY period DESC LIMIT 1", (indicator,)
    )
    return float(row["value"]) if row else None


def macro_regime_score(database: Database) -> float:
    """宏观环境评分（0-100）：增长、通胀、流动性。"""
    gdp = latest_macro(database, "gdp_yoy")
    m2 = latest_macro(database, "m2_yoy")
    cpi = latest_macro(database, "cpi_yoy")
    score = 50.0
    if gdp is not None:
        score += (gdp - 5.0) * 5.0
    if m2 is not None:
        score += (m2 - 8.0) * 3.0
    if cpi is not None:
        score -= max(0.0, cpi - 3.0) * 10.0
    return _clip(score)


def recent_events(database: Database, limit: int = 200) -> list[dict[str, Any]]:
    return database.query_all(
        "SELECT title, event_type FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
    )


_PERSISTENCE_WEIGHT: dict[str, float] = {"短期": 1.0, "中期": 2.0, "长期": 3.0}


def event_industry_score(database: Database, industry: str) -> tuple[float, float]:
    """事件分 + 产业链分：行业命中近期事件的规模×方向×持续期（50 为中性）。"""
    stock_chains = [name for name in INDUSTRY_CHAIN if name in industry]
    if not stock_chains:
        return 50.0, 50.0
    events = database.query_all(
        "SELECT title, magnitude, causal_direction, affected_industries, persistence FROM events ORDER BY created_at DESC LIMIT 200"
    )
    event_score = 0.0
    chain_score = 0.0
    for ev in events:
        title = ev["title"] or ""
        sign = 1.0 if ev["causal_direction"] == "正" else (-1.0 if ev["causal_direction"] == "负" else 0.0)
        mag = float(ev["magnitude"] or 1)
        pw = _PERSISTENCE_WEIGHT.get(ev.get("persistence") or "短期", 1.0)
        affected = (ev["affected_industries"] or "").split(",")
        for chain in stock_chains:
            info = INDUSTRY_CHAIN[chain]
            if any(kw in title for kw in info["keywords"]) or chain in affected:
                event_score += mag * sign * pw
                break
            if any(up in affected for up in info["upstream"]) or any(down in affected for down in info["downstream"]):
                chain_score += mag * sign * pw
                break
    return _clip(50 + event_score * 3.0), _clip(50 + chain_score * 3.0)


def fetch_earnings(code: str) -> tuple[float, float]:
    """获取个股财务摘要，返回 (营收同比%, 净利同比%)。AkShare 失败时回退 Tushare 财务指标。"""
    import akshare as ak

    frame = None
    try:
        frame = ak.stock_financial_abstract_ths(symbol=code)
    except Exception as error:
        LOG.warning("个股财务摘要获取失败 %s：%s", code, error)
    if frame is not None and not frame.empty:
        latest = frame.iloc[-1]  # 数据按报告期升序，最后一行为最新一期
        return _num(latest.get("营业总收入同比增长率")), _num(latest.get("净利润同比增长率"))
    return _earnings_from_tushare(code)


def _earnings_from_tushare(code: str) -> tuple[float, float]:
    """Tushare fina_indicator 回退（第三方代理套餐含官方 2000 积分级财务指标）。"""
    from .data.providers import tushare_pro_from_env

    pro = tushare_pro_from_env()
    if pro is None:
        return 0.0, 0.0
    suffix = ".SH" if code.startswith(("5", "6", "9")) else ".SZ"
    try:
        frame = pro.fina_indicator(ts_code=f"{code}{suffix}", fields="ts_code,end_date,or_yoy,netprofit_yoy")
    except Exception as error:
        LOG.warning("Tushare 财务指标获取失败 %s：%s", code, error)
        return 0.0, 0.0
    if frame is None or frame.empty:
        return 0.0, 0.0
    frame = frame.sort_values("end_date", ascending=False)
    latest = frame.iloc[0]
    return _num(latest.get("or_yoy")), _num(latest.get("netprofit_yoy"))


def earnings_score(revenue_yoy: float, profit_yoy: float) -> float:
    """盈利预期评分（0-100）：营收/净利增速。"""
    score = 50.0 + revenue_yoy * 2.0 + profit_yoy
    return _clip(score)


def main_rally_score(
    event_score: float,
    macro_score: float,
    industry_score: float,
    earnings_score: float,
    technical_score: float,
) -> float:
    """加权合成主升浪评分（0-100）。"""
    return (
        WEIGHTS["event"] * event_score
        + WEIGHTS["macro"] * macro_score
        + WEIGHTS["industry"] * industry_score
        + WEIGHTS["earnings"] * earnings_score
        + WEIGHTS["technical"] * technical_score
    )


# 技术方案 §11.2 选股门槛（V1 示例）：全部通过视为「门槛达标」候选
THRESHOLDS: list[tuple[str, str, float]] = [
    ("event", "事件分", 70.0),
    ("macro", "宏观环境分", 50.0),
    ("industry_chain", "产业链分", 70.0),
    ("earnings", "盈利预期分", 60.0),
    ("technical", "趋势分", 60.0),
]


def threshold_checks(r: dict[str, Any]) -> list[tuple[str, float, float, bool]]:
    """返回 [(分项名, 实际分, 门槛, 是否通过), ...]。"""
    return [(label, float(r[key]), th, float(r[key]) >= th) for key, label, th in THRESHOLDS]


def passes_all_thresholds(r: dict[str, Any]) -> bool:
    return all(ok for _label, _value, _th, ok in threshold_checks(r))


def get_stock_industry(database: Database, code: str) -> str:
    row = database.query_one("SELECT industry FROM stock_industry WHERE code=?", (code,))
    return str(row["industry"]) if row else ""


def technical_score(factors: dict[str, float]) -> float:
    """技术分（0-100）：突破强度 + 量比 + 趋势 + 均线多头。"""
    score = 0.0
    score += _clip(float(factors.get("breakout_20", 0.0)) * 800, 0, 40)
    score += _clip((float(factors.get("vol_ratio", 0.0)) - 1.5) * 12.5, 0, 25)
    score += _clip(float(factors.get("ma20_dev", 0.0)) * 133, 0, 20)
    score += _clip((float(factors.get("ma_trend", 0.0)) - 0.5) * 30, 0, 15)
    return _clip(score)


def score_main_rally_candidates(runtime: Any) -> list[dict[str, Any]]:
    """对「主升浪·启动突破」候选打分排序，返回完整评分明细。"""
    from .lab import fetch_industries, find_buy_candidates

    from .ml_model import feature_vector_from_frame

    sc = runtime.settings.raw.get("scan", {})
    max_price = float(sc.get("max_price", 0) or 0) or None
    min_price = float(sc.get("min_price", 0) or 0) or None
    min_amt = float(sc.get("min_average_amount", 0) or 0) or None
    candidates = find_buy_candidates(
        runtime.data, runtime.database, max_price=max_price, min_price=min_price,
        min_average_amount=min_amt, strategy_names=["主升浪·启动突破"],
    )
    # 按需补全候选的行业数据（避免逐个全市场拉取）
    codes = [m["code"] for item in candidates for m in item["matches"]]
    if codes:
        fetch_industries(runtime.database, codes)
    bars_by_code = runtime.data.load_bars_many(sorted(set(codes))) if codes else {}
    macro = macro_regime_score(runtime.database)
    results: list[dict[str, Any]] = []
    for item in candidates:
        for m in item["matches"]:
            code = m["code"]
            industry = get_stock_industry(runtime.database, code)
            ev, ind = event_industry_score(runtime.database, industry)
            rev, prof = fetch_earnings(code)
            earn = earnings_score(rev, prof)
            tech = technical_score(m.get("factors", {}))
            score = main_rally_score(ev, macro, ind, earn, tech)
            frame = bars_by_code.get(code)
            ml_features = feature_vector_from_frame(frame) if frame is not None else None
            results.append({
                "code": code, "name": m["name"], "price": m["price"], "industry": industry,
                "score": round(score, 1), "event": round(ev, 1), "macro": round(macro, 1),
                "industry_chain": round(ind, 1), "earnings": round(earn, 1), "technical": round(tech, 1),
                "revenue_yoy": rev, "profit_yoy": prof,
                "ml_features": ml_features,
                "vwap_dev": None if ml_features is None else round(ml_features["vwap_dev"], 4),
                "amount_z": None if ml_features is None else round(ml_features["amount_z"], 2),
                "vol_ratio": None if ml_features is None else round(ml_features["vol_ratio"], 2),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_industry_events(database: Database, industry: str) -> list[dict[str, Any]]:
    """获取与股票行业相关的事件（按规模降序）。"""
    stock_chains = [name for name in INDUSTRY_CHAIN if name in industry]
    if not stock_chains:
        return []
    events = database.query_all(
        "SELECT title, magnitude, causal_direction, persistence, affected_industries FROM events ORDER BY created_at DESC LIMIT 300"
    )
    matched = []
    for ev in events:
        affected = (ev["affected_industries"] or "").split(",")
        title = ev["title"] or ""
        if any(chain in affected or chain in title for chain in stock_chains):
            matched.append(ev)
    matched.sort(key=lambda e: e["magnitude"], reverse=True)
    return matched


def get_stage(technical: float) -> str:
    if technical >= 80:
        return "主升 → 加速"
    if technical >= 60:
        return "启动 → 主升转换"
    if technical >= 40:
        return "启动"
    return "潜伏"


def get_chain_desc(industry: str) -> str:
    parts = []
    for chain in (name for name in INDUSTRY_CHAIN if name in industry):
        info = INDUSTRY_CHAIN[chain]
        if info["upstream"]:
            parts.append(f"上游 {'/'.join(info['upstream'][:2])} ↑")
        parts.append(f"{chain} ↑↑")
        if info["downstream"]:
            parts.append(f"下游 {'/'.join(info['downstream'][:2])} ↑↑")
    return " / ".join(parts) if parts else (industry[:12] if industry else "未分类")


def main_rally_probability(settings: Any, r: dict[str, Any]) -> tuple[float, str]:
    """P(未来20交易日进入主升)：优先模型输出，回退评分近似映射。"""
    prob = predict_main_rally_probability(settings, r.get("ml_features")) if settings else None
    if prob is not None:
        return prob, "模型"
    return _clip(r["score"] - 10, 0, 85) / 100.0, "近似"


def _fund_flow_text(r: dict[str, Any]) -> str:
    if r.get("vwap_dev") is None:
        return "VWAP偏离 - / 量比 - / 成交额z -"
    return (f"VWAP偏离 {r['vwap_dev']:+.2%} / 量比 {r['vol_ratio']:.2f} / 成交额z {r['amount_z']:+.1f}")


def format_main_rally_report(database: Database, r: dict[str, Any], settings: Any = None) -> str:
    """单只股票的主升浪富文本报告（文档「最终输出样例」样式）。"""
    events = get_industry_events(database, r["industry"])
    pos = [e for e in events if e["causal_direction"] == "正"]
    neg = [e for e in events if e["causal_direction"] == "负"]
    macro_label = "扩张" if r["macro"] >= 60 else ("中性" if r["macro"] >= 40 else "收缩")
    prob, source = main_rally_probability(settings, r)
    lines = [
        f"股票：{r['code']} {r['name']}",
        f"主升浪评分：{r['score']}",
        f"未来20交易日进入主升阶段概率：{prob:.0%}（{source}）",
        "",
    ]
    if events:
        top = events[0]
        lines.append(f"事件：{top['title'][:40]}")
        lines.append(f"事件强度：{top['magnitude']}  事件持续性：{top['persistence']}")
    else:
        lines.append("事件：无显著行业事件")
        lines.append("事件强度：-  事件持续性：-")
    lines += [
        "",
        f"宏观环境：{macro_label}（{r['macro']:.0f}）",
        f"产业链分：{r['industry_chain']:.0f}（门槛 70）",
        "",
        f"产业链：{get_chain_desc(r['industry'])}",
        f"盈利预期：{r['earnings']:.0f}（营收同比{r['revenue_yoy']}% / 净利同比{r['profit_yoy']}%）",
        f"趋势：{r['technical']:.0f}  拥挤度：{max(0, int(r['event'] - 50))}",
        f"资金行为：{_fund_flow_text(r)}",
        f"当前阶段：{get_stage(r['technical'])}",
        "",
        f"主要催化剂：{' / '.join(e['title'][:20] for e in pos[:2]) if pos else '-'}",
        f"主要风险：{' / '.join(e['title'][:20] for e in neg[:2]) if neg else '-'}",
    ]
    return "\n".join(lines)


def _score_bar(score: float, color: str) -> str:
    width = max(0, min(100, int(score)))
    return (
        f'<div style="background:#e9ecef;border-radius:4px;height:8px;width:110px;display:inline-block;vertical-align:middle;">'
        f'<div style="background:{color};height:8px;border-radius:4px;width:{width}%;"></div></div>'
    )


def _candidate_card(database: Database, r: dict[str, Any], settings: Any = None) -> str:
    """单只候选的完整卡片（技术方案「最终输出样例」字段）。"""
    esc = _html.escape
    events = get_industry_events(database, r["industry"])
    pos = [e for e in events if e["causal_direction"] == "正"]
    neg = [e for e in events if e["causal_direction"] == "负"]
    top = events[0] if events else None
    macro_label = "扩张" if r["macro"] >= 60 else ("中性" if r["macro"] >= 40 else "收缩")
    stage = get_stage(r["technical"])
    crowd = max(0, int(r["event"] - 50))
    prob, prob_source = main_rally_probability(settings, r)
    checks = threshold_checks(r)
    gate_html = "　".join(
        f'<span style="display:inline-block;padding:1px 7px;margin:1px 2px;border-radius:9px;font-size:11px;'
        f'color:{"#fff" if ok else "#fff"};background:{"#27ae60" if ok else "#b0b6bd"};">'
        f'{esc(label)} {value:.0f}/{th:.0f}</span>'
        for label, value, th, ok in checks
    )
    parts: list[str] = []
    parts.append('<div style="border:1px solid #dfe3e8;border-radius:8px;margin:10px 0;overflow:hidden;">')
    parts.append('<div style="background:#f4f6f8;padding:8px 14px;font-size:14px;color:#2c3e50;">'
                 f'<b style="font-size:15px;">{esc(r["name"])}</b>　{r["code"]}　'
                 f'<span style="color:#d35400;font-weight:700;">¥{r["price"]:.2f}</span>　'
                 f'<span style="color:#666;font-size:12px;">{esc(r["industry"] or "未分类")}</span>'
                 f'<span style="float:right;">主升浪评分 <b style="font-size:17px;color:#e67e22;">{r["score"]:.1f}</b>'
                 f'　进入主升概率 <b style="color:#e67e22;">{prob:.0%}</b>'
                 f'<span style="color:#999;font-size:11px;">（{prob_source}）</span></span></div>')
    parts.append('<div style="padding:9px 14px;">')
    parts.append(f'<div style="margin-bottom:6px;">{gate_html}</div>')
    parts.append('<table border="0" cellspacing="0" cellpadding="3" style="font-size:12.5px;color:#333;width:100%;">')
    event_row = (
        f'{esc(str(top["title"])[:44])}　强度 {top["magnitude"]}　持续性 {esc(str(top["persistence"]))}'
        if top else '无显著行业事件'
    )
    parts.append(f'<tr><td style="color:#888;width:76px;vertical-align:top;">事件</td><td>{event_row}</td></tr>')
    parts.append(f'<tr><td style="color:#888;vertical-align:top;">宏观环境</td><td>{macro_label}（{r["macro"]:.0f}，门槛 50）</td></tr>')
    parts.append('<tr><td style="color:#888;vertical-align:top;">产业链</td>'
                 f'<td>{esc(get_chain_desc(r["industry"]))}　{_score_bar(r["industry_chain"], "#3498db")} {r["industry_chain"]:.0f}</td></tr>')
    parts.append(f'<tr><td style="color:#888;vertical-align:top;">盈利预期</td><td>{r["earnings"]:.0f}'
                 f'　{_score_bar(r["earnings"], "#27ae60")}（营收同比 {r["revenue_yoy"]}% / 净利同比 {r["profit_yoy"]}%）</td></tr>')
    parts.append('<tr><td style="color:#888;vertical-align:top;">趋势</td>'
                 f'<td>{r["technical"]:.0f}　{_score_bar(r["technical"], "#e67e22")}　当前阶段：<b>{esc(stage)}</b>'
                 f'　拥挤度（近似）：{crowd}</td></tr>')
    parts.append(f'<tr><td style="color:#888;vertical-align:top;">资金行为</td><td>{esc(_fund_flow_text(r))}</td></tr>')
    catalysts = " / ".join(str(e["title"])[:22] for e in pos[:2]) if pos else "-"
    risks = " / ".join(str(e["title"])[:22] for e in neg[:2]) if neg else "-"
    parts.append(f'<tr><td style="color:#888;vertical-align:top;">主要催化剂</td><td style="color:#c0392b;">{esc(catalysts)}</td></tr>')
    parts.append(f'<tr><td style="color:#888;vertical-align:top;">主要风险</td><td style="color:#1e8449;">{esc(risks)}</td></tr>')
    parts.append('</table></div></div>')
    return "".join(parts)


def format_main_rally_report_html(database: Database, results: list[dict[str, Any]], settings: Any = None, top_n: int = 10) -> str:
    """生成主升浪盘前 HTML 富文本报告（技术方案 §17.2 五类输出 + §11.2 门槛 + §20 样例）。"""
    if not results:
        return "<p>今日无主升浪候选</p>"
    esc = _html.escape
    passed = [r for r in results if passes_all_thresholds(r)]
    macro = results[0]["macro"]
    macro_label = "扩张" if macro >= 60 else ("中性" if macro >= 40 else "收缩")
    event_driven = sorted((r for r in results if r["event"] >= 70), key=lambda x: x["event"], reverse=True)[:20]
    chain_core = sorted((r for r in results if r["industry_chain"] >= 70), key=lambda x: x["industry_chain"], reverse=True)[:10]
    early_stage = sorted((r for r in results if r["technical"] < 60), key=lambda x: x["technical"], reverse=True)[:10]
    risk_watch: list[dict[str, Any]] = []
    for r in results:
        neg = [e for e in get_industry_events(database, r["industry"]) if e["causal_direction"] == "负"]
        if neg:
            risk_watch.append({**r, "_risk": neg[0]["title"]})
    risk_watch = risk_watch[:10]

    parts: list[str] = []
    parts.append('<div style="font-family:Microsoft YaHei,Arial,sans-serif;max-width:860px;">')
    parts.append('<div style="background:#34495e;color:#fff;padding:12px 16px;border-radius:6px;">')
    parts.append(f'<h3 style="margin:0;font-size:17px;">A股主升浪盘前扫描 · {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</h3>')
    parts.append(f'<p style="margin:5px 0 0;font-size:13px;opacity:.92;">'
                 f'候选总数 {len(results)}（「主升浪·启动突破」策略命中）　'
                 f'门槛达标 {len(passed)} 只　宏观环境：{macro_label}（{macro:.0f}）　'
                 f'评分权重：事件{WEIGHTS["event"]:.0%} / 宏观{WEIGHTS["macro"]:.0%} / 产业链{WEIGHTS["industry"]:.0%} '
                 f'/ 盈利{WEIGHTS["earnings"]:.0%} / 技术{WEIGHTS["technical"]:.0%}</p>')
    parts.append(f'<p style="margin:5px 0 0;font-size:12px;opacity:.75;">P(主升)模型：{esc(model_info(settings) if settings else "未训练（使用评分近似）")}</p>')
    parts.append('</div>')

    parts.append('<h4 style="margin:18px 0 4px;padding-left:9px;border-left:4px solid #e67e22;font-size:15px;color:#2c3e50;">'
                 f'一、主升浪候选 Top {min(top_n, len(results))}（按综合评分，完整样例卡片）</h4>')
    for r in results[:top_n]:
        parts.append(_candidate_card(database, r, settings))

    def _section(title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str, Any]], note: str = "") -> None:
        if not rows:
            return
        parts.append(f'<h4 style="margin:18px 0 4px;padding-left:9px;border-left:4px solid #3498db;font-size:15px;color:#2c3e50;">'
                     f'{esc(title)}（{len(rows)}）</h4>')
        if note:
            parts.append(f'<p style="margin:2px 0 6px;font-size:12px;color:#888;">{esc(note)}</p>')
        head = "".join(f'<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">{esc(h)}</th>' for h, _k, _f in columns)
        body = []
        for i, r in enumerate(rows, 1):
            bg = ' style="background:#f8f9fa;"' if i % 2 == 0 else ''
            cells = "".join(f'<td style="padding:5px 9px;">{f(r.get(k, "-"))}</td>' for _h, k, f in columns)
            body.append(f'<tr{bg}><td style="padding:5px 9px;color:#999;">{i}</td>{cells}</tr>')
        parts.append('<table border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;width:100%;font-size:12.5px;">'
                     f'<tr style="background:#34495e;"><th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">#</th>{head}</tr>'
                     + "".join(body) + "</table>")

    _section("二、高置信事件驱动（事件分 ≥ 70，Top 20）", event_driven,
             [("代码", "code", esc), ("名称", "name", esc), ("事件分", "event", lambda v: f"{v:.0f}"),
              ("评分", "score", lambda v: f"<b style='color:#e67e22;'>{v:.1f}</b>"), ("行业", "industry", esc)],
             "对应技术方案 §17.2「Top 20 高置信事件驱动」：事件强度高、方向为正的行业事件直接命中。")
    _section("三、产业链核心（产业链分 ≥ 70，Top 10）", chain_core,
             [("代码", "code", esc), ("名称", "name", esc), ("产业链分", "industry_chain", lambda v: f"{v:.0f}"),
              ("评分", "score", lambda v: f"<b style='color:#e67e22;'>{v:.1f}</b>"), ("行业", "industry", esc)],
             "对应「Top 10 产业链核心」：处于事件传导链上的行业中上游标的。")
    _section("四、潜伏 / 启动阶段关注（趋势分 < 60，Top 10）", early_stage,
             [("代码", "code", esc), ("名称", "name", esc), ("趋势分", "technical", lambda v: f"{v:.0f}"),
              ("阶段", "technical", lambda v: esc(get_stage(float(v)))), ("评分", "score", lambda v: f"{v:.1f}")],
             "对应「Top 10 潜伏/主升阶段」：事件与基本面达标但趋势尚未确认，适合跟踪而非立即介入。")
    _section("五、风险提示（命中负面事件，Top 10）", risk_watch,
             [("代码", "code", esc), ("名称", "name", esc), ("风险事件", "_risk", esc),
              ("评分", "score", lambda v: f"{v:.1f}"), ("行业", "industry", esc)],
             "对应「Top 10 退潮风险」：行业命中减持/处罚/利空等负面事件，谨慎对待。")

    seed_titles: list[str] = []
    for r in results[:top_n]:
        events = get_industry_events(database, r["industry"])
        if events and events[0]["title"] not in seed_titles:
            seed_titles.append(str(events[0]["title"]))
    reviews = similar_event_review(database, seed_titles) if seed_titles else []
    similarity_html = format_similarity_section_html(reviews, esc)
    if similarity_html:
        parts.append(similarity_html)

    parts.append('<p style="margin:16px 0 0;color:#999;font-size:12px;">'
                 '评分为规则近似（事件/宏观/产业链/盈利/技术加权）；P(主升)优先由模型输出，'
                 '未训练时为评分近似映射；相似事件收益为历史等权组合回溯，不代表未来表现。'
                 '系统自动扫描，仅供研究参考，不构成投资建议。</p>')
    parts.append('</div>')
    return "".join(parts)


def generate_main_rally_report(runtime: Any) -> tuple[bool, str, str]:
    """生成主升浪候选报告，返回 (是否有候选, 纯文本报告, HTML 富文本报告)。"""
    results = score_main_rally_candidates(runtime)
    if not results:
        return False, "今日无主升浪候选", "<p>今日无主升浪候选</p>"
    sections = [format_main_rally_report(runtime.database, r, runtime.settings) for r in results[:10]]
    header = "A股主升浪扫描 · 候选 Top " + str(len(sections))
    text = header + "\n" + "=" * 40 + "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(sections)
    html = format_main_rally_report_html(runtime.database, results, runtime.settings)
    return True, text, html
