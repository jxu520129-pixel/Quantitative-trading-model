"""主升浪评分：事件分 + 宏观分 + 产业链分 + 盈利分 + 技术分 加权合成。

MVP 版：事件/宏观/产业链用本地数据库 + 关键词匹配，盈利用 AkShare 财务摘要，
技术分由调用方从现有 OHLCV 因子传入。
"""

from __future__ import annotations

import html as _html
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .database import Database
from .event_similarity import format_similarity_section_html, similar_event_review
from .industry_chain import (
    INDUSTRY_CHAIN,
    TUSHARE_TO_CHAIN,
    affected_industries,
    direct_industries,
    macro_industries,
)
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


def score_main_rally_candidates(runtime: Any, current_quotes: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """对「主升浪·启动突破」候选打分排序，返回完整评分明细。"""
    from .lab import fetch_industries, find_buy_candidates

    from .ml_model import feature_vector_from_frame

    sc = runtime.settings.raw.get("scan", {})
    max_price = float(sc.get("max_price", 0) or 0) or None
    min_price = float(sc.get("min_price", 0) or 0) or None
    min_amt = float(sc.get("min_average_amount", 0) or 0) or None
    candidates = find_buy_candidates(
        runtime.data, runtime.database, current_quotes=current_quotes,
        max_price=max_price, min_price=min_price,
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


# 事件方向 -> 展示颜色（A 股涨红跌绿）
_DIR_COLOR = {"正": "#c0392b", "负": "#1e8449", "中性": "#7f8c8d"}


def top_events_today(database: Database, limit: int = 15) -> list[dict[str, Any]]:
    """当日重要事件（按重要性 magnitude 降序，同强度按时间倒序）。

    ``event_time`` 存的是北京时间字符串（YYYY-MM-DD HH:MM:SS），用北京时间判断
    「当日」；当日无事件时回退到最近入库的事件，保证盘前报告不空。
    """
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    fields = "title, summary, event_time, event_type, magnitude, causal_direction, affected_industries, persistence"
    events = database.query_all(
        f"SELECT {fields} FROM events WHERE event_time LIKE ? ORDER BY magnitude DESC, event_time DESC LIMIT ?",
        (f"{today}%", limit),
    )
    if not events:
        events = database.query_all(
            f"SELECT {fields} FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    return events


def match_event_candidates(
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
    limit: int = 5,
    exact: bool = True,
    industries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """找出与某事件相关的候选股票（按综合评分降序）。

    匹配依据：事件标题 + 摘要命中产业链关键词 → 映射到标准行业，候选股票所属行业
    命中即匹配（候选池股票先经 ``TUSHARE_TO_CHAIN`` 把 Tushare 行业名桥接成标准行业）。

    ``exact=True``（默认）只用**直接命中**的行业，避免上下游传导让几乎所有科技类事件
    都带上「半导体」，从而匹配到同一批个股；``exact=False`` 回退到含上下游的宽松匹配。
    ``industries`` 显式给定命中行业时优先级最高，用于宏观兜底（降准 → 银行/地产/券商）
    这类事件本身没有行业词、但需要按指定板块取股的情形。
    """
    title = event.get("title") or ""
    text = (title + " " + (event.get("summary") or "")).strip()
    if not text:
        text = title + " " + (event.get("affected_industries") or "")
    if industries is not None:
        hit_industries = set(industries)
    elif exact:
        hit_industries = set(direct_industries(text))
    else:
        affected = {a.strip() for a in (event.get("affected_industries") or "").split(",") if a.strip()}
        hit_industries = set(affected_industries(text)) | affected
    if not hit_industries:
        # 文本自身没有任何行业词时，退回事件**分类**（event_type，如「医药」「政策」）。
        # 例：「AI制药正式进入临床验证阶段」早期不含「医药」关键词，靠分类兜底才不至于
        # 落到空匹配、进而被后备池的无关行业（有色金属）顶替。
        hit_industries = set(direct_industries(str(event.get("event_type") or "")))
    if not hit_industries:
        return []
    matched: list[dict[str, Any]] = []
    for c in candidates:
        tushare_industry = c.get("industry") or ""
        # 优先用 Tushare 行业映射到标准行业（解决 stock_industry 存的是 Tushare 原始名
        # 而 INDUSTRY_CHAIN 是 17 个标准 key 之间的不匹配问题）
        chain = TUSHARE_TO_CHAIN.get(tushare_industry)
        hit = False
        if chain and chain in hit_industries:
            hit = True
        else:
            # 退化：直接在 Tushare 行业名里包含 INDUSTRY_CHAIN 的 key
            for k in INDUSTRY_CHAIN:
                if k in tushare_industry and k in hit_industries:
                    hit = True
                    break
        if hit:
            matched.append(c)
    matched.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    return matched[:limit]


def _event_stars(magnitude: Any) -> str:
    mag = max(1, min(5, int(magnitude or 1)))
    return "★" * mag + "☆" * (5 - mag)


def format_daily_events_html(
    database: Database,
    candidates: list[dict[str, Any]],
    limit: int = 12,
    standalone: bool = False,
) -> str:
    """生成「今日重要事件」HTML：核心机会 + 利空风险提醒，每个事件关联核心候选股票。

    只展示当日最新且重要性 ≥3 的核心消息；按方向拆成「核心机会（正/中性）」与
    「风险提醒（负面/利空）」两个区，每条的关联标的只保留评分最高的核心股票。

    ``standalone=True`` 时带完整容器（用于无主升浪候选时单独发「今日重要事件」邮件）；
    否则只返回板块片段（嵌入主升浪报告的头部）。
    """
    esc = _html.escape
    events = top_events_today(database, limit=limit * 3)
    if not events:
        return ""

    # 核心事件：当日 + 重要性 ≥3
    core = [e for e in events if int(e.get("magnitude") or 1) >= 3]
    if not core:
        core = events[:limit]
    opportunities = [e for e in core if str(e.get("causal_direction") or "") != "负"]
    risks = [e for e in core if str(e.get("causal_direction") or "") == "负"]

    # 事件专属后备池：主升浪候选池（candidates）为空、或某事件在候选池无匹配时，
    # 按**该事件自己的直接命中行业**查数据库活跃股票（每行业分别限额），作为该事件专属池。
    #
    # 注意：这里刻意「按事件单独建池」，而非把所有事件的行业合并成一个全局池。
    # 旧实现把所有事件行业求并集后 `ORDER BY close DESC LIMIT 80`，等价于取全市场最贵的
    # 80 只股票；再叠加 affected_industries 的上下游传导（半导体↔计算机↔通信↔化工↔电力设备
    # 高度连通，几乎每个科技事件都会带出「半导体」），于是不同事件全部匹配到同一批高价股。
    _name_cache: dict[str, list[tuple[str, str]]] = {}

    def _event_text(ev: dict[str, Any]) -> str:
        text = ((ev.get("title") or "") + " " + (ev.get("summary") or "")).strip()
        return text or ((ev.get("title") or "") + " " + (ev.get("affected_industries") or ""))

    def _event_direct_industries(ev: dict[str, Any]) -> list[str]:
        """事件直接命中的行业；文本里没有任何行业词时，退回事件分类（event_type）。

        例：「AI制药正式进入临床验证阶段」在补词前不含「医药」关键词，仅靠文本会得到空集，
        于是被后备池里恰好命中的无关行业（有色金属）顶替。分类兜底可避免这类静默错配。
        """
        return direct_industries(_event_text(ev)) or direct_industries(str(ev.get("event_type") or ""))

    def _event_risk_industries(ev: dict[str, Any], limit: int = 6) -> list[str]:
        """利空事件「受影响的敏感板块」：直接命中行业 + 宏观承压板块（如 美联储/加息 → 银行/保险/有色）。

        **刻意不落到个股**：把利空事件关联到具体股票，读者会误读成买入建议；而宏观利空
        （加息/制裁/关税）的板块影响本就是方向性、非公司性的，用「行业成交额代表股」去
        承载它必然错位——曾把「美联储加息概率升至72.4%」关联到厦门钨业/北方稀土/盛和资源。
        """
        text = _event_text(ev)
        merged = (
            direct_industries(text) or direct_industries(str(ev.get("event_type") or ""))
        ) + macro_industries(text)
        return list(dict.fromkeys(merged))[:limit]

    def _stock_names() -> list[tuple[str, str]]:
        """全市场股票简称（≥3 字、排除 ST/退市），用于事件文本里的公司名直配。"""
        if "v" not in _name_cache:
            try:
                rows = database.query_all(
                    "SELECT code, name FROM stock_basic "
                    "WHERE security_type='STOCK' AND is_st=0 AND is_delisted=0 AND length(name)>=3"
                )
                _name_cache["v"] = [(r["code"], r["name"]) for r in rows]
            except Exception as error:  # noqa: BLE001
                LOG.warning("查询股票名称索引失败：%s", error)
                _name_cache["v"] = []
        return _name_cache["v"]

    def _query_industry_pool(industries: list[str], per_industry: int = 12) -> list[dict[str, Any]]:
        """按标准行业查活跃股票，**跨行业轮流取**（每个行业各自限额）。

        排序用 **成交额** 而非收盘价：股价高低与「与事件的相关度」无关，按 close 排序会让
        高价股垄断每个行业的名额；成交额代表流动性与市场关注度，更适合作为行业代表标的。

        命中多个行业时必须**轮流（round-robin）交织**，不能按 Tushare 行业名顺序依次堆叠：
        `TUSHARE_TO_CHAIN` 里「有色金属」下挂着小金属/铜/铝/铅锌/黄金/矿物制品/稀土永磁 共 7 个
        Tushare 行业名，靠前堆叠会让它们把前几个名额全部占满 —— 命中「银行·保险·有色金属」
        的事件最终只反映出「小金属」（厦门钨业/北方稀土/盛和资源就是这么来的）。
        """
        if not industries:
            return []
        tushare_names = [t for t, c in TUSHARE_TO_CHAIN.items() if c and c in industries]
        buckets: dict[str, list[dict[str, Any]]] = {}
        for name in tushare_names:
            chain = TUSHARE_TO_CHAIN.get(name) or ""
            try:
                rows = database.query_all(
                    """SELECT sb.code, sb.name, si.industry, db.close AS price
                       FROM stock_basic sb
                       JOIN stock_industry si ON sb.code = si.code
                       JOIN daily_bars db ON db.code = sb.code
                          AND db.trade_date = (SELECT MAX(trade_date) FROM daily_bars)
                       WHERE si.industry = ?
                         AND sb.security_type='STOCK' AND sb.is_st=0 AND sb.is_delisted=0
                       ORDER BY db.amount DESC LIMIT ?""",
                    (name, per_industry),
                )
            except Exception as error:  # noqa: BLE001
                LOG.warning("查询事件后备股票池失败（%s）：%s", name, error)
                continue
            bucket = buckets.setdefault(chain, [])
            for r in rows:
                bucket.append({
                    "code": r["code"], "name": r["name"], "industry": r["industry"],
                    "price": float(r["price"] or 0), "score": 50.0, "price_date": "",
                    "source": "pool",
                })
        pooled: list[dict[str, Any]] = []
        for rank in range(per_industry):
            for bucket in buckets.values():
                if rank < len(bucket):
                    pooled.append(bucket[rank])
        return pooled

    def _match_by_name(ev: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        """事件文本里直接出现 A 股公司简称时，优先精确命中该股（最可靠的相关性信号）。"""
        text = _event_text(ev)
        if not text:
            return []
        codes = [code for code, name in _stock_names() if name in text][:limit]
        if not codes:
            return []
        placeholders = ",".join("?" * len(codes))
        try:
            rows = database.query_all(
                f"""SELECT sb.code, sb.name, si.industry, db.close AS price
                    FROM stock_basic sb
                    LEFT JOIN stock_industry si ON sb.code = si.code
                    LEFT JOIN daily_bars db ON db.code = sb.code
                       AND db.trade_date = (SELECT MAX(trade_date) FROM daily_bars)
                    WHERE sb.code IN ({placeholders})""",
                codes,
            )
        except Exception as error:  # noqa: BLE001
            LOG.warning("公司名直配查询失败：%s", error)
            return []
        return [
            {"code": r["code"], "name": r["name"], "industry": r["industry"] or "",
             "price": float(r["price"] or 0), "score": 50.0, "price_date": "",
             "source": "name"}
            for r in rows
        ]

    def _event_stocks(ev: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
        """多级匹配该事件的核心标的，由精确到宽松依次尝试：

        ① 事件文本直接提到 A 股公司简称 → 直接命中该股；
        ② 主升浪候选池按「直接命中行业」匹配；
        ③ 事件专属后备池（按直接命中行业查库，成交额排序取代表股）；
        ④ 宏观兜底（事件无行业词时用宏观→受益板块，如降准 → 银行/地产/券商）；
        ⑤ 候选池放宽到含上下游的宽松匹配。

        ③ 起的行业取不到时会退回事件分类（event_type），详见 ``_event_direct_industries``。
        ③ 返回的标的是「行业代表股」，其 ``score`` 是占位值 50（未做个股评分），
        报告里据此标注为「行业代表标的」而非「核心标的」。
        """
        hit = _match_by_name(ev, limit)
        if hit:
            return hit
        hit = match_event_candidates(ev, candidates, limit=limit)
        if hit:
            return hit
        text = _event_text(ev)
        hit = match_event_candidates(ev, _query_industry_pool(_event_direct_industries(ev)), limit=limit)
        if hit:
            return hit
        macro = macro_industries(text)
        hit = match_event_candidates(
            ev, _query_industry_pool(macro), limit=limit, industries=macro
        )
        if hit:
            return hit
        return match_event_candidates(ev, candidates, limit=limit, exact=False)

    def _event_card(idx: int, ev: dict[str, Any], border_color: str) -> None:
        mag = int(ev.get("magnitude") or 1)
        direction = str(ev.get("causal_direction") or "中性")
        color = _DIR_COLOR.get(direction, "#7f8c8d")
        parts.append(
            f'<div style="border:1px solid #e8ecef;border-left:3px solid {border_color};border-radius:6px;'
            'padding:8px 12px;margin:8px 0;background:#fbfcfd;">'
        )
        parts.append(
            f'<div style="font-size:13.5px;color:#2c3e50;"><b>{idx}. {esc(str(ev.get("title") or ""))}</b></div>'
        )
        parts.append(
            f'<div style="font-size:12px;color:#888;margin:3px 0;">'
            f'<span style="color:#e67e22;">{_event_stars(mag)}</span>　'
            f'<span style="color:{color};font-weight:600;">{esc(direction)}</span>　'
            f'{esc(str(ev.get("event_type") or "其他"))}　'
            f'持续性 {esc(str(ev.get("persistence") or "短期"))}　'
            f'{esc(str(ev.get("event_time") or "")[:16])}</div>'
        )
        if direction == "负":
            # 利空事件不落到个股：给出股票会被读成「推荐买入」，且宏观利空的受益板块映射
            # 方向本身就是反的（曾把「美联储加息概率升至72.4%」关联到稀土/钨股）。
            sectors = _event_risk_industries(ev)
            if sectors:
                risk_str = " · ".join(esc(s) for s in sectors)
                parts.append(
                    f'<div style="font-size:12.5px;color:#993c1d;">受影响的敏感板块：{risk_str}'
                    '<span style="color:#b0b6bd;">（利空事件，仅提示板块，不提供个股标的）</span></div>'
                )
            else:
                parts.append(
                    '<div style="font-size:12px;color:#b0b6bd;">利空事件，未识别到明确的受影响板块</div>'
                )
            parts.append('</div>')
            return
        stocks = _event_stocks(ev, limit=3)
        if stocks:
            from_pool = all(s.get("source") == "pool" for s in stocks)

            def _one_stock(s: dict[str, Any]) -> str:
                head = f'{esc(str(s["name"]))}({s["code"]}) ¥{float(s.get("price") or 0):.2f}'
                # 后备池的 score 恒为占位值 50（未做个股评分），展示出来会误导，
                # 因此这类标的只显示价格，并在下方标注「行业代表标的」。
                if s.get("source") == "pool":
                    return f'<span style="white-space:nowrap;">{head}</span>'
                return (f'<span style="white-space:nowrap;">{head}　评分'
                        f'<b style="color:#e67e22;">{float(s.get("score") or 0):.0f}</b></span>')

            stock_str = "　".join(_one_stock(s) for s in stocks)
            if from_pool:
                label = '关联行业代表标的（按成交额取该行业代表股，未做个股评分）：'
            elif any(s.get("source") == "name" for s in stocks):
                label = '关联核心标的（事件直接提及）：'
            else:
                label = '关联核心标的：'
            parts.append(f'<div style="font-size:12.5px;color:#333;">{label}{stock_str}</div>')
        else:
            parts.append('<div style="font-size:12px;color:#b0b6bd;">关联核心标的：当日候选池无直接匹配</div>')
        parts.append('</div>')

    parts: list[str] = []
    # 一、核心机会（正/中性）
    parts.append(
        '<h4 style="margin:18px 0 6px;padding-left:9px;border-left:4px solid #e74c3c;font-size:15px;color:#2c3e50;">'
        f'📰 今日核心事件 Top {len(opportunities)}（按重要性排序）</h4>'
    )
    parts.append(
        '<p style="margin:2px 0 8px;font-size:12px;color:#888;">'
        '只展示当日最新、重要性≥3 的核心消息；★ 越多越重要；每条关联评分最高的核心候选股票。</p>'
    )
    if opportunities:
        for i, ev in enumerate(opportunities[:limit], 1):
            _event_card(i, ev, "#e74c3c")
    else:
        parts.append('<p style="font-size:12.5px;color:#888;">今日无正向核心事件。</p>')

    # 二、风险提醒（负面/利空）
    if risks:
        parts.append(
            '<h4 style="margin:20px 0 6px;padding-left:9px;border-left:4px solid #1e8449;font-size:15px;color:#2c3e50;">'
            f'⚠️ 风险提醒（利空 / 宏观风险，{len(risks)} 条）</h4>'
        )
        parts.append(
            '<p style="margin:2px 0 8px;font-size:12px;color:#888;">'
            '当日负面事件，可能对相关板块或大盘构成压制，注意规避。</p>'
        )
        for i, ev in enumerate(risks[:6], 1):
            _event_card(i, ev, "#1e8449")

    if not standalone:
        return "".join(parts)

    header = (
        '<div style="font-family:Microsoft YaHei,Arial,sans-serif;max-width:860px;">'
        '<div style="background:#34495e;color:#fff;padding:12px 16px;border-radius:6px;">'
        f'<h3 style="margin:0;font-size:17px;">A股今日重要事件 · {datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")}</h3>'
        '<p style="margin:5px 0 0;font-size:13px;opacity:.92;">当日核心要闻按重要性排序；正向事件附命中行业的关联标的，利空事件仅提示敏感板块。</p>'
        '</div>'
    )
    footer = (
        '<p style="margin:16px 0 0;color:#999;font-size:12px;">'
        '事件重要性与方向由关键词/模型近似判断，仅供参考；系统自动扫描，不构成投资建议。</p>'
        '</div>'
    )
    return header + "".join(parts) + footer


def generate_daily_events_report(runtime: Any, limit: int = 12) -> tuple[bool, str, str]:
    """生成独立「今日重要事件」报告，返回 (是否有内容, 纯文本, HTML)。

    用于盘前无主升浪候选时兜底推送，保证用户每天早上仍能收到当日要闻。
    """
    events = top_events_today(runtime.database, limit=limit * 3)
    if not events:
        return False, "", ""
    core = [e for e in events if int(e.get("magnitude") or 1) >= 3] or events[:limit]
    opportunities = [e for e in core if str(e.get("causal_direction") or "") != "负"]
    risks = [e for e in core if str(e.get("causal_direction") or "") == "负"]
    text_lines = [f"A股今日重要事件 · 核心机会 {len(opportunities)} / 利空风险 {len(risks)}", "=" * 40]
    text_lines.append("【核心机会】")
    for i, ev in enumerate(opportunities[:limit], 1):
        text_lines.append(f"{i}. [{ev['event_type']}|强度{ev['magnitude']}|{ev['causal_direction']}] {ev['title']}")
    if risks:
        text_lines.append("")
        text_lines.append("【风险提醒·利空】")
        for i, ev in enumerate(risks[:6], 1):
            text_lines.append(f"{i}. [{ev['event_type']}|强度{ev['magnitude']}|{ev['causal_direction']}] {ev['title']}")
    html = format_daily_events_html(runtime.database, [], limit=limit, standalone=True)
    return True, "\n".join(text_lines), html


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

    daily_events = format_daily_events_html(database, results, limit=12)
    if daily_events:
        parts.append(daily_events)

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


def generate_main_rally_report(runtime: Any, current_quotes: dict[str, float] | None = None) -> tuple[bool, str, str]:
    """生成主升浪候选报告，返回 (是否有候选, 纯文本报告, HTML 富文本报告)。

    ``current_quotes`` 为可选的实时行情（代码→价格），盘前扫描会传入以保证价格是
    集合竞价价而非过时的数据库 bar。
    """
    results = score_main_rally_candidates(runtime, current_quotes=current_quotes)
    if not results:
        return False, "今日无主升浪候选", "<p>今日无主升浪候选</p>"
    sections = [format_main_rally_report(runtime.database, r, runtime.settings) for r in results[:10]]
    header = "A股主升浪扫描 · 候选 Top " + str(len(sections))
    text = header + "\n" + "=" * 40 + "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(sections)
    html = format_main_rally_report_html(runtime.database, results, runtime.settings)
    return True, text, html
