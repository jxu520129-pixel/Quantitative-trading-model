"""事件库与宏观数据采集：主升浪策略「事件→宏观」基础层。

MVP 阶段用关键词做事件分类（替代 LLM 结构化抽取），用 AkShare 采集
权威宏观指标与全球财经快讯。产业链传导与盈利预期在后续阶段接入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .database import Database
from .industry_chain import affected_industries
from .models import utc_now_text


LOG = logging.getLogger(__name__)

# 事件类型 -> 关键词（MVP 粗分类，后续可替换为 LLM 结构化抽取）
EVENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "政策": ("政策", "央行", "财政", "监管", "证监会", "发改委", "国务院", "补贴", "降息", "降准", "降税", "关税", "规划", "批复"),
    "金融": ("利率", "汇率", "债市", "流动性", "M2", "信贷", "融资", "资本", "银行", "券商", "保险", "基金", "IPO"),
    "科技": ("AI", "人工智能", "芯片", "半导体", "算力", "机器人", "新能源", "光伏", "锂电", "云计算", "数据", "软件", "大模型"),
    "能源": ("能源", "石油", "原油", "天然气", "煤炭", "电力", "电网", "储能", "氢能"),
    "资源": ("黄金", "铜", "铝", "锂", "稀土", "钢铁", "铁矿石", "有色金属", "贵金属", "白银"),
    "农业": ("农业", "粮食", "种植", "养殖", "生猪", "饲料", "种业", "化肥", "渔业"),
    "医药": ("医药", "医疗", "药品", "疫苗", "创新药", "医疗器械", "生物", "医院"),
    "消费": ("消费", "零售", "食品", "饮料", "白酒", "汽车", "家电", "旅游", "免税", "餐饮"),
    "地产基建": ("房地产", "地产", "房价", "楼市", "基建", "交通", "铁路", "机场", "港口", "建筑", "水利"),
    "产业": ("产能", "涨价", "供需", "订单", "投产", "扩产", "库存", "供给", "需求"),
    "战争": ("战争", "冲突", "制裁", "地缘", "军事", "导弹", "封锁"),
    "自然灾害": ("地震", "台风", "洪水", "干旱", "疫情"),
}

# 宏观指标 -> (AkShare 函数名, 取值列, 日期列候选)
MACRO_INDICATORS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gdp_yoy": ("macro_china_gdp", "国内生产总值-同比增长", ("季度",)),
    "cpi_yoy": ("macro_china_cpi", "全国-同比增长", ("月份",)),
    "ppi_yoy": ("macro_china_ppi", "当月同比增长", ("月份",)),
    "m2_yoy": ("macro_china_money_supply", "货币和准货币(M2)-同比增长", ("月份",)),
    "bond_10y": ("bond_zh_us_rate", "中国国债收益率10年", ("日期",)),
}


def classify_event(text: str) -> str:
    """用关键词做事件类型粗分类。"""
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return event_type
    return "其他"


# 事件规模关键词 -> 强度分（1-5）
_MAGNITUDE_KEYWORDS: dict[str, int] = {
    "历史": 5, "历史性": 5, "危机": 5, "紧急": 4, "突破": 4, "暴涨": 4, "暴跌": 4,
    "罕见": 4, "重磅": 4, "重大": 4, "创新高": 4, "大规模": 4,
    "宣布": 3, "利好": 3, "利空": 3, "超预期": 3, "制裁": 3, "出台": 3,
    "增长": 2, "下滑": 2, "发布": 2, "报告": 2, "调整": 2,
}

_POSITIVE_KEYWORDS = ("上涨", "利好", "增长", "突破", "新高", "盈利", "增持", "回购", "扩张", "繁荣", "回暖", "上调", "复苏")
_NEGATIVE_KEYWORDS = ("下跌", "利空", "衰退", "制裁", "暴跌", "亏损", "减持", "危机", "风险", "收缩", "下调", "违约")


def event_magnitude(text: str) -> int:
    """事件规模（1-5）：按关键词强度打分。"""
    magnitude = 1
    for keyword, score in _MAGNITUDE_KEYWORDS.items():
        if keyword in text:
            magnitude = max(magnitude, score)
    return magnitude


def event_direction(text: str) -> str:
    """事件正负方向：正（利好）/负（利空）/中性。"""
    positive = any(kw in text for kw in _POSITIVE_KEYWORDS)
    negative = any(kw in text for kw in _NEGATIVE_KEYWORDS)
    if positive and not negative:
        return "正"
    if negative and not positive:
        return "负"
    return "中性"


def _strip_json(content: str) -> str:
    """去除 LLM 可能包裹的 markdown 代码块（```json ... ```）。"""
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.lstrip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.rstrip("`").strip()
    return content


def _llm_call_once(api_key: str, base_url: str, model: str, prompt: str, retries: int = 3) -> dict[int, dict[str, Any]]:
    """单次 LLM 抽取调用（带重试与 JSON 清洗），返回 {index: {...}}。失败返回空。"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}, "temperature": 0},
                timeout=90,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(_strip_json(content))
            batch_result: dict[int, dict[str, Any]] = {}
            for item in data.get("results", []):
                try:
                    idx = int(item.get("index", -1))
                except (TypeError, ValueError):
                    continue
                if idx >= 0:
                    batch_result[idx] = item
            return batch_result
        except Exception as error:
            if attempt == retries:
                LOG.warning("LLM 抽取失败（已重试 %s 次），该批回退关键词：%s", retries, error)
                return {}
            time.sleep(1.5 * attempt)
    return {}


def llm_extract_events(events: list[dict[str, str]], batch_size: int = 30) -> dict[int, dict[str, Any]]:
    """分批调用 LLM 结构化抽取事件字段，返回 {index: {...}}。无 key 或失败返回空。

    每条事件抽取：event_type / magnitude（1-5）/ causal_direction / persistence。
    注意：affected_industries 不在这里抽取（统一由关键词函数映射到标准行业，见
    ``fetch_global_events``），避免 LLM 自由文本与标准行业表不一致。
    """
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key or not events:
        if events:
            LOG.warning("未配置 LLM_API_KEY，事件抽取回退关键词分类")
        return {}

    result: dict[int, dict[str, Any]] = {}
    started = time.time()
    for batch_start in range(0, len(events), batch_size):
        batch = events[batch_start:batch_start + batch_size]
        lines = [f"{i}. 标题：{ev['title']} 摘要：{ev['summary'][:80]}" for i, ev in enumerate(batch, start=batch_start)]
        prompt = (
            "你是A股事件分析助手。对下面每条新闻提取：event_type（政策/金融/科技/能源/资源/农业/医药/消费/地产基建/产业/战争/自然灾害/其他）、"
            "magnitude（1-5整数，重要性）、causal_direction（正/负/中性）、persistence（短期/中期/长期）。\n"
            "只返回JSON对象，格式：{\"results\":[{\"index\":0,\"event_type\":\"科技\",\"magnitude\":4,\"causal_direction\":\"正\",\"persistence\":\"中期\"},...]}\n\n"
            + "\n".join(lines)
        )
        result.update(_llm_call_once(api_key, base_url, model, prompt))
    LOG.info("LLM 事件抽取完成：%s/%s 条成功，耗时 %.1fs", len(result), len(events), time.time() - started)
    return result


def llm_extract_entities(events: list[dict[str, str]], batch_size: int = 30) -> dict[int, list[str]]:
    """用 LLM 从新闻里抽取 A 股公司实体（公司名/市场俗称），返回 ``{index: [实体, ...]}``。

    抽取的实体供「查找股票」的公司名精确匹配使用：配合 ``ALIAS_TO_CODE`` + 股票简称，
    把新闻里提到的公司精确锁定到股票代码，弥补纯子串匹配的歧义（如「长城」既指汽车又指地名）
    与遗漏（新闻用全称、别名而非股票简称）。无 key 或失败返回空。
    """
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key or not events:
        return {}

    result: dict[int, list[str]] = {}
    for batch_start in range(0, len(events), batch_size):
        batch = events[batch_start:batch_start + batch_size]
        lines = [f"{i}. {ev['title']} {ev['summary'][:60]}" for i, ev in enumerate(batch, start=batch_start)]
        prompt = (
            "你是A股信息抽取助手。对下面每条新闻，抽取其中明确提到的**中国A股上市公司名称或市场俗称**"
            "（如：宁德时代、贵州茅台、宁王、比亚迪、隆基绿能）。只抽取公司实体，"
            "不要抽取产品名、行业、地名、宏观政策词；没有公司实体则返回空数组。\n"
            "只返回JSON，格式：{\"results\":[{\"index\":0,\"entities\":[\"宁德时代\",\"比亚迪\"]},...]}\n\n"
            + "\n".join(lines)
        )
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}],
                          "response_format": {"type": "json_object"}, "temperature": 0},
                    timeout=90,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(_strip_json(content))
                for item in data.get("results", []):
                    try:
                        idx = int(item.get("index", -1))
                    except (TypeError, ValueError):
                        continue
                    if idx >= 0:
                        entities = [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
                        if entities:
                            result[idx] = entities
                break
            except Exception as error:  # noqa: BLE001
                if attempt == 3:
                    LOG.warning("LLM 实体抽取失败（已重试）：%s", error)
                else:
                    time.sleep(1.5 * attempt)
    return result


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_title(title: str) -> str:
    """N3：标题归一化——仅保留中文/字母/数字，去掉标点与空白，用于跨来源去重。"""
    import re

    return re.sub(r"[^\w\u4e00-\u9fff]", "", title or "")


def _dedupe_events(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """N3：跨来源去重——归一化标题完全相同，或相似度 ≥0.85，视为同一事件。"""
    import difflib

    result: list[dict[str, str]] = []
    seen_norm: list[str] = []
    for item in items:
        norm = _normalize_title(item["title"])
        if not norm:
            continue
        dup = False
        for prev in seen_norm:
            if norm == prev or (
                len(norm) >= 10 and len(prev) >= 10
                and difflib.SequenceMatcher(None, norm, prev).ratio() >= 0.85
            ):
                dup = True
                break
        if dup:
            continue
        seen_norm.append(norm)
        result.append(item)
    return result


def fetch_global_events(database: Database, limit: int = 200) -> int:
    """采集多源财经快讯（东方财富 + 财联社电报），LLM 分类 + 实体抽取后写入事件库。返回新增条数。"""
    import akshare as ak

    now = utc_now_text()
    items: list[dict[str, str]] = []
    # 源 1：东方财富全球财经快讯
    try:
        frame = ak.stock_info_global_em()
        for item in frame.head(limit).to_dict("records"):
            title = str(item.get("标题", "")).strip()
            if not title or title == "nan":
                continue
            items.append({
                "title": title,
                "summary": str(item.get("摘要", "")).strip(),
                "event_time": str(item.get("发布时间", "")).strip(),
                "source": "东方财富快讯",
            })
    except Exception as error:  # noqa: BLE001
        LOG.warning("东方财富快讯获取失败：%s", error)
    # 源 2：财联社电报（标题列可能为空，回退用内容前 40 字）
    try:
        frame = ak.stock_info_global_cls(symbol="全部")
        for item in frame.head(limit).to_dict("records"):
            title = str(item.get("标题", "")).strip()
            content = str(item.get("内容", "")).strip()
            if (not title or title == "nan") and content and content != "nan":
                title = content[:40]
            if not title or title == "nan":
                continue
            pub_date = str(item.get("发布日期", "")).strip()[:10]
            pub_time = str(item.get("发布时间", "")).strip()
            event_time = f"{pub_date} {pub_time}".strip()
            items.append({
                "title": title,
                "summary": content[:200] if content != "nan" else "",
                "event_time": event_time,
                "source": "财联社电报",
            })
    except Exception as error:  # noqa: BLE001
        LOG.warning("财联社电报获取失败：%s", error)
    # 源 3：华尔街见闻快讯流（公开 API，全球财经，无需 key）
    try:
        resp = requests.get(
            "https://api-one.wallstcn.com/apiv1/content/lives",
            params={"channel": "global-china", "limit": "50"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        wscn_items = (resp.json().get("data", {}).get("items") or [])[:limit]
        for item in wscn_items:
            content = re.sub(r"<[^>]+>", "", str(item.get("content_text") or "")).strip()
            if not content:
                continue
            ts = item.get("display_time") or 0
            try:
                event_time = datetime.fromtimestamp(int(ts), tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OSError):
                event_time = ""
            items.append({
                "title": content[:40],
                "summary": content[:200],
                "event_time": event_time,
                "source": "华尔街见闻",
            })
    except Exception as error:  # noqa: BLE001
        LOG.warning("华尔街见闻快讯获取失败：%s", error)
    # 源 4：富途牛牛快讯（港美股）
    try:
        frame = ak.stock_info_global_futu()
        for item in frame.head(limit).to_dict("records"):
            title = str(item.get("标题", "")).strip()
            if not title or title == "nan":
                continue
            items.append({
                "title": title,
                "summary": str(item.get("内容", "")).strip()[:200],
                "event_time": str(item.get("发布时间", "")).strip(),
                "source": "富途牛牛",
            })
    except Exception as error:  # noqa: BLE001
        LOG.warning("富途牛牛快讯获取失败：%s", error)

    if not items:
        return 0
    items = _dedupe_events(items)
    llm_results = llm_extract_events(items)
    entity_results = llm_extract_entities(items)
    rows = []
    for i, item in enumerate(items):
        title = item["title"]
        summary = item["summary"]
        # event_time 为事件发布时间（北京时间，AkShare 返回）；created_at 为入库时间（UTC）。
        # 「当日」判断必须用 event_time，勿用 created_at（两者差 8 小时）。
        event_time = item["event_time"]
        text = title + " " + summary
        event_id = hashlib.md5(f"{title}|{event_time}".encode("utf-8")).hexdigest()
        # 受影响行业统一用关键词函数映射到标准行业（含上下游传导），不依赖 LLM 自由文本。
        # 注意：本字段含产业链上下游传导，集合很宽（科技类新闻几乎都会带上「半导体」），
        # 只适合做「个股事件分」这类聚合评分。做「事件 → 个股」精确匹配请改用
        # industry_chain.direct_industries，否则不同事件会匹配到同一批个股。
        industries = ",".join(affected_industries(text))
        if i in llm_results:
            r = llm_results[i]
            event_type = str(r.get("event_type") or classify_event(text))
            magnitude = max(1, min(5, int(_num(r.get("magnitude")) or event_magnitude(text))))
            direction = str(r.get("causal_direction") or event_direction(text))
            persistence = str(r.get("persistence") or ("中期" if magnitude >= 3 else "短期"))
        else:
            event_type = classify_event(text)
            magnitude = event_magnitude(text)
            direction = event_direction(text)
            persistence = "中期" if magnitude >= 3 else "短期"
        entities = ",".join(entity_results.get(i, []))
        rows.append((event_id, title, summary[:200], event_time, event_type, item["source"], "B", magnitude, direction, industries, persistence, entities, now))
    if rows:
        database.executemany(
            """INSERT OR IGNORE INTO events(id,title,summary,event_time,event_type,source,source_level,magnitude,causal_direction,affected_industries,persistence,entities,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    return len(rows)


def fetch_macro_indicators(database: Database, recent: int = 60) -> int:
    """采集宏观指标（GDP/CPI/PPI/M2/国债收益率）写入宏观库。返回写入条数。"""
    import akshare as ak

    now = utc_now_text()
    inserted = 0
    for indicator, (func_name, value_col, date_cols) in MACRO_INDICATORS.items():
        try:
            frame = getattr(ak, func_name)()
        except Exception as error:
            LOG.warning("%s 获取失败：%s", func_name, error)
            continue
        date_col = next((c for c in frame.columns if c in date_cols), None)
        if date_col is None or value_col not in frame.columns:
            LOG.warning("%s 缺少日期/值列：%s", func_name, list(frame.columns))
            continue
        frame = frame.sort_values(date_col)
        for item in frame.tail(recent).to_dict("records"):
            period = str(item[date_col]).strip()
            value = _num(item[value_col])
            if not period or value == 0:
                continue
            database.execute(
                "INSERT OR REPLACE INTO macro_indicators(indicator,period,value,updated_at) VALUES(?,?,?,?)",
                (indicator, period, value, now),
            )
            inserted += 1
    return inserted


def fetch_stock_hot_rank(database: Database, limit: int = 100) -> int:
    """采集东财人气榜，写入 ``stock_hot_rank``（按 code+trade_date 覆盖）。返回写入条数。"""
    import akshare as ak

    try:
        frame = ak.stock_hot_rank_em()
    except Exception as error:  # noqa: BLE001
        LOG.warning("东财人气榜获取失败：%s", error)
        return 0
    now = utc_now_text()
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    rows = []
    for i, item in enumerate(frame.head(limit).to_dict("records"), start=1):
        code = str(item.get("代码") or item.get("股票代码") or "").strip()
        name = str(item.get("股票名称") or item.get("名称") or "").strip()
        if not code:
            continue
        rows.append((code, name, i, today, now))
    if rows:
        database.executemany(
            "INSERT OR REPLACE INTO stock_hot_rank(code,name,rank,trade_date,updated_at) VALUES(?,?,?,?,?)",
            rows,
        )
    return len(rows)


def core_hot_stocks(database: Database, days: int = 3, top_n: int = 50) -> list[dict[str, Any]]:
    """识别「连续多日热门」的核心炒作标的。

    对最近 ``days`` 个交易日的人气榜，统计每只股票的上榜天数与最佳排名，
    综合分 = 上榜天数 × 10 + (30 - 最佳排名)，至少上榜 2 天才算核心标的。
    返回按综合分降序的列表，供「实体抽取归类」与「文字梗识别」使用。
    """
    dates = [
        r["trade_date"]
        for r in database.query_all(
            "SELECT DISTINCT trade_date FROM stock_hot_rank ORDER BY trade_date DESC LIMIT ?", (days,)
        )
    ]
    if len(dates) < 2:
        return []
    agg: dict[str, dict[str, Any]] = {}
    for d in dates:
        for r in database.query_all(
            "SELECT code, name, rank FROM stock_hot_rank WHERE trade_date=? ORDER BY rank ASC", (d,)
        ):
            code = r["code"]
            entry = agg.setdefault(code, {"code": code, "name": r["name"], "days": 0, "best_rank": 999})
            entry["days"] += 1
            entry["best_rank"] = min(entry["best_rank"], int(r["rank"]))
    core = []
    for entry in agg.values():
        if entry["days"] < 2:
            continue
        entry["score"] = entry["days"] * 10 + max(0, 30 - entry["best_rank"])
        core.append(entry)
    core.sort(key=lambda x: x["score"], reverse=True)
    return core[:top_n]


# 股票简称里的「通用/行业词」，统计文字梗时排除，避免把「股份」「科技」当成炒作概念
_STOP_PAIRS = {
    "股份", "集团", "科技", "控股", "有限", "实业", "国际", "投资", "发展",
    "电气", "电子", "智能", "能源", "材料", "新材", "生物", "医药", "环保",
    "证券", "银行", "汽车", "重工", "工业", "建设", "电力", "通信", "信息",
    "软件", "数据", "传媒", "地产", "食品", "饮料", "光电", "精工", "动力",
}


def detect_word_concepts(database: Database, min_count: int = 3, top_n: int = 100) -> list[dict[str, Any]]:
    """识别 A 股「文字炒作概念」（如「东方」字、「洋」字、生肖字等）。

    统计核心热门股票简称里的**连续二字组合**（排除通用词），某个二字词在
    ``min_count`` 只以上热门股票简称里出现，视为一个「XX字概念」。
    返回按出现次数降序的概念列表，含命中的股票代码。
    """
    core = core_hot_stocks(database, days=3, top_n=top_n)
    if not core:
        return []
    pair_count: dict[str, int] = {}
    pair_stocks: dict[str, list[str]] = {}
    for c in core:
        name = c["name"]
        for i in range(len(name) - 1):
            pair = name[i:i + 2]
            if pair in _STOP_PAIRS:
                continue
            pair_count[pair] = pair_count.get(pair, 0) + 1
            stocks = pair_stocks.setdefault(pair, [])
            if c["code"] not in stocks:
                stocks.append(c["code"])
    concepts = []
    for pair, count in pair_count.items():
        if count >= min_count:
            concepts.append({"concept": pair, "count": count, "stocks": pair_stocks[pair][:10]})
    concepts.sort(key=lambda x: x["count"], reverse=True)
    return concepts


def update_fundamentals(database: Database) -> dict[str, int]:
    """采集事件 + 宏观 + 人气榜数据，返回 {events, macro, hot}。"""
    events = fetch_global_events(database)
    macro = fetch_macro_indicators(database)
    hot = fetch_stock_hot_rank(database)
    LOG.info("基本面数据更新完成：事件 %s 条，宏观 %s 条，人气榜 %s 只", events, macro, hot)
    return {"events": events, "macro": macro, "hot": hot}
