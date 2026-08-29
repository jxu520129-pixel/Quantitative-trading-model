"""事件库与宏观数据采集：主升浪策略「事件→宏观」基础层。

MVP 阶段用关键词做事件分类（替代 LLM 结构化抽取），用 AkShare 采集
权威宏观指标与全球财经快讯。产业链传导与盈利预期在后续阶段接入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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


def llm_extract_events(events: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    """批量调用 LLM 结构化抽取事件字段，返回 {index: {...}}。无 key 或失败返回空。"""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key or not events:
        return {}
    lines = [f"{i}. 标题：{ev['title']} 摘要：{ev['summary'][:80]}" for i, ev in enumerate(events)]
    prompt = (
        "你是A股事件分析助手。对下面每条新闻提取：event_type（政策/金融/科技/能源/资源/农业/医药/消费/地产基建/产业/战争/自然灾害/其他）、"
        "magnitude（1-5整数，重要性）、causal_direction（正/负/中性）、"
        "affected_industries（受影响行业关键词数组，可为空）、persistence（短期/中期/长期）。\n"
        "只返回JSON对象，格式：{\"results\":[{\"index\":0,\"event_type\":\"科技\",\"magnitude\":4,\"causal_direction\":\"正\",\"affected_industries\":[\"半导体\"],\"persistence\":\"中期\"},...]}\n\n"
        + "\n".join(lines)
    )
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
        data = json.loads(content)
        result: dict[int, dict[str, Any]] = {}
        for item in data.get("results", []):
            result[int(item.get("index", -1))] = item
        return result
    except Exception as error:
        LOG.warning("LLM 事件抽取失败，回退关键词：%s", error)
        return {}


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fetch_global_events(database: Database, limit: int = 200) -> int:
    """采集全球财经快讯，关键词分类后写入事件库。返回新增条数。"""
    import akshare as ak

    try:
        frame = ak.stock_info_global_em()
    except Exception as error:
        LOG.warning("全球财经快讯获取失败：%s", error)
        return 0
    now = utc_now_text()
    items: list[dict[str, str]] = []
    for item in frame.head(limit).to_dict("records"):
        title = str(item.get("标题", "")).strip()
        if not title:
            continue
        items.append({
            "title": title,
            "summary": str(item.get("摘要", "")).strip(),
            "event_time": str(item.get("发布时间", "")).strip(),
        })
    llm_results = llm_extract_events(items)
    rows = []
    for i, item in enumerate(items):
        title = item["title"]
        summary = item["summary"]
        event_time = item["event_time"]
        text = title + " " + summary
        event_id = hashlib.md5(f"{title}|{event_time}".encode("utf-8")).hexdigest()
        if i in llm_results:
            r = llm_results[i]
            event_type = str(r.get("event_type") or "其他")
            magnitude = int(r.get("magnitude") or event_magnitude(text))
            direction = str(r.get("causal_direction") or "中性")
            industries = ",".join(r.get("affected_industries") or []) or ",".join(affected_industries(text))
            persistence = str(r.get("persistence") or "短期")
        else:
            event_type = classify_event(text)
            magnitude = event_magnitude(text)
            direction = event_direction(text)
            industries = ",".join(affected_industries(text))
            persistence = "中期" if magnitude >= 3 else "短期"
        rows.append((event_id, title, summary[:200], event_time, event_type, "全球财经快讯", "B", magnitude, direction, industries, persistence, now))
    if rows:
        database.executemany(
            """INSERT OR IGNORE INTO events(id,title,summary,event_time,event_type,source,source_level,magnitude,causal_direction,affected_industries,persistence,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
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


def update_fundamentals(database: Database) -> dict[str, int]:
    """采集事件 + 宏观数据，返回 {events, macro}。"""
    events = fetch_global_events(database)
    macro = fetch_macro_indicators(database)
    LOG.info("基本面数据更新完成：事件 %s 条，宏观 %s 条", events, macro)
    return {"events": events, "macro": macro}
