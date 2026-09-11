"""事件 → 行业 → 个股 匹配的回归测试。

覆盖三类历史 bug：
1. **修辞/地名误命中**——「黄金发展期」被 `有色金属` 关键词「黄金」命中，导致
   「AI制药正式进入临床验证阶段 产业链迎来黄金发展期」关联到厦门钨业/北方稀土/盛和资源。
   同源问题还有「无锡」命中「锡」、「种子轮」命中「种子」、「游戏规则」命中「游戏」等。
2. **文本无行业词时静默落空**——事件分类（event_type）为「医药」但标题不含任何行业词，
   直接行业集合为空，后备池于是用恰好命中的无关行业顶替。
3. **多行业后备池被单一行业占满 / 利空事件给个股**——「有色金属」下挂着 7 个 Tushare
   行业名（小金属/铜/铝/铅锌/黄金/矿物制品/稀土永磁），按行业名顺序堆叠会把名额占满，
   命中「银行·保险·有色金属」的事件最终只反映「小金属」；且利空事件（美联储加息）
   本不该给出个股标的。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ashare_quant.database import Database
from ashare_quant.industry_chain import direct_industries
from ashare_quant.main_rally import format_daily_events_html, match_event_candidates
from ashare_quant.models import utc_now_text


@pytest.mark.parametrize(
    "text, expected",
    [
        # 报告里实际出错的标题：必须命中「医药」，且**不得**命中「有色金属」
        ("AI制药正式进入临床验证阶段 产业链迎来黄金发展期", {"医药"}),
        ("医药板块迎来黄金发展期", {"医药"}),
        # 真正的黄金新闻仍要命中
        ("黄金价格创历史新高 央行持续增持", {"有色金属"}),
        ("现货黄金期货主力合约走强", {"有色金属"}),
        ("白银价格跟随黄金上涨", {"有色金属"}),
        ("锡价持续上涨 锡矿供应偏紧", {"有色金属"}),
        # 修辞 / 地名 / 术语
        ("股权激励设定黄金分割行权价", set()),
        ("国庆黄金周旅游消费数据超预期", {"旅游"}),
        ("江苏省无锡市推进集成电路产业", {"半导体"}),
        ("甘肃白银市新能源项目开工", set()),
        ("创业投资：种子轮融资回暖", set()),
        ("面板数据回归结果显示显著", set()),
    ],
)
def test_ambiguous_keywords(text: str, expected: set[str]) -> None:
    assert set(direct_industries(text)) == expected


def test_event_type_fallback_when_text_has_no_industry_word() -> None:
    """标题里没有任何行业词时，用事件分类（event_type）兜底匹配，而非落空。"""
    candidates = [
        {"code": "601318", "name": "中国平安", "industry": "保险", "score": 88.0},
        {"code": "600549", "name": "厦门钨业", "industry": "小金属", "score": 70.0},
    ]
    event = {
        "title": "监管发布新规规范行业发展",
        "summary": "",
        "event_type": "保险",
        "affected_industries": "",
    }
    assert direct_industries(event["title"]) == []
    assert [c["code"] for c in match_event_candidates(event, candidates, limit=3)] == ["601318"]


RARE_EARTH = [
    ("600549", "厦门钨业", "小金属", 49.73, 2.0e9),
    ("600111", "北方稀土", "小金属", 38.18, 2.4e9),
    ("600392", "盛和资源", "小金属", 21.85, 1.5e9),
]
PHARMA = [
    ("603259", "药明康德", "医疗保健", 68.40, 3.0e9),
    ("300760", "迈瑞医疗", "医疗器械", 245.10, 1.2e9),
]
FINANCE = [
    ("000002", "万科A", "全国地产", 9.12, 8.0e8),
    ("600036", "招商银行", "银行", 42.30, 4.0e9),
    ("600030", "中信证券", "证券", 28.60, 5.0e9),
]


def _seed_stocks(database: Database, trade_date: str, stocks: list[tuple]) -> None:
    now = utc_now_text()
    with database.connect() as conn:
        conn.executemany(
            "INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,"
            "is_suspended,list_date,updated_at) VALUES(?,?,'STOCK','MAIN',0,0,0,'20100101',?)",
            [(code, name, now) for code, name, _, _, _ in stocks],
        )
        conn.executemany(
            "INSERT INTO stock_industry(code,industry,updated_at) VALUES(?,?,?)",
            [(code, industry, now) for code, _, industry, _, _ in stocks],
        )
        conn.executemany(
            "INSERT INTO daily_bars(code,trade_date,open,high,low,close,volume,amount,"
            "pre_close,source) VALUES(?,?,?,?,?,?,?,?,'','tushare')",
            [
                (code, trade_date, price, price, price, price, 1.0e7, amount)
                for code, _, _, price, amount in stocks
            ],
        )


def _seed_event(
    database: Database,
    event_id: str,
    title: str,
    event_type: str,
    direction: str = "正",
    magnitude: int = 4,
    affected_industries: str = "",
) -> None:
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO events(id,title,summary,event_time,event_type,source,source_level,"
            "magnitude,causal_direction,affected_industries,persistence,created_at) "
            "VALUES(?,?,'',?,?,'测试','A',?,?,?,'长期',?)",
            (
                event_id,
                title,
                f"{today} 08:48:00",
                event_type,
                magnitude,
                direction,
                affected_industries,
                utc_now_text(),
            ),
        )


def _fresh_db(tmp_path) -> tuple[Database, str]:
    database = Database(tmp_path / "events.db")
    database.ensure_schema()
    trade_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    _seed_stocks(database, trade_date, RARE_EARTH + PHARMA + FINANCE)
    return database, trade_date


def test_pharma_event_matches_pharma_stocks_not_rare_earth(tmp_path) -> None:
    database, _ = _fresh_db(tmp_path)
    _seed_event(
        database,
        "ev-pharma",
        "AI制药正式进入临床验证阶段 产业链迎来黄金发展期",
        event_type="医药",
        affected_industries="医药",
    )

    html = format_daily_events_html(database, candidates=[], limit=5)

    assert "药明康德" in html
    assert "迈瑞医疗" in html
    assert "厦门钨业" not in html
    assert "北方稀土" not in html
    assert "盛和资源" not in html
    # 后备池标的的 score 是占位值，报告须如实标注为「行业代表标的」而非「核心标的」
    assert "关联行业代表标的" in html


def test_negative_event_reports_sectors_without_stocks(tmp_path) -> None:
    """利空事件只提示敏感板块，不得给出个股标的（否则会被读成买入建议）。"""
    database, _ = _fresh_db(tmp_path)
    _seed_event(
        database,
        "ev-fed",
        "美联储观察：市场预计美联储9月加息概率升至72.4%",
        event_type="金融",
        direction="负",
        affected_industries="",
    )

    html = format_daily_events_html(database, candidates=[], limit=5)

    assert "受影响的敏感板块" in html
    assert "仅提示板块，不提供个股标的" in html
    assert "银行" in html
    assert "有色金属" in html
    for name in ("厦门钨业", "北方稀土", "盛和资源", "药明康德", "招商银行"):
        assert name not in html


def test_multi_industry_pool_spreads_across_sectors(tmp_path) -> None:
    """命中多个行业时后备池须跨行业轮流取，不能被排在前的单一行业占满。"""
    database, _ = _fresh_db(tmp_path)
    _seed_event(
        database,
        "ev-rrr",
        "央行宣布降准0.5个百分点 释放长期资金",
        event_type="政策",
        affected_industries="",
    )

    html = format_daily_events_html(database, candidates=[], limit=5)

    # 降准 → 银行/券商/房地产，三个板块各出一只代表股
    assert "招商银行" in html
    assert "中信证券" in html
    assert "万科A" in html
    assert "厦门钨业" not in html
    assert "药明康德" not in html
