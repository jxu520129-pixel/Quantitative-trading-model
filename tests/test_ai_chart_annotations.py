"""AI 画线标注（levels/markers）的校验与解析测试。

覆盖要点：
- 价格必须落在 K 线最高/最低价 ±5% 范围内（防止大模型编造价位压缩坐标轴）；
- 日期必须对齐到真实交易日（大模型常给出非交易日/节假日）；
- 类型白名单外回退 other；同价/同日去重；label 截断 8 字；各最多 3 条；
- 文字模式末尾 ANNOTATION_JSON 行的剥离与解析。
"""

from __future__ import annotations

import pandas as pd

from ashare_quant.ai_analyst import (
    _align_trade_date,
    _sanitize_chart_hints,
    _split_annotation_json,
)


def _frame() -> pd.DataFrame:
    """30 根模拟日线：close 从 10 递增到 39，日期 2026-10-01~30。"""
    closes = [float(i) for i in range(10, 40)]
    return pd.DataFrame({
        "trade_date": [f"2026-10-{d:02d}" for d in range(1, 31)],
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * 30,
        "amount": [1e7] * 30,
        "pre_close": [9.9] * 30,
    })


class TestAlignTradeDate:
    def test_aligns_to_previous_trading_day(self):
        dates = ["2026-10-08", "2026-10-09", "2026-10-12"]
        assert _align_trade_date(dates, "2026-10-10") == "2026-10-09"

    def test_before_first_returns_first(self):
        dates = ["2026-10-08", "2026-10-09"]
        assert _align_trade_date(dates, "2026-10-01") == "2026-10-08"

    def test_empty_returns_none(self):
        assert _align_trade_date([], "2026-10-10") is None


class TestSanitizeChartHints:
    def test_valid_level_passes_through(self):
        hints = _sanitize_chart_hints(
            [{"price": 20.0, "label": "支撑位", "type": "support"}], [], _frame()
        )
        assert hints["levels"] == [{"price": 20.0, "label": "支撑位", "type": "support"}]
        assert hints["markers"] == []

    def test_out_of_range_price_dropped(self):
        hints = _sanitize_chart_hints(
            [
                {"price": 9999.0, "label": "支撑位", "type": "support"},
                {"price": 20.0, "label": "压力位", "type": "resistance"},
            ],
            [],
            _frame(),
        )
        assert hints["levels"] == [{"price": 20.0, "label": "压力位", "type": "resistance"}]

    def test_negative_and_missing_label_dropped(self):
        hints = _sanitize_chart_hints(
            [
                {"price": -3.0, "label": "支撑位", "type": "support"},
                {"price": 20.0, "label": "", "type": "support"},
                {"price": 21.0, "label": None, "type": "support"},
            ],
            [],
            _frame(),
        )
        assert hints["levels"] == []

    def test_unknown_level_type_falls_back_to_other(self):
        hints = _sanitize_chart_hints(
            [{"price": 20.0, "label": "关键位", "type": "haha"}], [], _frame()
        )
        assert hints["levels"][0]["type"] == "other"

    def test_duplicate_price_kept_once(self):
        hints = _sanitize_chart_hints(
            [
                {"price": 20.0, "label": "支撑位", "type": "support"},
                {"price": 20.001, "label": "重复价", "type": "stop"},
            ],
            [],
            _frame(),
        )
        assert len(hints["levels"]) == 1
        assert hints["levels"][0]["label"] == "支撑位"

    def test_label_truncated_to_8_chars(self):
        hints = _sanitize_chart_hints(
            [{"price": 20.0, "label": "这是一个非常非常长的标签说明文字", "type": "support"}],
            [],
            _frame(),
        )
        assert hints["levels"][0]["label"] == "这是一个非常非常"

    def test_marker_date_aligned_and_close_filled(self):
        # 2026-10-31 不在数据中 → 对齐到最近的交易日 2026-10-30，close=39.0
        hints = _sanitize_chart_hints(
            [], [{"date": "2026-10-31", "label": "放量突破", "type": "buy"}], _frame()
        )
        assert hints["markers"] == [
            {"date": "2026-10-30", "price": 39.0, "label": "放量突破", "type": "buy"}
        ]

    def test_marker_duplicate_date_kept_once(self):
        hints = _sanitize_chart_hints(
            [],
            [
                {"date": "2026-10-09", "label": "买点", "type": "buy"},
                {"date": "2026-10-09", "label": "另一个", "type": "risk"},  # 同日重复
            ],
            _frame(),
        )
        assert len(hints["markers"]) == 1
        assert hints["markers"][0]["label"] == "买点"

    def test_max_three_each(self):
        levels = [{"price": float(i), "label": f"L{i}", "type": "other"} for i in range(11, 16)]
        markers = [{"date": f"2026-10-{d:02d}", "label": f"M{d}", "type": "risk"} for d in range(2, 8)]
        hints = _sanitize_chart_hints(levels, markers, _frame())
        assert len(hints["levels"]) == 3
        assert len(hints["markers"]) == 3

    def test_empty_frame_returns_empty(self):
        hints = _sanitize_chart_hints([{"price": 1, "label": "x"}], [], pd.DataFrame())
        assert hints == {"levels": [], "markers": []}


class TestSplitAnnotationJson:
    def test_splits_tail_json(self):
        text = '正文内容…\n\nANNOTATION_JSON={"levels": [], "markers": []}'
        body, data = _split_annotation_json(text)
        assert body == "正文内容…"
        assert data == {"levels": [], "markers": []}

    def test_no_mark_returns_original(self):
        body, data = _split_annotation_json("纯文本")
        assert body == "纯文本"
        assert data is None

    def test_broken_json_returns_original(self):
        text = "正文\nANNOTATION_JSON={broken"
        body, data = _split_annotation_json(text)
        assert body == text
        assert data is None
