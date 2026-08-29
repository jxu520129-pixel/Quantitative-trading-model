"""事件相似度检索与历史表现回溯（技术方案 Phase 3 MVP）。

用字符 n-gram TF-IDF 在本地事件库中检索相似历史事件，并用日线数据
回测相似事件发生后 5/10/20/60 个交易日受影响行业组合的平均收益，
对应技术方案 §13 事件研究与 §17.2 盘前「相似事件→历史表现」输出。
50 年全球事件库尚未建立时，相似度检索范围以本地 events 表为准。
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .database import Database


_HORIZONS = (5, 10, 20, 60)
_STOP_CHARS = re.compile(r"[\s，。、；：！？,.;:!?\-—()（）\[\]【】\"'“”‘’%+%]")


def _tokenize(text: str) -> list[str]:
    """字符 2/3-gram 分词：无需中文分词依赖，对新闻标题足够有效。"""
    text = _STOP_CHARS.sub("", str(text or ""))
    grams: list[str] = []
    for n in (2, 3):
        grams += [text[i:i + n] for i in range(len(text) - n + 1)]
    return grams


def _tfidf_matrix(docs: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """构建 L2 归一化的 TF-IDF 矩阵（子线性 TF）。返回 (矩阵, 词表)。"""
    vocab: dict[str, int] = {}
    rows = []
    for doc in docs:
        counts: dict[int, float] = {}
        for gram in _tokenize(doc):
            index = vocab.setdefault(gram, len(vocab))
            counts[index] = counts.get(index, 0.0) + 1.0
        rows.append({i: 1.0 + np.log(c) for i, c in counts.items()})

    n_docs = max(1, len(docs))
    df = np.zeros(len(vocab))
    for row in rows:
        for i in row:
            df[i] += 1.0
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    matrix = np.zeros((len(docs), len(vocab)))
    for r, row in enumerate(rows):
        for i, tf in row.items():
            matrix[r, i] = tf * idf[i]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocab


def _vectorize(text: str, vocab: dict[str, int], idf: np.ndarray, width: int) -> np.ndarray:
    vec = np.zeros(width)
    counts: dict[int, float] = {}
    for gram in _tokenize(text):
        index = vocab.get(gram)
        if index is not None:
            counts[index] = counts.get(index, 0.0) + 1.0
    for i, tf in counts.items():
        vec[i] = (1.0 + np.log(tf)) * idf[i]
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _load_events(database: Database, limit: int = 2000) -> list[dict[str, Any]]:
    return database.query_all(
        "SELECT id,title,summary,event_time,created_at,event_type,magnitude,"
        "causal_direction,affected_industries,persistence FROM events "
        "ORDER BY created_at DESC LIMIT ?", (limit,)
    )


def similar_events(database: Database, seed_title: str, top_k: int = 3) -> list[dict[str, Any]]:
    """在事件库中检索与 seed_title 最相似的历史事件（含相似度）。"""
    events = _load_events(database)
    if not events:
        return []
    docs = [f"{e['title']} {e['summary'] or ''}" for e in events]
    matrix, vocab = _tfidf_matrix(docs)
    width = matrix.shape[1]
    # 重建 idf 用于查询向量加权（与 _tfidf_matrix 保持一致）
    df = np.maximum((matrix > 0).sum(axis=0), 1.0)
    idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
    query = _vectorize(seed_title, vocab, idf, width)
    scores = matrix @ query
    order = np.argsort(-scores)
    results: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    seed_norm = _STOP_CHARS.sub("", str(seed_title or ""))
    for index in order[: top_k * 3]:
        event = events[int(index)]
        score = float(scores[int(index)])
        event_norm = _STOP_CHARS.sub("", str(event["title"] or ""))
        if score <= 0.05 or event_norm == seed_norm or event["title"] in seen_titles:
            continue
        seen_titles.add(event["title"])
        results.append({**event, "similarity": round(score, 4)})
        if len(results) >= top_k:
            break
    return results


def industries_of(event: dict[str, Any]) -> list[str]:
    return [item.strip() for item in (event.get("affected_industries") or "").split(",") if item.strip()]


def industry_forward_returns(
    database: Database, industries: list[str], as_of_date: str,
    horizons: tuple[int, ...] = _HORIZONS, max_codes: int = 60,
) -> dict[str, Any] | None:
    """事件日后受影响行业组合在 horizon 个交易日内的平均收益（等权）。"""
    if not industries or not as_of_date:
        return None
    clauses = " OR ".join(["industry LIKE ?"] * len(industries))
    params = [f"%{name}%" for name in industries]
    rows = database.query_all(f"SELECT code FROM stock_industry WHERE {clauses}", params)
    if not rows:
        return None
    codes = [row["code"] for row in rows][:max_codes]
    as_of = str(as_of_date)[:10]
    returns: dict[int, list[float]] = {h: [] for h in horizons}
    for code in codes:
        bars = database.query_all(
            "SELECT trade_date,close FROM daily_bars WHERE code=? AND trade_date>=? ORDER BY trade_date",
            (code, as_of),
        )
        if len(bars) < 2:
            continue
        base = float(bars[0]["close"])
        if base <= 0:
            continue
        for h in horizons:
            if len(bars) > h:
                returns[h].append(float(bars[h]["close"]) / base - 1.0)
    if not any(returns.values()):
        return None
    return {
        "n": max(len(v) for v in returns.values()),
        **{f"{h}d": (round(float(np.mean(v)), 4) if v else None) for h, v in returns.items()},
    }


def similar_event_review(
    database: Database, seed_titles: list[str], top_k: int = 3, max_seeds: int = 4,
) -> list[dict[str, Any]]:
    """对盘前驱动事件批量生成「相似事件→历史表现」回顾。"""
    out: list[dict[str, Any]] = []
    for title in seed_titles[:max_seeds]:
        if not title:
            continue
        sims = similar_events(database, title, top_k=top_k)
        for sim in sims:
            event_date = str(sim.get("event_time") or sim.get("created_at") or "")[:10]
            sim["performance"] = industry_forward_returns(database, industries_of(sim), event_date)
        if sims:
            out.append({"seed": title, "similar": sims})
    return out


def format_similarity_section_html(reviews: list[dict[str, Any]], esc: Any = None) -> str:
    """相似历史事件回顾的 HTML 片段（供 main_rally 报告复用样式）。"""
    if not reviews:
        return ""
    import html as _html

    esc = esc or _html.escape
    parts = ['<h4 style="margin:18px 0 4px;padding-left:9px;border-left:4px solid #8e44ad;font-size:15px;color:#2c3e50;">'
             '六、相似历史事件回顾（TF-IDF 相似度 → 受影响行业后续收益）</h4>']
    for review in reviews:
        parts.append(f'<p style="margin:8px 0 3px;font-size:13px;color:#2c3e50;"><b>当前事件：</b>{esc(str(review["seed"])[:50])}</p>')
        parts.append('<table border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;width:100%;font-size:12.5px;">'
                     '<tr style="background:#34495e;">'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">相似度</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">历史事件</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">方向</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">受影响行业</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">后续5日</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">后续10日</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">后续20日</th>'
                     '<th style="color:#fff;text-align:left;font-weight:600;padding:5px 9px;">后续60日</th></tr>')
        for i, sim in enumerate(review["similar"], 1):
            perf = sim.get("performance") or {}
            direction = str(sim.get("causal_direction") or "-")
            color = "#c0392b" if direction == "正" else ("#1e8449" if direction == "负" else "#666")

            def _fmt(v: Any) -> str:
                if v is None:
                    return '<span style="color:#aaa;">样本不足</span>'
                pct_color = "#c0392b" if v >= 0 else "#1e8449"
                return f'<span style="color:{pct_color};font-weight:600;">{v:+.2%}</span>'

            bg = ' style="background:#f8f9fa;"' if i % 2 == 0 else ''
            parts.append(
                f'<tr{bg}><td style="padding:5px 9px;color:#8e44ad;font-weight:600;">{sim["similarity"]:.2f}</td>'
                f'<td style="padding:5px 9px;">{esc(str(sim["title"])[:40])}</td>'
                f'<td style="padding:5px 9px;color:{color};">{esc(direction)}</td>'
                f'<td style="padding:5px 9px;color:#666;">{esc(str(sim.get("affected_industries") or "-")[:24])}</td>'
                f'<td style="padding:5px 9px;">{_fmt(perf.get("5d"))}</td>'
                f'<td style="padding:5px 9px;">{_fmt(perf.get("10d"))}</td>'
                f'<td style="padding:5px 9px;">{_fmt(perf.get("20d"))}</td>'
                f'<td style="padding:5px 9px;">{_fmt(perf.get("60d"))}</td></tr>'
            )
        parts.append('</table>')
    return "".join(parts)
