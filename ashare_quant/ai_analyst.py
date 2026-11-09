"""AI 智能分析层（自然语言问股 → 行情/指标 → 大模型生成分析）。

在已有量化系统之上新增的「AI 大模型分析」能力，求职简历中对应
「AI 大模型应用开发」这一差异化亮点。

设计：
- 复用项目既有的 LLM 配置（``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``，
  默认 DeepSeek，与 ``fundamental.py`` 一致），无需额外配置即可运行。
- 复用 ``DataService.load_bars``（历史行情，中文列名）与 ``spot_prices``（实时价）。
- 指标计算纯 pandas 实现，不引入额外依赖。

用法：
    runtime = build_runtime()
    from ashare_quant.ai_analyst import analyze_stock
    print(analyze_stock("600519", "贵州茅台", "最近一个月走势如何，能买吗", runtime))
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd
import requests

from .data.providers import spot_prices

LOG = logging.getLogger(__name__)

# 英文列名（DataService.load_bars 从 daily_bars 表读出的列名）
C_DATE, C_OPEN, C_HIGH, C_LOW, C_CLOSE = "trade_date", "open", "high", "low", "close"
C_VOL, C_AMOUNT, C_PRE = "volume", "amount", "pre_close"


def _llm_config() -> tuple[str, str, str]:
    """读取 LLM 配置，与 fundamental.py 保持一致。"""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    return api_key, base_url, model


def llm_available() -> bool:
    return bool(os.getenv("LLM_API_KEY", ""))


def ask_llm(prompt: str, system: str | None = None, temperature: float = 0.3,
            retries: int = 2, response_json: bool = False) -> str | None:
    """调用大模型生成自然语言文本，返回 None 表示未配置或失败。

    ``retries`` 控制超时/网络失败后的重试次数（D1）；``response_json=True`` 时
    强制 JSON 输出（结构化分析用）。
    """
    api_key, base_url, model = _llm_config()
    if not api_key:
        return None
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as error:  # noqa: BLE001
            if attempt == retries:
                LOG.warning("LLM 调用失败（已重试 %s 次）：%s", retries, error)
                return None
            time.sleep(1.5 * attempt)
    return None


def compute_indicators(frame: pd.DataFrame) -> dict[str, Any]:
    """从日线 DataFrame 计算基础技术指标，返回可读字典。数据不足时返回空。

    返回键：latest、close、ma5/ma10/ma20/ma60、rsi14、macd、dif、dea、
    ret_5d/ret_20d/ret_60d、high_60d、low_60d、vol_ratio、ann_vol。
    """
    if frame is None or frame.empty or C_CLOSE not in frame.columns:
        return {}
    # load_bars 返回的 frame 以 trade_date 为索引且保留同名列，先去掉索引歧义再排序
    df = frame.reset_index(drop=True).sort_values(C_DATE).reset_index(drop=True)
    close = pd.to_numeric(df[C_CLOSE], errors="coerce")
    if len(close) < 5:
        return {}

    def _sma(series: pd.Series, n: int) -> float | None:
        return round(float(series.tail(n).mean()), 3) if len(series) >= n else None

    rsi = _calc_rsi(close, 14)
    macd, dif, dea = _calc_macd(close)
    vol = pd.to_numeric(df[C_VOL], errors="coerce")

    latest_date = str(df[C_DATE].iloc[-1])[:10]
    latest_close = float(close.iloc[-1])

    return {
        "latest_date": latest_date,
        "close": round(latest_close, 3),
        "ma5": _sma(close, 5),
        "ma10": _sma(close, 10),
        "ma20": _sma(close, 20),
        "ma60": _sma(close, 60),
        "rsi14": rsi,
        "macd": macd,
        "dif": dif,
        "dea": dea,
        "ret_5d": round((latest_close / float(close.iloc[-6]) - 1) * 100, 2) if len(close) >= 6 else None,
        "ret_20d": round((latest_close / float(close.iloc[-21]) - 1) * 100, 2) if len(close) >= 21 else None,
        "ret_60d": round((latest_close / float(close.iloc[-61]) - 1) * 100, 2) if len(close) >= 61 else None,
        "high_60d": round(float(df[C_HIGH].tail(60).max()), 3),
        "low_60d": round(float(df[C_LOW].tail(60).min()), 3),
        "vol_ratio": round(float(vol.tail(5).mean() / vol.tail(60).mean()), 2) if len(vol) >= 60 and vol.tail(60).mean() > 0 else None,
        "ann_vol": round(float(close.pct_change().std() * (252 ** 0.5) * 100), 2),
    }


def _calc_rsi(close: pd.Series, n: int = 14) -> float | None:
    if len(close) < n + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if pd.notna(val) else None


def _calc_macd(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(close) < 26:
        return None, None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    return (
        round(float(macd.iloc[-1]), 3),
        round(float(dif.iloc[-1]), 3),
        round(float(dea.iloc[-1]), 3),
    )


def _fmt_indicators(ind: dict[str, Any]) -> str:
    if not ind:
        return "（历史行情数据不足，无法计算技术指标）"
    lines = [
        f"- 最新交易日：{ind.get('latest_date')}，收盘价 {ind.get('close')} 元",
        f"- 均线：MA5={ind.get('ma5')}，MA10={ind.get('ma10')}，MA20={ind.get('ma20')}，MA60={ind.get('ma60')}",
        f"- 技术指标：RSI(14)={ind.get('rsi14')}，MACD={ind.get('macd')}，DIF={ind.get('dif')}，DEA={ind.get('dea')}",
        f"- 区间涨跌幅：近5日 {ind.get('ret_5d')}%，近20日 {ind.get('ret_20d')}%，近60日 {ind.get('ret_60d')}%",
        f"- 近60日最高 {ind.get('high_60d')} 元 / 最低 {ind.get('low_60d')} 元",
        f"- 量能比（近5日/近60日均量）{ind.get('vol_ratio')}，年化波动率 {ind.get('ann_vol')}%",
    ]
    return "\n".join(lines)


def build_market_context(code: str, name: str, runtime: Any, lookback: int = 130) -> str:
    """取历史行情 + 实时价，组装成给大模型的市场上下文。"""
    data = runtime.data
    try:
        frame = data.load_bars(code, end_date=None)
    except Exception as error:  # noqa: BLE001
        LOG.warning("加载行情失败 %s：%s", code, error)
        frame = pd.DataFrame()
    if frame is not None and not frame.empty:
        frame = frame.tail(lookback)

    ind = compute_indicators(frame)

    quote_text = ""
    try:
        prices, src = spot_prices([code])
        price = prices.get(code)
        if price:
            quote_text = f"- 实时价（{src}）：{price} 元\n"
    except Exception:  # noqa: BLE001
        pass

    head = f"股票：{name}（{code}）\n{quote_text}{_fmt_indicators(ind)}"
    return head


def build_industry_context(runtime: Any, code: str) -> str:
    """A1 行业板块：个股所属行业 + 产业链上下游。失败返回降级文案。"""
    try:
        from .main_rally import get_chain_desc, get_stock_industry

        industry = get_stock_industry(runtime.database, code)
        if not industry:
            return "（行业信息缺失）"
        return f"{industry}｜产业链：{get_chain_desc(industry)}"
    except Exception as error:  # noqa: BLE001
        LOG.warning("行业上下文构建失败 %s：%s", code, error)
        return "（行业信息暂不可用）"


def build_event_context(runtime: Any, industry: str) -> str:
    """A2 事件/新闻：近期与该股行业相关的事件（含影响方向）。"""
    if not industry:
        return "（无行业信息，跳过事件）"
    try:
        from .main_rally import get_industry_events

        events = get_industry_events(runtime.database, industry)
        if not events:
            return "（近期无相关事件）"
        lines = []
        for ev in events[:6]:
            direction = str(ev.get("causal_direction") or "中性")
            title = str(ev.get("title") or "")[:42]
            mag = ev.get("magnitude") or 1
            lines.append(f"- [{direction}/{mag}星] {title}")
        return "\n".join(lines)
    except Exception as error:  # noqa: BLE001
        LOG.warning("事件上下文构建失败：%s", error)
        return "（事件数据暂不可用）"


def build_market_breadth(runtime: Any) -> str:
    """A3 大盘环境：最新交易日上涨/下跌家数 + 涨停家数（近似）。"""
    try:
        latest = runtime.database.query_one("SELECT MAX(trade_date) AS d FROM daily_bars")
        if not latest or not latest["d"]:
            return "（大盘数据暂不可用）"
        day = latest["d"]
        up = runtime.database.query_one(
            "SELECT COUNT(*) AS n FROM daily_bars WHERE trade_date=? AND pre_close>0 AND close>pre_close", (day,)
        )
        down = runtime.database.query_one(
            "SELECT COUNT(*) AS n FROM daily_bars WHERE trade_date=? AND pre_close>0 AND close<pre_close", (day,)
        )
        limit_up = runtime.database.query_one(
            "SELECT COUNT(*) AS n FROM daily_bars WHERE trade_date=? AND pre_close>0 AND close>=pre_close*1.099", (day,)
        )
        return (
            f"最新交易日 {day}：上涨 {up['n'] if up else 0} 家 / 下跌 {down['n'] if down else 0} 家，"
            f"涨停（近似）{limit_up['n'] if limit_up else 0} 家"
        )
    except Exception as error:  # noqa: BLE001
        LOG.warning("大盘环境构建失败：%s", error)
        return "（大盘数据暂不可用）"


def build_fundamental_context(code: str) -> str:
    """A4 财务面：营收/净利润同比增速。复用主升浪策略的 fetch_earnings。"""
    try:
        from .main_rally import fetch_earnings

        revenue_yoy, profit_yoy = fetch_earnings(code)
        if not revenue_yoy and not profit_yoy:
            return "（财务数据暂不可用）"
        return f"最新报告期：营收同比 {revenue_yoy:+.2f}%，净利润同比 {profit_yoy:+.2f}%"
    except Exception as error:  # noqa: BLE001
        LOG.warning("财务上下文构建失败 %s：%s", code, error)
        return "（财务数据暂不可用）"


def build_strategy_signal_context(ind: dict[str, Any]) -> str:
    """B1 策略信号（轻量）：基于技术指标的规则信号，供大模型参考。

    不做全市场扫描（太重），而是用已算好的技术指标做规则判断：
    均线多头 / 箱体突破 / 超买超卖 / 量能异动。真正的盘中策略命中仍以
    调度器扫描为准，这里只提供「当前技术形态是否符合某类策略特征」。
    """
    if not ind:
        return "（数据不足，无法判断策略信号）"
    signals: list[str] = []
    ma5, ma10, ma20 = ind.get("ma5"), ind.get("ma10"), ind.get("ma20")
    if ma5 and ma10 and ma20 and ma5 > ma10 > ma20:
        signals.append("均线多头排列（MA5>MA10>MA20，趋势偏多）")
    close = ind.get("close")
    high_60d = ind.get("high_60d")
    if close and high_60d and close >= high_60d * 0.98:
        signals.append("逼近/创近60日新高（箱体突破特征）")
    rsi = ind.get("rsi14")
    if rsi is not None:
        if rsi >= 70:
            signals.append(f"RSI={rsi} 超买（短线追高风险）")
        elif rsi <= 30:
            signals.append(f"RSI={rsi} 超卖（存在超跌反弹机会）")
    vol_ratio = ind.get("vol_ratio")
    if vol_ratio is not None and vol_ratio >= 1.5:
        signals.append(f"量能放大（近5日/60日均量 {vol_ratio} 倍，交投活跃）")
    ret_20d = ind.get("ret_20d")
    if ret_20d is not None and ret_20d >= 20:
        signals.append(f"近20日累计涨幅 {ret_20d}%（短期涨幅较大，注意回调）")
    return "；".join(signals) if signals else "（暂无明显策略信号特征）"


def build_rich_context(runtime: Any, code: str, name: str, lookback: int = 130) -> str:
    """组装多维富上下文：技术指标 + 实时价 + 行业 + 事件 + 大盘 + 财务 + 策略信号。"""
    data = runtime.data
    try:
        frame = data.load_bars(code, end_date=None)
    except Exception as error:  # noqa: BLE001
        LOG.warning("加载行情失败 %s：%s", code, error)
        frame = pd.DataFrame()
    if frame is not None and not frame.empty:
        frame = frame.tail(lookback)
    ind = compute_indicators(frame)

    quote_text = ""
    try:
        prices, src = spot_prices([code])
        price = prices.get(code)
        if price:
            quote_text = f"实时价（{src}）：{price} 元"
    except Exception:  # noqa: BLE001
        pass

    industry = ""
    try:
        from .main_rally import get_stock_industry

        industry = get_stock_industry(runtime.database, code)
    except Exception:  # noqa: BLE001
        industry = ""

    sections = [
        f"【股票】{name}（{code}）{quote_text}",
        f"【技术指标】\n{_fmt_indicators(ind)}",
        f"【行业板块】{build_industry_context(runtime, code)}",
        f"【相关事件】\n{build_event_context(runtime, industry)}",
        f"【大盘环境】{build_market_breadth(runtime)}",
        f"【财务面】{build_fundamental_context(code)}",
        f"【策略信号】{build_strategy_signal_context(ind)}",
    ]
    return "\n\n".join(sections)


SYSTEM_PROMPT = (
    "你是一名专业的 A 股量化分析助手，面向个人投资者提供客观、克制、可执行的个股分析。"
    "要求：1) 基于给出的行情、技术指标、行业、事件、财务与策略信号数据作答，不编造未提供的数据；"
    "2) 结论要给出依据（如均线、RSI、MACD、量能、行业事件、财务增速等）；"
    "3) 明确风险提示，不构成投资建议，避免绝对化表述；"
    "4) 用中文、分点、简洁作答，控制在 400 字以内。"
)

STRUCTURED_SYSTEM_PROMPT = (
    "你是一名专业的 A 股量化分析助手。请基于给定的多维数据（行情/技术指标/行业/事件/财务/策略信号）"
    "给出结构化分析。只返回 JSON 对象，字段为：score（0-100 综合评分）、trend（偏多/中性/偏空）、"
    "risk_level（低/中/高）、key_points（字符串数组，3-5 条要点）、conclusion（一段 150 字以内的结论）。"
    "必须基于给定数据、克制客观、明确风险提示、不构成投资建议。"
)


def _strip_json(content: str) -> str:
    """去除 LLM 可能包裹的 markdown 代码块。"""
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.lstrip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.rstrip("`").strip()
    return content


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def analyze_stock(code: str, name: str, question: str, runtime: Any) -> str:
    """对单只股票做 AI 分析（多维富上下文 → 大模型生成分析）。"""
    if not llm_available():
        return "未配置 LLM_API_KEY，请在 .env 中配置后重试（默认 DeepSeek）。"
    context = build_rich_context(runtime, code, name)
    prompt = (
        f"以下是某只股票的多维数据：\n{context}\n\n"
        f"请根据以上数据回答用户问题：{question or '请给出这只股票的综合分析'}"
    )
    result = ask_llm(prompt, system=SYSTEM_PROMPT)
    if result is None:
        return "大模型调用失败，请检查 LLM_API_KEY 与网络后重试。"
    return result


def analyze_stock_structured(code: str, name: str, question: str, runtime: Any) -> dict[str, Any]:
    """B3 结构化分析：返回 {score, trend, risk_level, key_points, conclusion}。"""
    if not llm_available():
        return {"error": "未配置 LLM_API_KEY，请在 .env 中配置后重试。"}
    context = build_rich_context(runtime, code, name)
    prompt = (
        f"以下是某只股票的多维数据：\n{context}\n\n"
        f"用户问题：{question or '请给出这只股票的综合分析'}\n"
        f"请据此输出结构化 JSON。"
    )
    raw = ask_llm(prompt, system=STRUCTURED_SYSTEM_PROMPT, temperature=0.2, response_json=True)
    if raw is None:
        return {"error": "大模型调用失败，请检查 LLM_API_KEY 与网络后重试。"}
    try:
        data = json.loads(_strip_json(raw))
        return {
            "score": round(_num(data.get("score")), 1),
            "trend": str(data.get("trend") or "中性"),
            "risk_level": str(data.get("risk_level") or "中"),
            "key_points": [str(p) for p in (data.get("key_points") or [])][:5],
            "conclusion": str(data.get("conclusion") or ""),
        }
    except Exception as error:  # noqa: BLE001
        LOG.warning("结构化结果解析失败：%s", error)
        return {"error": "结构化解析失败，已回退原始文本", "raw": raw}


def chat(question: str, runtime: Any, code: str | None = None, name: str | None = None) -> str:
    """AI 分析统一入口。带 code 时走个股分析，否则做自由问答。"""
    if code:
        return analyze_stock(code, name or code, question, runtime)
    if not llm_available():
        return "未配置 LLM_API_KEY，请在 .env 中配置后重试。"
    return ask_llm(question, system=SYSTEM_PROMPT) or "大模型调用失败，请稍后重试。"


def chat_with_history(question: str, runtime: Any, history: list[dict[str, str]],
                      code: str | None = None, name: str | None = None) -> str:
    """B4 多轮对话：带历史消息追问。

    ``history`` 为 ``[{"role": "user"|"assistant", "content": ...}, ...]``，
    最多保留最近 10 条，本次 ``question`` 作为新的 user 消息追加。
    """
    api_key, base_url, model = _llm_config()
    if not api_key:
        return "未配置 LLM_API_KEY，请在 .env 中配置后重试。"
    messages: list[dict[str, str]] = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages.append({"role": "user", "content": question})
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.3},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as error:  # noqa: BLE001
        LOG.warning("多轮对话 LLM 调用失败：%s", error)
        return "大模型调用失败，请稍后重试。"
