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

import logging
import os
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


def ask_llm(prompt: str, system: str | None = None, temperature: float = 0.3) -> str | None:
    """调用大模型生成自然语言文本，返回 None 表示未配置或失败。"""
    api_key, base_url, model = _llm_config()
    if not api_key:
        return None
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as error:  # noqa: BLE001
        LOG.warning("LLM 调用失败：%s", error)
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


SYSTEM_PROMPT = (
    "你是一名专业的 A 股量化分析助手，面向个人投资者提供客观、克制、可执行的个股分析。"
    "要求：1) 基于给出的行情与技术指标数据作答，不编造未提供的数据；"
    "2) 结论要给出依据（如均线、RSI、MACD、量能等）；"
    "3) 明确风险提示，不构成投资建议，避免绝对化表述；"
    "4) 用中文、分点、简洁作答，控制在 400 字以内。"
)


def analyze_stock(code: str, name: str, question: str, runtime: Any) -> str:
    """对单只股票做 AI 分析：取数据 → 组装上下文 → 大模型生成分析。"""
    if not llm_available():
        return "未配置 LLM_API_KEY，请在 .env 中配置后重试（默认 DeepSeek）。"
    context = build_market_context(code, name, runtime)
    prompt = (
        f"以下是某只股票的最新行情与技术指标数据：\n{context}\n\n"
        f"请根据以上数据回答用户问题：{question or '请给出这只股票的综合技术面分析'}"
    )
    result = ask_llm(prompt, system=SYSTEM_PROMPT)
    if result is None:
        return "大模型调用失败，请检查 LLM_API_KEY 与网络后重试。"
    return result


def chat(question: str, runtime: Any, code: str | None = None, name: str | None = None) -> str:
    """AI 分析统一入口。带 code 时走个股分析，否则做自由问答。"""
    if code:
        return analyze_stock(code, name or code, question, runtime)
    if not llm_available():
        return "未配置 LLM_API_KEY，请在 .env 中配置后重试。"
    return ask_llm(question, system=SYSTEM_PROMPT) or "大模型调用失败，请稍后重试。"
