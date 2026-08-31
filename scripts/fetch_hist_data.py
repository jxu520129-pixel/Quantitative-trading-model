"""按交易日批量拉取全市场 A 股日线（前复权）并流式写入独立历史库，供短线策略回测。

用法（在项目根目录，需已安装 requests）：
    python scripts/fetch_hist_data.py --start 2023-07-01 --end 2026-08-28

数据源：Tushare 付费代理（.env 的 TUSHARE_TOKEN / TUSHARE_API_URL）。
产出：data/ashare_quant_hist.db（stock_basic / stock_industry / daily_bars 三张表），
不污染默认的 data/ashare_quant.db 演示库。

实现：先拉最新交易日复权因子作为前复权基准，再逐日拉 daily + adj_factor，
当场前复权并流式写库，内存占用 O(单日股票数)，避免全量累积。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "ashare_quant_hist.db"

# 北交所代码前缀（策略已排除，这里也一并排除以省空间）
_BJ_PREFIXES = ("43", "83", "87", "88", "920")


def _load_env() -> tuple[str, str]:
    token, url = "", ""
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "TUSHARE_TOKEN":
                token = value
            elif key == "TUSHARE_API_URL":
                url = value
    return token, url


def _call(api_name: str, token: str, url: str, params: dict, fields: str, timeout: int = 60):
    payload = {"api_name": api_name, "token": token, "params": params, "fields": fields}
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as error:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)
            continue
        if data.get("code") != 0:
            raise RuntimeError(f"{api_name} 返回错误：{data.get('msg')} params={params}")
        d = data["data"]
        return d["fields"], d["items"]
    raise RuntimeError("unreachable")


def _board_of(code: str) -> str:
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("688", "689")):
        return "科创板"
    return "主板"


def _is_st(name: str) -> int:
    return 1 if "ST" in name.upper() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-07-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    token, url = _load_env()
    if not token or not url:
        print("缺少 TUSHARE_TOKEN / TUSHARE_API_URL，请检查 .env", file=sys.stderr)
        sys.exit(1)

    start_s = args.start.replace("-", "")
    end_s = args.end.replace("-", "")
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE stock_basic (
            code TEXT PRIMARY KEY, name TEXT NOT NULL, security_type TEXT NOT NULL,
            board TEXT NOT NULL DEFAULT 'MAIN', is_st INTEGER NOT NULL DEFAULT 0,
            is_delisted INTEGER NOT NULL DEFAULT 0, is_suspended INTEGER NOT NULL DEFAULT 0,
            list_date TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE stock_industry (
            code TEXT PRIMARY KEY, industry TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE daily_bars (
            code TEXT NOT NULL, trade_date TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL DEFAULT 0,
            amount REAL NOT NULL DEFAULT 0, pre_close REAL, source TEXT NOT NULL,
            PRIMARY KEY (code, trade_date)
        );
        CREATE INDEX idx_daily_bars_date ON daily_bars(trade_date);
        """
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) 股票列表
    fields, items = _call("stock_basic", token, url, {"list_status": "L"},
                          "ts_code,symbol,name,list_date,industry")
    col = {f: i for i, f in enumerate(fields)}
    stock_rows, industry_rows = [], []
    seen = set()
    for it in items:
        code = it[col["ts_code"]].split(".")[0]
        if code.startswith(_BJ_PREFIXES) or code in seen:
            continue
        seen.add(code)
        name = it[col["name"]]
        list_date = it[col["list_date"]]
        industry = (it[col["industry"]] or "").strip()
        stock_rows.append((code, name, "STOCK", _board_of(code), _is_st(name), 0, 0, list_date, now))
        if industry:
            industry_rows.append((code, industry, now))
    conn.executemany(
        "INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,is_suspended,list_date,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?)", stock_rows)
    conn.executemany(
        "INSERT INTO stock_industry(code,industry,updated_at) VALUES(?,?,?)", industry_rows)
    conn.commit()
    print(f"证券池：{len(stock_rows)} 只（含行业映射 {len(industry_rows)} 只）", flush=True)

    # 2) 交易日历
    _, cal_items = _call("trade_cal", token, url,
                         {"exchange": "SSE", "start_date": start_s, "end_date": end_s, "is_open": "1"},
                         "cal_date")
    trade_days = sorted({it[0] for it in cal_items})
    print(f"交易日：{len(trade_days)} 天（{trade_days[0]} ~ {trade_days[-1]}）", flush=True)

    # 3) 前复权基准：拉最后 3 个交易日的 adj_factor，取每只股票最新因子
    latest_adj: dict[str, float] = {}
    for d in reversed(trade_days[-3:]):
        _, adj_items = _call("adj_factor", token, url, {"trade_date": d}, "ts_code,adj_factor")
        for it in adj_items:
            code = it[0].split(".")[0]
            if code not in latest_adj:
                latest_adj[code] = float(it[1])
        if len(latest_adj) >= len(stock_rows) * 0.99:
            break
    print(f"前复权基准：{len(latest_adj)} 只", flush=True)

    # 4) 逐日拉 daily + adj_factor，当场前复权并流式写库
    prev_close_cache: dict[str, float] = {}
    total_bars = 0
    t0 = time.time()
    for idx, d in enumerate(trade_days, 1):
        _, daily_items = _call("daily", token, url, {"trade_date": d},
                               "ts_code,open,high,low,close,vol,amount")
        _, adj_items = _call("adj_factor", token, url, {"trade_date": d}, "ts_code,adj_factor")
        adj_today = {it[0].split(".")[0]: float(it[1]) for it in adj_items}
        bar_rows = []
        for it in daily_items:
            code = it[0].split(".")[0]
            if code.startswith(_BJ_PREFIXES) or code not in seen:
                continue
            o, h, l, c, v, amt = float(it[1]), float(it[2]), float(it[3]), float(it[4]), float(it[5]), float(it[6])
            latest = latest_adj.get(code)
            af = adj_today.get(code, latest)
            factor = (af / latest) if (latest and af) else 1.0
            qc = c * factor
            pre_close = prev_close_cache.get(code)
            bar_rows.append((code, d, o * factor, h * factor, l * factor, qc, v, amt, pre_close, "tushare"))
            prev_close_cache[code] = qc
        conn.executemany(
            "INSERT INTO daily_bars(code,trade_date,open,high,low,close,volume,amount,pre_close,source)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)", bar_rows)
        total_bars += len(bar_rows)
        if idx % 20 == 0:
            conn.commit()
            print(f"  进度 {idx}/{len(trade_days)}，累计 {total_bars} 条，用时 {time.time()-t0:.0f}s", flush=True)
    conn.commit()
    print(f"完成：{total_bars} 条日线，数据库 {db_path}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
