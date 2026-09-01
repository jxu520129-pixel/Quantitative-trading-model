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
import os
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
    # 优先读环境变量（docker-compose 用 env_file 注入，容器内无 .env 文件），fallback 读 .env 文件
    token = os.environ.get("TUSHARE_TOKEN", "")
    url = os.environ.get("TUSHARE_API_URL", "")
    if token and url:
        return token, url
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
    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as error:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(1.5 * attempt)
            continue
        if data.get("code") != 0:
            # 限流/业务错误也退避重试（如「请求速度过快」），最终失败才抛出
            if attempt == 5:
                raise RuntimeError(f"{api_name} 返回错误：{data.get('msg')} params={params}")
            time.sleep(2.0 * attempt)
            continue
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
    parser.add_argument("--workers", type=int, default=4, help="并发拉取天数（1=串行；过高可能触发代理限流）")
    args = parser.parse_args()

    token, url = _load_env()
    if not token or not url:
        print("缺少 TUSHARE_TOKEN / TUSHARE_API_URL，请检查 .env", file=sys.stderr)
        sys.exit(1)

    start_s = args.start.replace("-", "")
    end_s = args.end.replace("-", "")
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        DROP TABLE IF EXISTS stock_basic;
        DROP TABLE IF EXISTS stock_industry;
        DROP TABLE IF EXISTS daily_bars;
        CREATE TABLE stock_basic (
            code TEXT PRIMARY KEY, name TEXT NOT NULL, security_type TEXT NOT NULL,
            board TEXT NOT NULL DEFAULT 'MAIN', is_st INTEGER NOT NULL DEFAULT 0,
            is_delisted INTEGER NOT NULL DEFAULT 0, is_suspended INTEGER NOT NULL DEFAULT 0,
            list_date TEXT, delist_date TEXT, updated_at TEXT NOT NULL
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

    # 1) 股票列表：拉取上市(L) + 退市(D)股，纳入历史退市股以修复幸存者偏差
    stock_rows, industry_rows = [], []
    seen = set()
    for status in ("L", "D"):
        fields, items = _call("stock_basic", token, url, {"list_status": status},
                              "ts_code,symbol,name,list_date,delist_date,industry")
        col = {f: i for i, f in enumerate(fields)}
        for it in items:
            code = it[col["ts_code"]].split(".")[0]
            if code.startswith(_BJ_PREFIXES) or code in seen:
                continue
            seen.add(code)
            name = it[col["name"]]
            list_date = it[col["list_date"]]
            delist_date = it[col["delist_date"]] if "delist_date" in col else None
            industry = (it[col["industry"]] or "").strip()
            is_delisted = 1 if delist_date else 0
            stock_rows.append((code, name, "STOCK", _board_of(code), _is_st(name), is_delisted, 0, list_date, delist_date, now))
            if industry:
                industry_rows.append((code, industry, now))
    conn.executemany(
        "INSERT INTO stock_basic(code,name,security_type,board,is_st,is_delisted,is_suspended,list_date,delist_date,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)", stock_rows)
    conn.executemany(
        "INSERT INTO stock_industry(code,industry,updated_at) VALUES(?,?,?)", industry_rows)
    conn.commit()
    n_delisted = sum(1 for r in stock_rows if r[5])
    print(f"证券池：{len(stock_rows)} 只（含退市 {n_delisted} 只，行业映射 {len(industry_rows)} 只）", flush=True)

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
    # 退市股在最后 3 个交易日已无 adj_factor，这里按「退市日前最后一个交易日」补拉其复权基准
    delisted = [(r[0], r[8]) for r in stock_rows if r[5]]
    if delisted:
        code_delist = {c: dl for c, dl in delisted}
        last_day: dict[str, str] = {}
        for code, dl in delisted:
            prev = None
            for d in trade_days:
                if d <= dl:
                    prev = d
                else:
                    break
            if prev:
                last_day[code] = prev
        by_day: dict[str, list[str]] = {}
        for code, day in last_day.items():
            by_day.setdefault(day, []).append(code)
        for day, codes in by_day.items():
            try:
                _, adj_items = _call("adj_factor", token, url, {"trade_date": day}, "ts_code,adj_factor")
                for it in adj_items:
                    c = it[0].split(".")[0]
                    if c in codes and c not in latest_adj:
                        latest_adj[c] = float(it[1])
            except Exception:
                pass  # 退市日无 adj_factor 数据，该退市股退化为不复权
    print(f"前复权基准：{len(latest_adj)} 只", flush=True)

    # 4) 逐日拉 daily + adj_factor，当场前复权并流式写库（并发预取，按日期顺序消费）
    prev_close_cache: dict[str, float] = {}
    total_bars = 0
    t0 = time.time()

    def _fetch_day(d: str):
        _, daily_items = _call("daily", token, url, {"trade_date": d},
                               "ts_code,open,high,low,close,vol,amount")
        _, adj_items = _call("adj_factor", token, url, {"trade_date": d}, "ts_code,adj_factor")
        return daily_items, adj_items

    def _process_day(d: str, daily_items, adj_items) -> int:
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
        return len(bar_rows)

    workers = max(1, args.workers)
    if workers > 1 and len(trade_days) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: dict[str, object] = {}
            for d in trade_days[:workers]:
                pending[d] = pool.submit(_fetch_day, d)
            next_i = workers
            for idx, d in enumerate(trade_days, 1):
                daily_items, adj_items = pending[d].result()
                del pending[d]
                if next_i < len(trade_days):
                    nd = trade_days[next_i]
                    pending[nd] = pool.submit(_fetch_day, nd)
                    next_i += 1
                total_bars += _process_day(d, daily_items, adj_items)
                if idx % 20 == 0:
                    conn.commit()
                    print(f"  进度 {idx}/{len(trade_days)}，累计 {total_bars} 条，用时 {time.time()-t0:.0f}s", flush=True)
    else:
        for idx, d in enumerate(trade_days, 1):
            daily_items, adj_items = _fetch_day(d)
            total_bars += _process_day(d, daily_items, adj_items)
            if idx % 20 == 0:
                conn.commit()
                print(f"  进度 {idx}/{len(trade_days)}，累计 {total_bars} 条，用时 {time.time()-t0:.0f}s", flush=True)
    conn.commit()
    print(f"完成：{total_bars} 条日线，数据库 {db_path}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
