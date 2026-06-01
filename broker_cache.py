"""
券商分點快取模組 — FinMind TaiwanStockTradingDailyReport 本地 Parquet 快取

每檔股票存一個 Parquet：data/broker/{stock_id}.parquet
欄位：date(str), securities_trader_id(str), securities_trader(str), buy(int32), sell(int32)
（已按券商加總，去除逐價位明細）

用法:
    from broker_cache import BrokerCache
    cache = BrokerCache()

    # 讀本地快取，回傳 pandas DataFrame
    df = cache.get("2330")
    df = cache.get("2330", start="2026-01-01")          # 過濾日期

    # 補抓日期範圍（全部液態股，可中斷繼續）
    cache.update_range("2021-06-30", "2026-05-29")

    # 只抓今天
    cache.update_today()

    # 更新單檔
    cache.update_one("2330")

    # 快取狀態
    cache.info()

CLI:
    python3 broker_cache.py range 2021-06-30 2026-05-29
    python3 broker_cache.py today
    python3 broker_cache.py one 2330
    python3 broker_cache.py info

速率限制：6,000 req/hr（有 Sponsor token）
補全 1,439 檔 × 1,235 天（2021-06-30 起）≈ 178 萬次 call，約 296 小時（可分多次跑）
"""

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).parent
BROKER_DIR = BASE_DIR / "data" / "broker"
META_PATH = BASE_DIR / "data" / "broker_cache_meta.json"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
PRICE_REF = BASE_DIR / "data" / "stock_prices" / "2330.csv"
ENV_PATH = BASE_DIR / ".env.local"

API_URL = "https://api.finmindtrade.com/api/v4/data"

SCHEMA = pa.schema([
    ("date", pa.string()),
    ("securities_trader_id", pa.string()),
    ("securities_trader", pa.string()),
    ("buy", pa.int32()),
    ("sell", pa.int32()),
])

# 6000/hr = 0.6s per call，留一點餘裕
SLEEP_SEC = 0.65
# 每抓 N 天就 flush 一次，避免中斷損失太多
FLUSH_EVERY = 30


def _load_token() -> str:
    val = os.environ.get("FINMIND_API_TOKEN")
    if val:
        return val
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("FINMIND_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("找不到 FINMIND_API_TOKEN，請確認 .env.local")


def _load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def _save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _load_liquid_stocks() -> list[str]:
    data = json.loads(STOCKS_PATH.read_text())
    return [s["stock_id"] for s in data if s.get("low_liquidity") is False]


def _read_trading_dates(start_date: str, end_date: str) -> list[str]:
    """從 2330 股價 CSV 取得實際交易日清單"""
    dates = []
    with open(PRICE_REF, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["date"]
            if start_date <= d <= end_date:
                dates.append(d)
    return sorted(dates)


def _fetch_day(stock_id: str, date: str, token: str) -> list[dict]:
    """抓單日單股券商分點，回傳 aggregated 列表（按券商加總）"""
    params = urllib.parse.urlencode({
        "dataset": "TaiwanStockTradingDailyReport",
        "data_id": stock_id,
        "start_date": date,
        "token": token,
    })
    req = urllib.request.Request(f"{API_URL}?{params}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    if data.get("status") != 200:
        return []

    agg: dict[str, dict] = {}
    for row in data["data"]:
        tid = row["securities_trader_id"]
        if tid not in agg:
            agg[tid] = {
                "date": date,
                "securities_trader_id": tid,
                "securities_trader": row["securities_trader"],
                "buy": 0,
                "sell": 0,
            }
        agg[tid]["buy"] += int(row["buy"])
        agg[tid]["sell"] += int(row["sell"])

    return sorted(agg.values(), key=lambda x: x["securities_trader_id"])


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "securities_trader_id", "securities_trader", "buy", "sell"])
    return pd.read_parquet(path)


def _write_parquet(path: Path, df: pd.DataFrame):
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


class BrokerCache:
    def __init__(self):
        BROKER_DIR.mkdir(parents=True, exist_ok=True)
        self._token = _load_token()
        self._meta = _load_meta()
        self._stocks = _load_liquid_stocks()

    def _path(self, stock_id: str) -> Path:
        return BROKER_DIR / f"{stock_id}.parquet"

    def get(self, stock_id: str, start: str = None, end: str = None) -> pd.DataFrame:
        """讀本地快取，回傳 DataFrame。可傳 start/end 過濾日期。"""
        filters = []
        if start:
            filters.append(("date", ">=", start))
        if end:
            filters.append(("date", "<=", end))
        path = self._path(stock_id)
        if not path.exists():
            return pd.DataFrame(columns=["date", "securities_trader_id", "securities_trader", "buy", "sell"])
        if filters:
            return pd.read_parquet(path, filters=filters)
        return pd.read_parquet(path)

    def update_today(self):
        """更新今天所有液態股的券商分點資料"""
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"更新 {today} 券商分點資料...")
        self.update_range(today, today)

    def update_one(self, stock_id: str, start_date: str = None):
        """更新單一股票（增量，從本地最後日期接續）"""
        meta = self._meta.get(stock_id, {})
        if start_date is None:
            start_date = meta.get("end", "2021-06-30")
        end_date = datetime.now().strftime("%Y-%m-%d")
        print(f"更新 {stock_id}（{start_date} ~ {end_date}）...")
        self.update_range(start_date, end_date, stock_ids=[stock_id])

    def update_range(self, start_date: str, end_date: str, stock_ids: list[str] = None):
        """
        補抓日期範圍內所有股票的券商分點資料。
        已快取的（date 在 meta [start,end] 區間內）自動跳過。
        可隨時中斷，下次繼續。
        """
        if stock_ids is None:
            stock_ids = self._stocks

        all_trading_dates = _read_trading_dates(start_date, end_date)
        if not all_trading_dates:
            print("指定區間內無交易日")
            return

        total_est = len(stock_ids) * len(all_trading_dates)
        eta_h = total_est * SLEEP_SEC / 3600
        print(f"目標：{len(stock_ids)} 檔 × {len(all_trading_dates)} 天 = 至多 {total_est:,} 次 call")
        print(f"上限時間：{eta_h:.1f} 小時（已快取的自動跳過，可隨時 Ctrl+C 中斷繼續）")
        print()

        api_calls = 0
        t0 = time.time()

        for i, stock_id in enumerate(stock_ids):
            meta = self._meta.get(stock_id, {})
            cached_start = meta.get("start", "")
            cached_end = meta.get("end", "")

            if cached_start and cached_end:
                missing = [d for d in all_trading_dates
                           if d < cached_start or d > cached_end]
            else:
                missing = list(all_trading_dates)

            if not missing:
                continue

            path = self._path(stock_id)
            existing_df = _read_parquet(path)
            existing_keys = set(
                zip(existing_df["date"], existing_df["securities_trader_id"])
            ) if not existing_df.empty else set()

            buffer: list[dict] = []

            for j, date in enumerate(sorted(missing)):
                try:
                    rows = _fetch_day(stock_id, date, self._token)
                    for r in rows:
                        key = (r["date"], r["securities_trader_id"])
                        if key not in existing_keys:
                            buffer.append(r)
                            existing_keys.add(key)
                    api_calls += 1
                    time.sleep(SLEEP_SEC)
                except KeyboardInterrupt:
                    if buffer:
                        existing_df = self._flush(path, existing_df, buffer)
                        self._update_meta(stock_id, missing[:j + 1])
                    print(f"\n中斷！已儲存至 {stock_id}/{date}，下次繼續")
                    _save_meta(self._meta)
                    return
                except Exception as e:
                    print(f"\n  ERROR {stock_id} {date}: {e}")
                    time.sleep(2)
                    api_calls += 1

                if buffer and (j + 1) % FLUSH_EVERY == 0:
                    existing_df = self._flush(path, existing_df, buffer)
                    buffer = []

            if buffer:
                self._flush(path, existing_df, buffer)
            self._update_meta(stock_id, missing)
            _save_meta(self._meta)

            if (i + 1) % 10 == 0 or i == len(stock_ids) - 1:
                elapsed = time.time() - t0
                rate = api_calls / elapsed if elapsed > 0 else 0
                avg_missing = api_calls / max(i + 1, 1)
                remaining_calls = (len(stock_ids) - i - 1) * avg_missing
                eta_sec = remaining_calls / rate if rate > 0 else 0
                print(
                    f"  [{i+1:4d}/{len(stock_ids)}]"
                    f"  已呼叫 {api_calls:,} 次"
                    f"  速率 {rate*3600:.0f}/hr"
                    f"  剩餘 {eta_sec/3600:.1f}h"
                )

        elapsed = time.time() - t0
        print(f"\n完成！共 {api_calls:,} 次 API call，耗時 {elapsed/3600:.2f} 小時")

    def _flush(self, path: Path, existing_df: pd.DataFrame, buffer: list[dict]) -> pd.DataFrame:
        """合併 existing_df + buffer，排序後寫入 Parquet，回傳合併後的 DataFrame"""
        new_df = pd.DataFrame(buffer).astype({
            "buy": "int32", "sell": "int32"
        })
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.sort_values(["date", "securities_trader_id"]).reset_index(drop=True)
        _write_parquet(path, merged)
        return merged

    def _update_meta(self, stock_id: str, fetched_dates: list[str]):
        meta = self._meta.get(stock_id, {})
        candidates = [d for d in fetched_dates if d]
        if meta.get("start"):
            candidates.append(meta["start"])
        if meta.get("end"):
            candidates.append(meta["end"])
        if candidates:
            self._meta[stock_id] = {
                "start": min(candidates),
                "end": max(candidates),
            }

    def info(self):
        cached = [sid for sid in self._stocks if self._path(sid).exists()]
        total_size = sum(self._path(sid).stat().st_size for sid in cached)
        print(f"液態股總數：{len(self._stocks)} 檔")
        print(f"已快取：    {len(cached)} 檔")
        print(f"總大小：    {total_size/1024/1024:.1f} MB")

        if self._meta:
            ends = [self._meta[sid]["end"] for sid in cached if sid in self._meta]
            starts = [self._meta[sid]["start"] for sid in cached if sid in self._meta]
            if ends:
                print(f"日期範圍：  {min(starts)} ~ {max(ends)}")

        recent = sorted(
            [(sid, self._meta[sid]["end"]) for sid in cached if sid in self._meta],
            key=lambda x: x[1], reverse=True
        )[:10]
        if recent:
            print(f"\n最新 10 檔：")
            for sid, end in recent:
                start = self._meta[sid]["start"]
                print(f"  {sid:<8} {start} ~ {end}")


if __name__ == "__main__":
    cache = BrokerCache()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "range":
        if len(sys.argv) < 4:
            print("用法：python3 broker_cache.py range <start_date> <end_date>")
            sys.exit(1)
        cache.update_range(sys.argv[2], sys.argv[3])

    elif cmd == "today":
        cache.update_today()

    elif cmd == "one":
        if len(sys.argv) < 3:
            print("用法：python3 broker_cache.py one <stock_id>")
            sys.exit(1)
        cache.update_one(sys.argv[2])

    elif cmd == "info":
        cache.info()

    else:
        print(f"未知命令：{cmd}")
        print("可用命令：range, today, one, info")
        sys.exit(1)
