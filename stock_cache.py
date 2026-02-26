"""
本地股價快取模組 — Supabase stock_prices 的本地 CSV 快取

用法:
    from stock_cache import StockCache

    cache = StockCache()

    # 取得單檔股票價格（自動快取/更新）
    rows = cache.get("2330")

    # 批次更新所有個股
    cache.update_all()
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
META_PATH = BASE_DIR / "data" / "cache_meta.json"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
ENV_PATH = BASE_DIR / ".env.local"

CSV_FIELDS = ["date", "open", "high", "low", "close", "volume", "trading_money", "trading_turnover", "spread"]

SUPABASE_URL = "https://jnikspnudxhecsqthlbo.supabase.co"
PAGE_SIZE = 1000


def _load_env():
    """從 .env.local 讀取 SUPABASE_SERVICE_ROLE_KEY"""
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
                return line.split("=", 1)[1]
    raise RuntimeError("找不到 SUPABASE_SERVICE_ROLE_KEY，請確認 .env.local")


def _load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def _save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _fetch_prices(stock_id: str, api_key: str, after_date: str = None) -> list[dict]:
    """從 Supabase 分頁抓取股價，可指定 after_date 做增量更新"""
    all_rows = []
    offset = 0
    while True:
        params = {
            "select": "date,open,high,low,close,volume,trading_money,trading_turnover,spread",
            "stock_id": f"eq.{stock_id}",
            "order": "date.asc",
            "offset": str(offset),
            "limit": str(PAGE_SIZE),
        }
        if after_date:
            params["date"] = f"gt.{after_date}"

        query = urllib.parse.urlencode(params)
        url = f"{SUPABASE_URL}/rest/v1/stock_prices?{query}"
        req = urllib.request.Request(url, headers={
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req) as resp:
            batch = json.loads(resp.read())

        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return all_rows


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class StockCache:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._api_key = _load_env()
        self._meta = _load_meta()
        self._stocks = self._load_stock_list()

    def _load_stock_list(self) -> list[str]:
        if not STOCKS_PATH.exists():
            raise FileNotFoundError(f"找不到 {STOCKS_PATH}")
        data = json.loads(STOCKS_PATH.read_text())
        return [s["stock_id"] for s in data]

    @property
    def stock_ids(self) -> list[str]:
        return list(self._stocks)

    def _csv_path(self, stock_id: str) -> Path:
        return DATA_DIR / f"{stock_id}.csv"

    def _is_stale(self, stock_id: str) -> bool:
        """快取是否需要更新（最後快取日不是今天）"""
        last = self._meta.get(stock_id)
        if not last:
            return True
        today = datetime.now().strftime("%Y-%m-%d")
        return last < today

    def get(self, stock_id: str, force_update: bool = False) -> list[dict]:
        """
        取得股價資料。快取不是最新就自動增量更新。

        回傳: list of dict，每筆含 date, open, high, low, close, volume, ...
        """
        if force_update or self._is_stale(stock_id):
            self._update_one(stock_id)
        return _read_csv(self._csv_path(stock_id))

    def _update_one(self, stock_id: str):
        """增量更新單檔股票"""
        csv_path = self._csv_path(stock_id)
        existing = _read_csv(csv_path)

        after_date = None
        if existing:
            after_date = existing[-1]["date"]  # CSV 已按日期排序

        new_rows = _fetch_prices(stock_id, self._api_key, after_date=after_date)

        if new_rows:
            all_rows = existing + [{k: r[k] for k in CSV_FIELDS} for r in new_rows]
            _write_csv(csv_path, all_rows)

        # 更新 meta：記錄快取中的最新日期
        final = _read_csv(csv_path) if new_rows or not existing else existing
        if final:
            self._meta[stock_id] = final[-1]["date"]
        else:
            self._meta[stock_id] = datetime.now().strftime("%Y-%m-%d")
        _save_meta(self._meta)

    def update_all(self, progress: bool = True):
        """批次更新所有個股快取"""
        total = len(self._stocks)
        updated = 0
        skipped = 0
        errors = []

        for i, sid in enumerate(self._stocks):
            if not self._is_stale(sid):
                skipped += 1
                if progress and (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{total}] 已跳過 {skipped}, 已更新 {updated}")
                continue
            try:
                self._update_one(sid)
                updated += 1
            except Exception as e:
                errors.append((sid, str(e)))

            if progress and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] 已更新 {updated}, 跳過 {skipped}, 錯誤 {len(errors)}")

            # 避免打太快被限流
            time.sleep(0.05)

        if progress:
            print(f"\n完成！更新 {updated}, 跳過 {skipped}, 錯誤 {len(errors)}")
            if errors:
                print("錯誤清單:")
                for sid, err in errors[:20]:
                    print(f"  {sid}: {err}")

        return {"updated": updated, "skipped": skipped, "errors": errors}

    def info(self) -> dict:
        """快取狀態摘要"""
        cached = [sid for sid in self._stocks if self._csv_path(sid).exists()]
        stale = [sid for sid in cached if self._is_stale(sid)]
        return {
            "total_stocks": len(self._stocks),
            "cached": len(cached),
            "stale": len(stale),
            "not_cached": len(self._stocks) - len(cached),
        }


if __name__ == "__main__":
    cache = StockCache()
    print("快取狀態:", json.dumps(cache.info(), ensure_ascii=False))

    # 測試：抓一檔台積電
    print("\n抓取 2330 台積電...")
    rows = cache.get("2330")
    print(f"共 {len(rows)} 筆，日期範圍: {rows[0]['date']} ~ {rows[-1]['date']}")
    print("最近 3 筆:")
    for r in rows[-3:]:
        print(f"  {r['date']}  開:{r['open']}  高:{r['high']}  低:{r['low']}  收:{r['close']}  量:{r['volume']}")
