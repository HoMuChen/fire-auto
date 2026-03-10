"""
本地股價快取模組 — 透過 FinMind API 更新本地 CSV

用法:
    from stock_cache import StockCache

    cache = StockCache()

    # 取得單檔股票價格（讀本地 CSV）
    rows = cache.get("2330")

    # 更新今天的股價（一次抓全市場，盤中/盤後皆可）
    cache.update_today()

    # 補抓多天（例如補 2/27 ~ 3/2）
    cache.update_range("2026-02-27", "2026-03-02")

    # 更新單檔股票（指定日期範圍）
    cache.update_one("2330", start_date="2020-01-01")
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

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind 欄位 → 本地 CSV 欄位
FINMIND_FIELD_MAP = {
    "date": "date",
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
    "Trading_money": "trading_money",
    "Trading_turnover": "trading_turnover",
    "spread": "spread",
}


def _load_token():
    """從 .env.local 讀取 FINMIND_API_TOKEN"""
    key = os.environ.get("FINMIND_API_TOKEN")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("FINMIND_API_TOKEN="):
                return line.split("=", 1)[1]
    raise RuntimeError("找不到 FINMIND_API_TOKEN，請確認 .env.local")


def _load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def _save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _finmind_request(token, dataset, start_date, end_date=None, data_id=None):
    """呼叫 FinMind API，回傳 data 陣列"""
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "token": token,
    }
    if end_date:
        params["end_date"] = end_date
    if data_id:
        params["data_id"] = data_id

    query = urllib.parse.urlencode(params)
    url = f"{FINMIND_API_URL}?{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    if body.get("status") != 200:
        msg = body.get("msg", str(body))
        raise RuntimeError(f"FinMind API 錯誤: {msg}")

    return body.get("data", [])


def _map_row(finmind_row):
    """將 FinMind 回傳的一筆轉成本地 CSV 格式"""
    return {csv_field: finmind_row.get(fm_field, "")
            for fm_field, csv_field in FINMIND_FIELD_MAP.items()}


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


def _upsert_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """將 new_rows 合併進 existing（同日期覆蓋，新日期追加），回傳排序後的完整列表"""
    by_date = {r["date"]: r for r in existing}
    for r in new_rows:
        by_date[r["date"]] = r
    return sorted(by_date.values(), key=lambda r: r["date"])


class StockCache:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._token = _load_token()
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

    def get(self, stock_id: str) -> list[dict]:
        """讀取本地 CSV（不自動更新，用 update_today() 更新）"""
        return _read_csv(self._csv_path(stock_id))

    def update_today(self, progress=True):
        """
        抓今天全市場股價，更新所有本地 CSV。
        盤中呼叫會拿到盤中即時價，盤後呼叫拿到收盤價。
        同日多次呼叫安全（upsert 覆蓋）。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        self._update_by_date(today, today, progress)

    def update_range(self, start_date, end_date=None, progress=True):
        """
        補抓指定日期範圍的全市場股價。
        一天一個 API call，每天的回應包含所有股票。
        """
        if end_date is None:
            end_date = start_date
        self._update_by_date(start_date, end_date, progress)

    def _update_by_date(self, start_date, end_date, progress=True):
        """用日期抓全市場，逐天呼叫 API（全市場一天 ~46,000 筆會觸及 API 回傳上限）"""
        # 產生日期列表（逐天）
        d = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        while d <= d_end:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)

        if progress:
            print(f"從 FinMind 抓取 {start_date} ~ {end_date} 全市場股價（{len(dates)} 天）...")

        total_rows = 0
        total_updated = 0
        known_stocks = set(self._stocks)

        for day in dates:
            data = _finmind_request(
                self._token, "TaiwanStockPrice",
                start_date=day, end_date=day
            )

            if not data:
                continue

            total_rows += len(data)

            # 按 stock_id 分組
            by_stock = {}
            for row in data:
                sid = row.get("stock_id", "")
                if sid:
                    by_stock.setdefault(sid, []).append(_map_row(row))

            # 更新每檔的 CSV
            for sid, new_rows in by_stock.items():
                if sid not in known_stocks:
                    continue
                csv_path = self._csv_path(sid)
                existing = _read_csv(csv_path)
                merged = _upsert_rows(existing, new_rows)
                _write_csv(csv_path, merged)

                if merged:
                    self._meta[sid] = {"start": merged[0]["date"], "end": merged[-1]["date"]}
                total_updated += 1

            if progress:
                print(f"  {day}：{len(data)} 筆，{len(by_stock)} 檔")

        _save_meta(self._meta)
        if progress:
            trading_days = sum(1 for d in dates if total_rows > 0 or True)
            print(f"  完成：共 {total_rows} 筆，更新 {total_updated} 檔次 CSV")

    def update_one(self, stock_id, start_date=None, end_date=None, progress=True):
        """更新單檔股票（用 data_id 指定）"""
        if start_date is None:
            # 從本地最後一天的隔天開始
            existing = _read_csv(self._csv_path(stock_id))
            if existing:
                last = existing[-1]["date"]
                start_date = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "2020-01-01"

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if progress:
            print(f"更新 {stock_id}: {start_date} ~ {end_date}...")

        data = _finmind_request(
            self._token, "TaiwanStockPrice",
            start_date=start_date, end_date=end_date, data_id=stock_id
        )

        if not data:
            if progress:
                print("  無新資料")
            return

        new_rows = [_map_row(r) for r in data]
        csv_path = self._csv_path(stock_id)
        existing = _read_csv(csv_path)
        merged = _upsert_rows(existing, new_rows)
        _write_csv(csv_path, merged)

        if merged:
            self._meta[stock_id] = {"start": merged[0]["date"], "end": merged[-1]["date"]}
            _save_meta(self._meta)

        if progress:
            print(f"  新增 {len(new_rows)} 筆，共 {len(merged)} 筆 ({merged[0]['date']} ~ {merged[-1]['date']})")

    def info(self) -> dict:
        """快取狀態摘要"""
        cached = [sid for sid in self._stocks if self._csv_path(sid).exists()]
        today = datetime.now().strftime("%Y-%m-%d")
        stale = []
        for sid in cached:
            m = self._meta.get(sid, {})
            end = m.get("end", "") if isinstance(m, dict) else m
            if end < today:
                stale.append(sid)
        return {
            "total_stocks": len(self._stocks),
            "cached": len(cached),
            "up_to_date": len(cached) - len(stale),
            "stale": len(stale),
            "not_cached": len(self._stocks) - len(cached),
        }


if __name__ == "__main__":
    import sys

    cache = StockCache()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "today":
            cache.update_today()
        elif cmd == "range" and len(sys.argv) >= 3:
            end = sys.argv[3] if len(sys.argv) > 3 else None
            cache.update_range(sys.argv[2], end)
        elif cmd == "one" and len(sys.argv) >= 3:
            start = sys.argv[3] if len(sys.argv) > 3 else None
            cache.update_one(sys.argv[2], start_date=start)
        elif cmd == "info":
            print(json.dumps(cache.info(), ensure_ascii=False, indent=2))
        else:
            print("用法:")
            print("  python3 stock_cache.py today              # 更新今天全市場")
            print("  python3 stock_cache.py range 2026-02-27 2026-03-02  # 補抓日期範圍")
            print("  python3 stock_cache.py one 2330            # 更新單檔（增量）")
            print("  python3 stock_cache.py one 2330 2020-01-01 # 更新單檔（指定起始）")
            print("  python3 stock_cache.py info                # 快取狀態")
    else:
        print("快取狀態:")
        print(json.dumps(cache.info(), ensure_ascii=False, indent=2))
