"""
本地籌碼快取模組 — 透過 FinMind API 更新三大法人 / 融資融券的本地 CSV 快取

用法:
    from chip_cache import ChipCache

    cache = ChipCache()

    # 取得單檔三大法人買賣超（自動快取/更新）
    rows = cache.get_institutional("2330")

    # 取得單檔融資融券（自動快取/更新）
    rows = cache.get_margin("2330")

    # 批次更新所有個股
    cache.update_all()
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
INST_DIR = BASE_DIR / "data" / "institutional"
MARGIN_DIR = BASE_DIR / "data" / "margin"
META_PATH = BASE_DIR / "data" / "chip_cache_meta.json"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
ENV_PATH = BASE_DIR / ".env.local"

INST_FIELDS = ["date", "investor_name", "buy", "sell"]
MARGIN_FIELDS = [
    "date",
    "margin_purchase_buy", "margin_purchase_sell",
    "margin_purchase_cash_repayment",
    "margin_purchase_yesterday_balance", "margin_purchase_today_balance",
    "short_sale_buy", "short_sale_sell",
    "short_sale_cash_repayment",
    "short_sale_yesterday_balance", "short_sale_today_balance",
]

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind dataset 名稱與欄位對應（FinMind 欄位 → 本地 CSV 欄位）
DATASETS = {
    "inst": {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "field_map": {
            "date": "date",
            "name": "investor_name",
            "buy": "buy",
            "sell": "sell",
        },
    },
    "margin": {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "field_map": {
            "date": "date",
            "MarginPurchaseBuy": "margin_purchase_buy",
            "MarginPurchaseSell": "margin_purchase_sell",
            "MarginPurchaseCashRepayment": "margin_purchase_cash_repayment",
            "MarginPurchaseYesterdayBalance": "margin_purchase_yesterday_balance",
            "MarginPurchaseTodayBalance": "margin_purchase_today_balance",
            "ShortSaleBuy": "short_sale_buy",
            "ShortSaleSell": "short_sale_sell",
            "ShortSaleCashRepayment": "short_sale_cash_repayment",
            "ShortSaleYesterdayBalance": "short_sale_yesterday_balance",
            "ShortSaleTodayBalance": "short_sale_today_balance",
        },
    },
}


def _load_env():
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


def _fetch_rows(data_type: str, stock_id: str, token: str,
                after_date: str = None) -> list[dict]:
    """從 FinMind 抓某檔股票的籌碼資料，回傳已轉成本地欄位的 rows（date 升冪）。

    after_date: 只回傳此日期「之後」的資料（exclusive）。
    """
    spec = DATASETS[data_type]
    params = {
        "dataset": spec["dataset"],
        "data_id": stock_id,
        "start_date": after_date or "2020-01-01",
        "token": token,
    }
    query = urllib.parse.urlencode(params)
    url = f"{FINMIND_API_URL}?{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    if body.get("status") != 200:
        msg = body.get("msg", str(body))
        raise RuntimeError(f"FinMind API 錯誤: {msg}")

    field_map = spec["field_map"]
    rows = [{local: r.get(fm, "") for fm, local in field_map.items()}
            for r in body.get("data", [])]
    if after_date:
        rows = [r for r in rows if r["date"] > after_date]
    rows.sort(key=lambda r: r["date"])
    return rows


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ChipCache:
    def __init__(self):
        INST_DIR.mkdir(parents=True, exist_ok=True)
        MARGIN_DIR.mkdir(parents=True, exist_ok=True)
        self._api_key = _load_env()
        self._meta = _load_meta()
        self._stocks = self._load_stock_list()

    def _load_stock_list(self) -> list[str]:
        if not STOCKS_PATH.exists():
            raise FileNotFoundError(f"找不到 {STOCKS_PATH}")
        data = json.loads(STOCKS_PATH.read_text())
        return [s["stock_id"] for s in data if s.get("low_liquidity") is False]

    @property
    def stock_ids(self) -> list[str]:
        return list(self._stocks)

    # ─── 路徑 ───

    def _inst_path(self, stock_id: str) -> Path:
        return INST_DIR / f"{stock_id}.csv"

    def _margin_path(self, stock_id: str) -> Path:
        return MARGIN_DIR / f"{stock_id}.csv"

    # ─── 是否過期 ───

    def _is_stale(self, stock_id: str, data_type: str) -> bool:
        key = f"{data_type}:{stock_id}"
        last = self._meta.get(key)
        if not last:
            return True
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = last if isinstance(last, str) else last.get("end", "")
        return end_date < today

    # ─── 讀取介面 ───

    def get_institutional(self, stock_id: str, force_update: bool = False) -> list[dict]:
        """取得三大法人買賣超。回傳 list of dict (date, investor_name, buy, sell)"""
        if force_update or self._is_stale(stock_id, "inst"):
            self._update_institutional(stock_id)
        return _read_csv(self._inst_path(stock_id))

    def get_margin(self, stock_id: str, force_update: bool = False) -> list[dict]:
        """取得融資融券。回傳 list of dict (date, margin_purchase_*, short_sale_*)"""
        if force_update or self._is_stale(stock_id, "margin"):
            self._update_margin(stock_id)
        return _read_csv(self._margin_path(stock_id))

    # ─── 更新單檔 ───

    def _update_institutional(self, stock_id: str):
        csv_path = self._inst_path(stock_id)
        existing = _read_csv(csv_path)

        after_date = None
        if existing:
            after_date = existing[-1]["date"]

        new_rows = _fetch_rows("inst", stock_id, self._api_key, after_date)

        if new_rows:
            cleaned = [{k: r[k] for k in INST_FIELDS} for r in new_rows]
            all_rows = existing + cleaned
            _write_csv(csv_path, all_rows, INST_FIELDS)

        final = _read_csv(csv_path) if new_rows or not existing else existing
        meta_key = f"inst:{stock_id}"
        if final:
            self._meta[meta_key] = {"start": final[0]["date"], "end": final[-1]["date"]}
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            self._meta[meta_key] = {"start": today, "end": today}
        _save_meta(self._meta)

    def _update_margin(self, stock_id: str):
        csv_path = self._margin_path(stock_id)
        existing = _read_csv(csv_path)

        after_date = None
        if existing:
            after_date = existing[-1]["date"]

        new_rows = _fetch_rows("margin", stock_id, self._api_key, after_date)

        if new_rows:
            cleaned = [{k: r[k] for k in MARGIN_FIELDS} for r in new_rows]
            all_rows = existing + cleaned
            _write_csv(csv_path, all_rows, MARGIN_FIELDS)

        final = _read_csv(csv_path) if new_rows or not existing else existing
        meta_key = f"margin:{stock_id}"
        if final:
            self._meta[meta_key] = {"start": final[0]["date"], "end": final[-1]["date"]}
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            self._meta[meta_key] = {"start": today, "end": today}
        _save_meta(self._meta)

    # ─── 批次更新 ───

    def update_all(self, progress: bool = True, sleep: float = 1.2):
        """批次更新所有有流動性個股的籌碼資料

        sleep: 每檔（2 個 API call）間隔秒數。預設 1.2s ≈ 3,600 call/hr，
        留餘裕給 4,000 req/hr 的 rate limit。
        """
        total = len(self._stocks)
        updated = 0
        skipped = 0
        errors = []

        for i, sid in enumerate(self._stocks):
            inst_stale = self._is_stale(sid, "inst")
            margin_stale = self._is_stale(sid, "margin")

            if not inst_stale and not margin_stale:
                skipped += 1
                if progress and (i + 1) % 200 == 0:
                    print(f"  [{i+1}/{total}] 已更新 {updated}, 跳過 {skipped}, 錯誤 {len(errors)}")
                continue

            try:
                if inst_stale:
                    self._update_institutional(sid)
                if margin_stale:
                    self._update_margin(sid)
                updated += 1
            except Exception as e:
                errors.append((sid, str(e)))

            if progress and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] 已更新 {updated}, 跳過 {skipped}, 錯誤 {len(errors)}")

            time.sleep(sleep)

        if progress:
            print(f"\n完成！更新 {updated}, 跳過 {skipped}, 錯誤 {len(errors)}")
            if errors:
                print("錯誤清單（前20）:")
                for sid, err in errors[:20]:
                    print(f"  {sid}: {err}")

        return {"updated": updated, "skipped": skipped, "errors": errors}

    def info(self) -> dict:
        inst_cached = [sid for sid in self._stocks if self._inst_path(sid).exists()]
        margin_cached = [sid for sid in self._stocks if self._margin_path(sid).exists()]
        return {
            "total_stocks": len(self._stocks),
            "institutional_cached": len(inst_cached),
            "margin_cached": len(margin_cached),
        }


if __name__ == "__main__":
    cache = ChipCache()
    print("快取狀態:", json.dumps(cache.info(), ensure_ascii=False))

    print("\n抓取 2330 三大法人...")
    rows = cache.get_institutional("2330")
    print(f"共 {len(rows)} 筆，日期: {rows[0]['date']} ~ {rows[-1]['date']}")
    print("最近 5 筆:")
    for r in rows[-5:]:
        print(f"  {r['date']}  {r['investor_name']:<20} 買={r['buy']:>12}  賣={r['sell']:>12}")

    print("\n抓取 2330 融資融券...")
    rows = cache.get_margin("2330")
    print(f"共 {len(rows)} 筆，日期: {rows[0]['date']} ~ {rows[-1]['date']}")
    print("最近 3 筆:")
    for r in rows[-3:]:
        print(f"  {r['date']}  融資餘={r['margin_purchase_today_balance']:>8}  融券餘={r['short_sale_today_balance']:>8}")
