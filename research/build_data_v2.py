"""重建研究用資料 v2（2026-10-20）：去倖存者偏誤、還原權息、拉長到 2010。

輸出（data/v2/，gitignore，不影響實盤用的 data/stock_prices/）：
  universe.csv        所有普通股：現存（上市/上櫃/興櫃）+ 2010 年後下市；欄位 stock_id,name,industry,type,delist_date
  raw/{id}.parquet    原始價 TaiwanStockPrice（算流動性、漲跌停）
  adj/{id}.parquet    還原權息價 TaiwanStockPriceAdj（算報酬）
                      FinMind 對已不在 TaiwanStockInfo 的下市股不給 adj → 用除權息結果 + 減資參考價自建
                      （同法重建現存股與 FinMind adj 誤差中位 <0.2%）；事件存 events/{div,cr}_{id}.parquet
欄位統一：date,open,high,low,close,volume,money,turnover

用法：python3 research/build_data_v2.py            # 斷點續抓（已存在的檔案跳過）
      python3 research/build_data_v2.py --rate 80  # 每分鐘請求上限（預設 80，留額度給 cron）
"""
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data/v2"
URL = "https://api.finmindtrade.com/api/v4/data"
START = "2010-01-01"
NON_STOCK = {"ETF", "ETN", "受益證券", "存託憑證", "大盤", "Index", "上櫃指數股票型基金(ETF)", "指數投資證券(ETN)"}
TOKEN = [l.split("=", 1)[1].strip() for l in open(BASE / ".env.local") if l.startswith("FINMIND_API_TOKEN")][0]
COLS = {"Trading_Volume": "volume", "Trading_money": "money", "max": "high", "min": "low", "Trading_turnover": "turnover"}


class RateLimiter:
    def __init__(self, per_min):
        self.gap, self.lock, self.next = 60.0 / per_min, threading.Lock(), time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0, t - now))


def api(lim, **p):
    for k in range(5):
        lim.wait()
        try:
            r = requests.get(URL, params=dict(token=TOKEN, **p), timeout=120)
            j = r.json()
            if j.get("msg") == "success":
                return pd.DataFrame(j.get("data", []))
            if r.status_code in (402, 429) or "limit" in str(j.get("msg", "")).lower():
                print(f"  rate limited: {j.get('msg')} → 等 5 分鐘", flush=True)
                time.sleep(300)
            else:
                print(f"  {p.get('data_id')} {p['dataset']}: {j.get('msg')}", flush=True)
                time.sleep(5 * (k + 1))
        except Exception as e:  # noqa: BLE001
            print(f"  {p.get('data_id')} 例外 {e}", flush=True)
            time.sleep(5 * (k + 1))
    return None


def build_universe(lim):
    info = api(lim, dataset="TaiwanStockInfo")
    dl = api(lim, dataset="TaiwanStockDelisting")
    is_stock = info.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~info.industry_category.isin(NON_STOCK)
    cur = info[is_stock].copy()
    # 一檔可能有多列產業；取第一個非「電子工業」的產業（電子工業是雜項類）
    cur["rank"] = (cur.industry_category == "電子工業").astype(int)
    cur = cur.sort_values(["stock_id", "rank"]).drop_duplicates("stock_id")
    u = cur[["stock_id", "stock_name", "industry_category", "type"]].rename(
        columns={"stock_name": "name", "industry_category": "industry"})
    u["delist_date"] = None
    d = dl[dl.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~dl.stock_id.str.match(r"91\d\d") & (dl.date >= START)]
    d = d.sort_values("date").drop_duplicates("stock_id", keep="last")
    d = d[~d.stock_id.isin(u.stock_id)]          # 代號已被新公司沿用者以現存為準
    d = pd.DataFrame({"stock_id": d.stock_id, "name": d.stock_name, "industry": None, "type": "delisted",
                      "delist_date": d.date})
    u = pd.concat([u, d], ignore_index=True).sort_values("stock_id")
    u.to_csv(OUT / "universe.csv", index=False)
    print(f"universe：現存 {len(cur)}（{cur.type.value_counts().to_dict()}）+ 2010 後下市 {len(d)} = {len(u)} 檔", flush=True)
    return u


def fetch_one(lim, sid, kind):
    f = OUT / kind / f"{sid}.parquet"
    if f.exists():
        return "skip"
    ds = "TaiwanStockPrice" if kind == "raw" else "TaiwanStockPriceAdj"
    df = api(lim, dataset=ds, data_id=sid, start_date=START)
    if df is None:
        return "fail"
    if len(df):
        df = df.rename(columns=COLS)[["date", "open", "high", "low", "close", "volume", "money", "turnover"]]
        df["date"] = pd.to_datetime(df.date)
    df.to_parquet(f, index=False)   # 空表也存，避免重抓
    return "ok" if len(df) else "empty"


def self_adjust(lim, sid):
    """後復權：事件日之前的價格 × Π(事件後參考價 / 事件前收盤)。"""
    ev = []
    for ds, tag in (("TaiwanStockDividendResult", "div"), ("TaiwanStockCapitalReductionReferencePrice", "cr")):
        f = OUT / "events" / f"{tag}_{sid}.parquet"
        if not f.exists():
            df = api(lim, dataset=ds, data_id=sid, start_date=START)
            if df is None:
                return "fail"
            df.to_parquet(f)
        df = pd.read_parquet(f)
        if not len(df):
            continue
        if tag == "div":
            ev += [(pd.Timestamp(d), a / b) for d, a, b in zip(df.date, df.after_price, df.before_price) if a > 0 and b > 0]
        else:
            ev += [(pd.Timestamp(d), a / b) for d, b, a in
                   zip(df.date, df.ClosingPriceonTheLastTradingDay, df.PostReductionReferencePrice) if a > 0 and b > 0]
    r = pd.read_parquet(OUT / "raw" / f"{sid}.parquet")
    if not len(r):
        return "empty"
    k = pd.Series(1.0, index=r.index)
    for d, x in ev:
        k[r.date < d] *= x
    for c in ("open", "high", "low", "close"):
        r[c] = r[c] * k
    r.to_parquet(OUT / "adj" / f"{sid}.parquet", index=False)
    return "self_adj"


def main():
    rate = int(sys.argv[sys.argv.index("--rate") + 1]) if "--rate" in sys.argv else 80
    lim = RateLimiter(rate)
    for k in ("raw", "adj"):
        (OUT / k).mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(OUT / "universe.csv", dtype=str) if (OUT / "universe.csv").exists() else build_universe(lim)
    jobs = [(s, k) for k in ("raw", "adj") for s in u.stock_id]
    print(f"待處理 {len(jobs)} 個請求（每分鐘 ≤{rate}）", flush=True)
    stats, t0 = {}, time.time()
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(lambda j: fetch_one(lim, *j), jobs)):
            stats[r] = stats.get(r, 0) + 1
            if i % 200 == 0:
                print(f"[{i}/{len(jobs)}] {stats} {(time.time()-t0)/60:.1f} 分", flush=True)
    print("完成", stats, flush=True)
    (OUT / "events").mkdir(exist_ok=True)
    todo = [s for s in u.stock_id if len(pd.read_parquet(OUT / "adj" / f"{s}.parquet")) == 0]
    res = [self_adjust(lim, s) for s in todo]
    print(f"adj 空表自建 {len(todo)} 檔：", pd.Series(res).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
