"""快取所有流動性個股的月營收（FinMind TaiwanStockMonthRevenue）到 data/revenue/{sid}.csv。

欄位：revenue_year, revenue_month, revenue, create_time（發布日，用於避免前視）。
"""
import csv
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE = Path(__file__).parent.parent
REV_DIR = BASE / "data" / "revenue"
STOCKS = BASE / "individual_stocks.json"
URL = "https://api.finmindtrade.com/api/v4/data"
FIELDS = ["revenue_year", "revenue_month", "revenue", "create_time"]


def token():
    for line in open(BASE / ".env.local"):
        if line.startswith("FINMIND_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no token")


def fetch(sid, tok):
    params = {"dataset": "TaiwanStockMonthRevenue", "data_id": sid,
              "start_date": "2018-01-01", "token": tok}
    url = URL + "?" + urllib.parse.urlencode(params)
    body = json.loads(urllib.request.urlopen(url, timeout=60).read())
    if body.get("status") != 200:
        raise RuntimeError(body.get("msg", "err"))
    return body.get("data", [])


def main():
    REV_DIR.mkdir(parents=True, exist_ok=True)
    tok = token()
    stocks = json.load(open(STOCKS, encoding="utf-8"))
    liquid = [s["stock_id"] for s in stocks if s.get("low_liquidity") is False]
    done = 0
    err = 0
    for i, sid in enumerate(liquid):
        out = REV_DIR / f"{sid}.csv"
        if out.exists():
            done += 1
            continue
        try:
            rows = fetch(sid, tok)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            done += 1
        except Exception as e:
            err += 1
            if err <= 10:
                print(f"  {sid}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(liquid)}] done={done} err={err}")
        time.sleep(1.0)
    print(f"完成 done={done} err={err}")


if __name__ == "__main__":
    main()
