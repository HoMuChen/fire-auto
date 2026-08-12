"""
小型台積電期貨（QFF）tick → 日盤 30 分 K 本地快取

- 資料源：FinMind TaiwanFuturesTick（逐筆），data_id=QFF
- 只取日盤：08:45:00 ~ 13:45:00
- 近月連續合約：每日取當日日盤成交量最大的 contract_date（近月），結算前自動換月
- 聚合 30 分 OHLCV（session 對齊，origin 08:45 → 每日 10 根：0845/0915/.../1315）
- 輸出：data/futures/QFF_30min_day.csv（欄位 datetime,contract_date,open,high,low,close,volume）
- 增量：從既有 CSV 最後日期接續

用法：
    python3 research/futures_cache.py                 # 從 2025-01-01（或既有最後日）補到今天
    python3 research/futures_cache.py 2025-01-01 2026-08-11
"""
import csv
import json
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, date, timedelta
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT_DIR = BASE / "data" / "futures"
OUT = OUT_DIR / "QFF_30min_day.csv"
URL = "https://api.finmindtrade.com/api/v4/data"
FUTURES_ID = "QFF"
SESSION_START = "08:45:00"
SESSION_END = "13:45:00"
FIELDS = ["datetime", "contract_date", "open", "high", "low", "close", "volume"]


def token():
    for line in open(BASE / ".env.local"):
        if line.startswith("FINMIND_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no FINMIND_API_TOKEN")


def fetch_ticks(day, tok):
    params = {"dataset": "TaiwanFuturesTick", "data_id": FUTURES_ID,
              "start_date": day, "end_date": day, "token": tok}
    url = URL + "?" + urllib.parse.urlencode(params)
    body = json.loads(urllib.request.urlopen(url, timeout=90).read())
    if body.get("status") != 200:
        raise RuntimeError(body.get("msg", "err"))
    return body.get("data", [])


def bars_for_day(ticks):
    """近月連續 + 日盤 30 分 K。回傳 list[dict]（10 根內，無資料則空）。"""
    # 只留日盤，且排除價差(calendar spread)合約（contract_date 形如 202501/202502）
    day_ticks = [t for t in ticks
                 if SESSION_START <= t["date"][11:19] <= SESSION_END
                 and "/" not in t["contract_date"]]
    if not day_ticks:
        return []
    # 近月 = 當日日盤量最大的 contract_date
    vol_by_c = {}
    for t in day_ticks:
        vol_by_c[t["contract_date"]] = vol_by_c.get(t["contract_date"], 0) + int(t["volume"])
    near = max(vol_by_c, key=vol_by_c.get)
    ticks_n = [t for t in day_ticks if t["contract_date"] == near]
    ticks_n.sort(key=lambda t: t["date"])

    # 分 30 分桶（origin 08:45）
    buckets = {}
    for t in ticks_n:
        ts = datetime.strptime(t["date"], "%Y-%m-%d %H:%M:%S")
        origin = ts.replace(hour=8, minute=45, second=0, microsecond=0)
        b = int((ts - origin).total_seconds() // 1800)
        b = min(b, 9)                      # 13:45:00 併入最後一根
        buckets.setdefault(b, []).append((ts, float(t["price"]), int(t["volume"])))

    rows = []
    for b in sorted(buckets):
        items = buckets[b]
        origin = items[0][0].replace(hour=8, minute=45, second=0, microsecond=0)
        bar_start = origin + timedelta(minutes=30 * b)
        prices = [p for _, p, _ in items]
        rows.append({
            "datetime": bar_start.strftime("%Y-%m-%d %H:%M"),
            "contract_date": near,
            "open": prices[0], "high": max(prices), "low": min(prices),
            "close": prices[-1],
            "volume": sum(v for _, _, v in items),
        })
    return rows


def last_cached_date():
    if not OUT.exists():
        return None
    last = None
    with open(OUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            last = r["datetime"][:10]
    return last


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = token()

    args = sys.argv[1:]
    if len(args) >= 2:
        start, end = args[0], args[1]
    else:
        last = last_cached_date()
        start = (datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)).isoformat() \
            if last else "2025-01-01"
        end = date.today().isoformat()

    new_file = not OUT.exists()
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    print(f"補 QFF 日盤 30 分 K：{start} ~ {end}")

    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        d = d0
        days_with = 0
        bars_total = 0
        errs = 0
        while d <= d1:
            if d.weekday() >= 5:           # 跳週末
                d += timedelta(days=1)
                continue
            ds = d.isoformat()
            try:
                ticks = fetch_ticks(ds, tok)
                rows = bars_for_day(ticks)
                if rows:
                    w.writerows(rows)
                    f.flush()
                    days_with += 1
                    bars_total += len(rows)
            except Exception as e:
                errs += 1
                if errs <= 10:
                    print(f"  {ds}: {e}")
            if d.day == 1 or days_with % 20 == 0:
                print(f"  ...{ds}  交易日 {days_with}  K棒 {bars_total}  錯誤 {errs}")
            time.sleep(1.0)
            d += timedelta(days=1)

    print(f"完成：交易日 {days_with}，30分K {bars_total} 根，錯誤 {errs}")
    print(f"輸出：{OUT}")


if __name__ == "__main__":
    main()
