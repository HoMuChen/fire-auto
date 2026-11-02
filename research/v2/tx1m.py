"""台指期（TX）日盤 1 分 K 快取：FinMind TaiwanFuturesTick 逐日抓逐筆 → 1 分 K。

輸出 data/v2/futures/tx1m/{YYYY-MM-DD}.parquet（一天一檔、已存在跳過＝斷點續抓；休市日存空表）：
  contract（到期月，如 202011）、minute（K 棒起始，台北時間）、open/high/low/close、volume（口）
只留日盤 08:45~13:45（13:45:00 收盤撮合併入 13:44 那根）、排除價差單，保留當天日盤成交量前 2 大的合約
（近月與次月，結算日換月時讓前一天的同合約 K 棒可接上）。2011-01 起才有資料。

用法：/usr/local/bin/python3 research/v2/tx1m.py fetch [起日] [迄日]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_data_v2 import RateLimiter, api  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data/v2/futures/tx1m"
COLS = ["contract", "minute", "open", "high", "low", "close", "volume"]


def to_bars(t):
    if t is None or not len(t):
        return pd.DataFrame(columns=COLS)
    t = t[~t.contract_date.str.contains("/")].copy()
    t["ts"] = pd.to_datetime(t.date)
    hm = t.ts.dt.strftime("%H:%M:%S")
    t = t[(hm >= "08:45:00") & (hm <= "13:45:00")]
    if not len(t):
        return pd.DataFrame(columns=COLS)
    t["minute"] = t.ts.dt.floor("min").where(t.ts.dt.strftime("%H:%M") != "13:45", t.ts.dt.floor("min") - pd.Timedelta(minutes=1))
    top = t.groupby("contract_date").volume.sum().nlargest(2).index
    t = t[t.contract_date.isin(top)].sort_values(["contract_date", "ts"], kind="stable")
    b = t.groupby(["contract_date", "minute"]).agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                                   close=("price", "last"), volume=("volume", "sum")).reset_index()
    return b.rename(columns={"contract_date": "contract"})[COLS]


def fetch(lim, d):
    f = OUT / f"{d}.parquet"
    if f.exists():
        return "skip"
    t = api(lim, dataset="TaiwanFuturesTick", data_id="TX", start_date=d, end_date=d)
    if t is None:
        return "fail"
    b = to_bars(t)
    b.to_parquet(f, index=False)
    return "ok" if len(b) else "empty"


def load(p0="2011-01-01", p1="2099-12-31"):
    fs = sorted(f for f in OUT.glob("*.parquet") if p0 <= f.stem <= p1)
    return pd.concat([pd.read_parquet(f).assign(day=f.stem) for f in fs], ignore_index=True)


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        a = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2011, 1, 3)
        b = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date.today() - timedelta(days=1)
        OUT.mkdir(parents=True, exist_ok=True)
        days = [(a + timedelta(k)).isoformat() for k in range((b - a).days + 1) if (a + timedelta(k)).weekday() < 5]
        lim, stats = RateLimiter(90), {}
        with ThreadPoolExecutor(4) as ex:
            for i, r in enumerate(ex.map(lambda d: fetch(lim, d), days)):
                stats[r] = stats.get(r, 0) + 1
                if i % 100 == 0:
                    print(f"[{i}/{len(days)}] {days[i]} {stats}", flush=True)
        print("完成", stats, flush=True)
