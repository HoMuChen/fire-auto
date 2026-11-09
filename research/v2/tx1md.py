"""台指期（TX）日盤 1 分 K ＋ tick rule 主動買賣（第七十五批 H188）：FinMind TaiwanFuturesTick 逐日抓逐筆 → 每分鐘彙總。

輸出 data/v2/futures/tx1md/{YYYY-MM-DD}.parquet（一天一檔、已存在跳過＝斷點續抓；休市日存空表）：
  contract、minute（K 棒起始，台北時間）、open/high/low/close、volume、buy、sell（口）
- 只留日盤 08:45~13:45（13:45 收盤撮合併入 13:44 那根）、排除價差單，只留當天日盤成交量最大的合約。
- FinMind volume 是買賣雙邊合計 → ÷2 成口數。
- tick rule：價 > 上一筆＝主動買、< ＝主動賣、＝ 沿用上一筆方向；當天第一筆（開盤集合競價）與 13:45:00 收盤集合競價不分類。
  FinMind 沒有內外盤欄位，這是推估。

用法：.venv/bin/python research/v2/tx1md.py fetch [起日] [迄日]（需網路）
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_data_v2 import RateLimiter, api  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data/v2/futures/tx1md"
COLS = ["contract", "minute", "open", "high", "low", "close", "volume", "buy", "sell"]


def to_bars(t):
    if t is None or not len(t):
        return pd.DataFrame(columns=COLS)
    t = t[~t.contract_date.astype(str).str.contains("/")].copy()
    t["ts"] = pd.to_datetime(t.date)
    hm = t.ts.dt.strftime("%H:%M:%S")
    t = t[(hm >= "08:45:00") & (hm <= "13:45:00")]
    if not len(t):
        return pd.DataFrame(columns=COLS)
    top = t.groupby("contract_date").volume.sum().idxmax()
    t = t[t.contract_date == top].sort_values("ts", kind="stable").reset_index(drop=True)
    t["volume"] = t.volume / 2
    s = np.sign(t.price.diff()).replace(0, np.nan).ffill().fillna(0)
    s.iloc[0] = 0
    s[t.ts.dt.strftime("%H:%M:%S") == "13:45:00"] = 0
    t["buy"] = t.volume.where(s > 0, 0.0)
    t["sell"] = t.volume.where(s < 0, 0.0)
    m = t.ts.dt.floor("min")
    t["minute"] = m.where(t.ts.dt.strftime("%H:%M") != "13:45", m - pd.Timedelta(minutes=1))
    b = t.groupby("minute").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
                                volume=("volume", "sum"), buy=("buy", "sum"), sell=("sell", "sum")).reset_index()
    b["contract"] = str(top)
    return b[COLS]


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
    return pd.concat([pd.read_parquet(f).assign(day=f.stem) for f in fs if f.stat().st_size], ignore_index=True)


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
                if i % 200 == 0:
                    print(f"[{i}/{len(days)}] {days[i]} {stats}", flush=True)
        print("完成", stats, flush=True)
