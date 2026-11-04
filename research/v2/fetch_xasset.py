"""第六十四批資料（PORTFOLIO_PLAN P4 跨資產）：
- 美國 ETF 日 K（FinMind USStockPrice，含 Adj_Close）：SPY、TLT、IEF、GLD → data/v2/us/（2009–2023，當 2011 年起的代理）
- 台灣上市的對應 ETF（TaiwanStockPriceAdj 還原、TaiwanStockPrice 原始含成交量）：00646、00679B、00697B、00635U → data/v2/twetf/
只抓到 2023-12-31（holdout 不抓）。用法：.venv/bin/python research/v2/fetch_xasset.py（要開網路）
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_data_v2 import OUT, RateLimiter, api  # noqa: E402

US = ["SPY", "TLT", "IEF", "GLD"]
TW = ["00646", "00679B", "00697B", "00635U"]
YEARS = range(2009, 2024)


def main():
    lim = RateLimiter(60)
    (OUT / "us").mkdir(parents=True, exist_ok=True)
    (OUT / "twetf").mkdir(parents=True, exist_ok=True)
    for t in US:
        parts = [api(lim, dataset="USStockPrice", data_id=t, start_date=f"{y}-01-01", end_date=f"{y}-12-31") for y in YEARS]
        df = pd.concat([x for x in parts if x is not None and len(x)], ignore_index=True).drop_duplicates("date").sort_values("date")
        df.to_parquet(OUT / "us" / f"{t}.parquet")
        print(t, len(df), df.date.min(), df.date.max(), flush=True)
    for t in TW:
        for kind, ds in (("adj", "TaiwanStockPriceAdj"), ("raw", "TaiwanStockPrice")):
            df = api(lim, dataset=ds, data_id=t, start_date="2015-01-01", end_date="2023-12-31")
            if df is None or not len(df):
                print(t, kind, "空", flush=True)
                continue
            df = df.drop_duplicates("date").sort_values("date")
            df.to_parquet(OUT / "twetf" / f"{t}_{kind}.parquet")
            print(t, kind, len(df), df.date.min(), df.date.max(), flush=True)


if __name__ == "__main__":
    main()
