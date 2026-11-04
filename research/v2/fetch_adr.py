"""第六十二批資料：台股 ADR 日 K（FinMind USStockPrice）與美元匯率（TaiwanExchangeRate）→ data/v2/us/。
只抓到 2023-12-31（holdout 不抓）。用法：.venv/bin/python research/v2/fetch_adr.py（要開網路）
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_data_v2 import OUT, RateLimiter, api  # noqa: E402

ADR = {"TSM": "2330", "UMC": "2303", "ASX": "2311→3711", "CHT": "2412"}
YEARS = range(2009, 2024)


def main():
    lim, out = RateLimiter(60), OUT / "us"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {t: dict(dataset="USStockPrice", data_id=t) for t in ADR} | {"usdtwd": dict(dataset="TaiwanExchangeRate", data_id="USD")}
    for name, kw in jobs.items():
        parts = [api(lim, start_date=f"{y}-01-01", end_date=f"{y}-12-31", **kw) for y in YEARS]
        df = pd.concat([x for x in parts if x is not None and len(x)], ignore_index=True).drop_duplicates("date").sort_values("date")
        df.to_parquet(out / f"{name}.parquet")
        print(name, len(df), df.date.min(), df.date.max(), flush=True)


if __name__ == "__main__":
    main()
