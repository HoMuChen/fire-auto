"""抓 v2 基本面/籌碼：月營收、日本益比/淨值比/殖利率（預設），或 --kinds inst,margin（三大法人 2012-05 起、融資券 2009 起）、
fin,bs（季報）、hold,mv,cf（集保股權分散表 2010 起：2010-14 月頻、2015 起週頻／日市值 2010 起／現金流量表 2012Q1 起）、
div（股利政策與除權息日）。

輸出 data/v2/revenue/{id}.parquet、data/v2/per/{id}.parquet（空表也存；已存在跳過＝斷點續抓）。
月營收前視防護：create_time 舊資料多為空 → 一律視為「營收月份次月 10 日收盤後」才可用（法規截止日）。
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_data_v2 import OUT, RateLimiter, api  # noqa: E402

ALL_JOBS = {"revenue": "TaiwanStockMonthRevenue", "per": "TaiwanStockPER",
            "inst": "TaiwanStockInstitutionalInvestorsBuySell", "margin": "TaiwanStockMarginPurchaseShortSale",
            "fin": "TaiwanStockFinancialStatements", "bs": "TaiwanStockBalanceSheet",
            "hold": "TaiwanStockHoldingSharesPer", "mv": "TaiwanStockMarketValue", "cf": "TaiwanStockCashFlowsStatement",
            "div": "TaiwanStockDividend"}
JOBS = {k: ALL_JOBS[k] for k in (sys.argv[sys.argv.index("--kinds") + 1].split(",") if "--kinds" in sys.argv else ["revenue", "per"])}


def fetch(lim, sid, kind):
    f = OUT / kind / f"{sid}.parquet"
    if f.exists():
        return "skip"
    df = api(lim, dataset=JOBS[kind], data_id=sid, start_date="2009-01-01")
    if df is None:
        return "fail"
    df.to_parquet(f, index=False)
    return "ok" if len(df) else "empty"


if __name__ == "__main__":
    lim = RateLimiter(int(sys.argv[sys.argv.index("--rate") + 1]) if "--rate" in sys.argv else 85)
    u = pd.read_csv(OUT / "universe.csv", dtype=str)
    if "--no-emerging" in sys.argv:
        u = u[u.type != "emerging"]
    for k in JOBS:
        (OUT / k).mkdir(exist_ok=True)
    jobs = [(s, k) for k in JOBS for s in u.stock_id]
    stats = {}
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(lambda j: fetch(lim, *j), jobs)):
            stats[r] = stats.get(r, 0) + 1
            if i % 400 == 0:
                print(f"[{i}/{len(jobs)}] {stats}", flush=True)
    print("完成", stats, flush=True)
